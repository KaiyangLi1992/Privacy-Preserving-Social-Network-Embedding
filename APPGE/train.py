
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function


import pickle as pkl
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from input_data import load_data
from meansuring import get_score
from preprocessing import load_edges,preprocess_graph,sparse_to_tuple,construct_feed_dict
from optimizer import OptimizerAE

# Train on CPU (hide GPU) due to memory constraints
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('dataset', 'rochester', 'Name of dateset')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in GCN layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in GCN layer 2.')


if FLAGS.dataset=='yale':
    flags.DEFINE_integer('epochs', 2000, 'Number of iterations.')
elif FLAGS.dataset=='rochester':
    flags.DEFINE_integer('epochs', 5000, 'Number of iterations.')



# Load data
adj, features,adj_train, val_edges, val_edges_false, test_edges, test_edges_false = load_data(FLAGS.dataset)



# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

#adj_train, val_edges, val_edges_false, test_edges, test_edges_false = load_edges() 
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)




## Remove privacy attributes from feature matrix 
# Bulid attibute labels
if FLAGS.dataset == 'yale':
    #On Yale, elements in columns 0 - 4 correspond to student/faculty status, 
    #elements in columns 5,6 correspond to gender,
    #and elements in  the bottom 6 columns correspond to class year,which is privacy here.
    features_mat = features.toarray()
    attr0_labels = features_mat[:,0:5]
    attr1_labels = features_mat[:,5:7]
    privacy_labels = features_mat[:,-6:]
    attr_labels_list = [attr0_labels,attr1_labels,privacy_labels]
    features_rm_privacy =features_mat[:,:-6]
    features_lil = sp.lil_matrix(features_rm_privacy)
    features_tuple = sparse_to_tuple(features_lil .tocoo())
elif FLAGS.dataset == 'rochester':
    #On Rochester, elements in columns 0 - 5 correspond to student/faculty status, 
    #elements in the bottom 19 columns correspond to class year,
    #and elements in  the bottom 6,7 columns correspond to gender,which is privacy here.
    features_mat = features.toarray()
    attr0_labels = features_mat[:,0:6]
    attr1_labels = features_mat[:,-19:]
    privacy_labels = features_mat[:,6:8]
    attr_labels_list = [attr0_labels,attr1_labels,privacy_labels]
    features_rm_privacy =np.hstack((features_mat[:,:6],features_mat[:,8:]))
    features_lil = sp.lil_matrix(features_rm_privacy)
    features_tuple = sparse_to_tuple(features_lil .tocoo())
    
#Calculate the dimensions of each attribute
dim_attr = [attr0_labels.shape[1], attr1_labels.shape[1], privacy_labels.shape[1]]


num_nodes = adj.shape[0]
features_sp = sparse_to_tuple(features_lil.tocoo())
num_features = features_sp[2][1]
features_nonzero = features_sp[1].shape[0]






# In[2]:


# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'sample': tf.placeholder(tf.float32)
}
# Create model

from model import APPGE
model = APPGE(placeholders, num_features, features_nonzero,dim_attr)




pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
attr_preds_list = [model.attr0_preds, model.attr1_preds,model.privacy_preds]


# Optimizer
with tf.name_scope('optimizer'):
    opt_O = OptimizerAE(OorA = 'obfuscator',
                        link_preds=model.reconstructions,
                        link_labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                  validate_indices=False), [-1]),
                        attr_preds_list=attr_preds_list,
                        attr_labels_list= attr_labels_list,
                        sample_list = model.sample,
                        pos_weight=pos_weight,
                        norm=norm)
    opt_A = OptimizerAE(OorA = 'attacker',
                        link_preds=model.reconstructions,
                        link_labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                  validate_indices=False), [-1]),
                        attr_preds_list=attr_preds_list,
                        attr_labels_list= attr_labels_list,
                        sample_list = model.sample,
                        pos_weight=pos_weight,
                        norm=norm)


# In[3]:


# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
import random

for epoch in range(FLAGS.epochs):
    # Trian Obfuscator
    for epoch_o in range(1):
        # Sample nodes to implement batchsize
        sampled_id=np.zeros((features_mat.shape[0],1))
        resultList=random.sample(range(0,features_mat.shape[0]),256);
        for i in resultList:
            sampled_id[i] = 1;
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features_sp, placeholders)
        feed_dict.update({placeholders['sample']: sampled_id})
        
        outs_A = sess.run([opt_A.A_opt_op, 
                           opt_A.A_cost], feed_dict=feed_dict)




    # Trian Attacker
    for epoch_a in range(10):  
        # Sample nodes to implement batchsize
        sampled_id=np.zeros((features_mat.shape[0],1))
        resultList=random.sample(range(0,features_mat.shape[0]),256);
        for i in resultList:
            sampled_id[i] = 1;
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features_sp, placeholders)
        feed_dict.update({placeholders['sample']: sampled_id})
        outs_O = sess.run([opt_O.O_opt_op, 
                           opt_O.O_cost], feed_dict=feed_dict)
        
    # Compute score of validation set   
    if (epoch+1)%200 ==0:
        z_embedding = sess.run(model.embeddings, feed_dict=feed_dict)
        roc_score,ap_score,p0_mlp,p1_mlp,p2_lr,p2_svm,p2_mlp = get_score(FLAGS.dataset,adj_orig,val_edges, val_edges_false,z_embedding)
        print('Epoch:' + str(epoch))
        print('Val Link ROC: ' + str(roc_score) +'\n')
        print('Val Link AP: ' + str(ap_score)+'\n')
        print('Attr0 MLP: ' + str(p0_mlp)+'\n')
        print('Attr1 MLP: ' + str(p1_mlp)+'\n')
        print('Pri LR: ' + str(p2_lr)+'\n')
        print('Pri MLP: ' + str(p2_mlp)+'\n')
        print('Pri SVM: ' + str(p2_svm)+'\n')

print("Optimization Finished!")


# In[ ]:


# Compute score of test set 
# Save embedding result 
z_embedding = sess.run(model.embeddings, feed_dict=feed_dict)
roc_score,ap_score,p0_mlp,p1_mlp,p2_lr,p2_svm,p2_mlp = get_score(FLAGS.dataset,adj_orig,test_edges, test_edges_false,z_embedding)
np.save('./data/APPGE_{}_embedding.npy'.format(FLAGS.dataset),z_embedding)
print('Test Results')
print('Test Link ROC: ' + str(roc_score) +'\n')
print('Test Link AP: ' + str(ap_score)+'\n')
print('Attr0 MLP: ' + str(p0_mlp)+'\n')
print('Attr1 MLP: ' + str(p1_mlp)+'\n')
print('Pri LR: ' + str(p2_lr)+'\n')
print('Pri MLP: ' + str(p2_mlp)+'\n')
print('Pri SVM: ' + str(p2_svm)+'\n')


# In[ ]:




