
# coding: utf-8

# In[6]:


from __future__ import division
from __future__ import print_function


import pickle as pkl
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from input_data import load_data
from meansuring import get_score
from preprocessing import load_edges,preprocess_graph,sparse_to_tuple,construct_feed_dict

# Train on CPU (hide GPU) due to memory constraints
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('f', '', 'Kernel')
flags.DEFINE_string('dataset', 'yale', 'Name of dateset')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in GCN layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in discriminator layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in discriminator layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('hidden2', 16, 'Number of units in GCN layer 2.')
flags.DEFINE_integer('epochs', 600, 'Number of iterations.')


# Load data
adj, features,adj_train, val_edges, val_edges_false, test_edges, test_edges_false = load_data(FLAGS.dataset)


# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# Bulid test set and valid set 
#adj_train, val_edges, val_edges_false, test_edges, test_edges_false = load_edges() 
adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)


# Remove privacy attributes from feature matrix 
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


pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = 1
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)


# In[2]:


# Define placeholders
from constructor import get_placeholder, get_model, get_optimizer, update
placeholders = get_placeholder(adj)
d_real, discriminator, ae_model = get_model(placeholders, num_features, 
                                            num_nodes, features_nonzero,privacy_labels,dim_attr)
opt = get_optimizer(ae_model, discriminator, placeholders, pos_weight, norm, d_real, 
                            num_nodes,attr_labels_list)


# In[3]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(FLAGS.epochs):
    emb,emb_long,avg_cost = update(ae_model, opt, sess,adj_norm,adj_label,features_tuple, placeholders, adj)
    #Compute score of validation set  
    if ((epoch+1)%200==0):
        roc_score,ap_score,p0_mlp,p1_mlp,p2_lr,p2_svm,p2_mlp = get_score(FLAGS.dataset,adj_orig,val_edges,val_edges_false,emb)
        print('Epoch: ' + str(epoch+1) +'\n')
        print('Val Link ROC: ' + str(roc_score) +'\n')
        print('Val Link AP: ' + str(ap_score)+'\n')
        print('Attr0 MLP: ' + str(p0_mlp)+'\n')
        print('Attr1 MLP: ' + str(p1_mlp)+'\n')
        print('Pri LR: ' + str(p2_lr)+'\n')
        print('Pri MLP: ' + str(p2_mlp)+'\n')
        print('Pri SVM: ' + str(p2_svm)+'\n')
print("Optimization Finished!")


# In[5]:


# Compute score of test set 
# Save embedding result 
roc_score,ap_score,p0_mlp,p1_mlp,p2_lr,p2_svm,p2_mlp = get_score(FLAGS.dataset,adj_orig,test_edges,test_edges_false,emb)
np.save('./data/ADPGE_{}_embedding.npy'.format(FLAGS.dataset),emb)
print('Test Link ROC: ' + str(roc_score) +'\n')
print('Test Link AP: ' + str(ap_score)+'\n')
print('Attr0 MLP: ' + str(p0_mlp)+'\n')
print('Attr1 MLP: ' + str(p1_mlp)+'\n')
print('Pri LR: ' + str(p2_lr)+'\n')
print('Pri MLP: ' + str(p2_mlp)+'\n')
print('Pri SVM: ' + str(p2_svm)+'\n')

