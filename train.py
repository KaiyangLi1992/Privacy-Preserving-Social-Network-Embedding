#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function

import sys
import pickle as pkl
import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp
from input_data import load_data
from meansuring import get_score
from preprocessing import preprocess_graph,sparse_to_tuple,construct_feed_dict
from constructor import get_placeholder, get_model, get_optimizer, update
from process_attr import get_attr_list
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
# Settings
flags.DEFINE_string('f', '', 'Kernel')
flags.DEFINE_string('dataset', 'yale', 'Name of dateset')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in GCN layer 1.')
flags.DEFINE_integer('hidden3', 32, 'Number of units in discriminator layer 1.')
flags.DEFINE_integer('hidden4', 16, 'Number of units in discriminator layer 2.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')



    
if FLAGS.dataset=='yale':
    flags.DEFINE_integer('epochs', 500, 'Number of iterations.')
    flags.DEFINE_integer('hidden2', 16, 'Number of units in GCN layer 2.')
    flags.DEFINE_integer('pri_weight', 1,'weight of privacy')
    flags.DEFINE_integer('uti_attr_weight', 10,'weight of utility_attr')
    flags.DEFINE_float('link_weight', 1,'weight of privacy')
elif FLAGS.dataset=='rochester':
    flags.DEFINE_integer('epochs', 2000, 'Number of iterations.')
    flags.DEFINE_integer('pri_weight', 10,'weight of privacy')
    flags.DEFINE_integer('uti_attr_weight', 1,'weight of utility_attr')
    flags.DEFINE_integer('hidden2', 8, 'Number of units in GCN layer 2.')
    flags.DEFINE_float('link_weight', 1,'weight of privacy')
 


# Load data
adj, features,adj_train, val_edges, val_edges_false, test_edges, test_edges_false,labels = load_data(FLAGS.dataset)


# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


adj = adj_train

# Some preprocessing
adj_norm = preprocess_graph(adj)
features_mat = features.toarray()
attr_labels_list,dim_attr,features_rm_privacy = get_attr_list(FLAGS.dataset,labels,features_mat)


features_lil = sp.lil_matrix(features_rm_privacy)
features_tuple = sparse_to_tuple(features_lil .tocoo())
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

placeholders = get_placeholder(adj)
d_real, discriminator, ae_model = get_model(placeholders, num_features, 
                                            num_nodes, features_nonzero,attr_labels_list[-1],dim_attr)
opt = get_optimizer(ae_model, discriminator, placeholders, pos_weight, norm, d_real, 
                            num_nodes,attr_labels_list)


# In[3]:


#train model
preds_all = None
labels_all = None
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(FLAGS.epochs):
    emb,emb_long,avg_cost,attr_loss,pri_loss,link_loss = update(ae_model, opt, sess,adj_norm,adj_label,features_tuple, placeholders, adj)
    #Compute score of validation set  
    if ((epoch+1)%100==0):
        
        p0_mlp,p0_f1,p2_mlp,p2_mlp_f1,p2_svm,p2_svm_f1 = get_score(FLAGS.dataset,adj_orig,test_edges,test_edges_false,emb)
        print('Epoch: ' + str(epoch+1) +'\n')
        print('Utility Attr MLP ACC: ' + str(p0_mlp)+'\n')
        print('Utility Attr MLP F1: ' + str(p0_f1)+'\n')
        print('Privacy MLP ACC: ' + str(p2_mlp)+'\n')
        print('Privacy MLP F1: ' + str(p2_mlp_f1)+'\n')
        print('Privacy SVM ACC: ' + str(p2_svm)+'\n')
        print('Privacy SVM F1: ' + str(p2_svm_f1)+'\n')
print("Optimization Finished!")


# In[ ]:


# Compute score of test set 
# Save embedding result 
emb,emb_long,avg_cost,attr_loss,pri_loss,link_loss = update(ae_model, opt, sess,adj_norm,adj_label,features_tuple, placeholders, adj)
p0_mlp,p0_f1,p2_mlp,p2_mlp_f1,p2_svm,p2_svm_f1 = get_score(FLAGS.dataset,adj_orig,test_edges,test_edges_false,emb)

print('Utility Attr MLP ACC: ' + str(p0_mlp)+'\n')
print('Utility Attr MLP F1: ' + str(p0_f1)+'\n')
print('Privacy MLP ACC: ' + str(p2_mlp)+'\n')
print('Privacy MLP F1: ' + str(p2_mlp_f1)+'\n')
print('Privacy SVM ACC: ' + str(p2_svm)+'\n')
print('Privacy SVM F1: ' + str(p2_svm_f1)+'\n')


# In[ ]:


folder = './data/'+ FLAGS.dataset+ '_' 
np.save(folder+'emb.npy',emb )
f = open(folder+'results.txt','a')
f.write( str(p0_mlp)+'\n')
f.write(str(p0_f1)+'\n')
f.write(str(p2_mlp)+'\n')
f.write(str(p2_mlp_f1)+'\n')
f.write(str(p2_svm)+'\n')
f.write(str(p2_svm_f1)+'\n')
f.close()

