#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import pickle as pkl
import scipy.sparse as sp
from input_data import load_data
import os
import tensorflow.compat.v1 as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# In[2]:


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)


# In[3]:


def pred_link(dataset,epochs):
    #load samples
    
    adj, features,adj_train, val_edges, val_edges_false, test_edges, test_edges_false,labels = load_data(dataset)
    adj_tuple = sparse_to_tuple(adj)
    adj_train_tuple = sparse_to_tuple(adj_train)
    train_edges_false = np.load('./data/'+dataset+'_train_edges_false.npy')
    train_all_edges = np.concatenate((adj_train_tuple[0],train_edges_false),axis=0) 
    labels=np.zeros(train_all_edges.shape)
    labels[:int(train_all_edges.shape[0]/2),0]=1
    labels[int(train_all_edges.shape[0]/2):,1]=1
    permutation = np.random.permutation(train_all_edges.shape[0])
    train_all_edges = train_all_edges[permutation,:]
    labels = labels[permutation,:]
    
    #load_embeddings
    emb = np.load('./data/'+dataset+'_emb.npy')

    tf.compat.v1.disable_eager_execution()
    x1 = tf.placeholder('float', [None, 64])
    x2 = tf.placeholder('float', [None, 64])
    y = tf.placeholder('float', [None, 2])


    x11 = tf.nn.relu(tf.layers.dense(inputs=x1, units=32))
    x21 = tf.nn.relu(tf.layers.dense(inputs=x2, units=32))
    x31= tf.concat([x11, x21], 1)
    x41 = tf.nn.relu(tf.layers.dense(inputs=x31, units=16))
    x4 = tf.nn.relu(tf.layers.dense(inputs=x41, units=8))
    preds = tf.layers.dense(inputs=x4, units=2)
    cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=preds,multi_class_labels=y))   

    sess =  tf.Session()


    train_op = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess.run(init)
    flag = 0



    for epoch in range(epochs):
        if flag*100+100 > train_all_edges.shape[0]:
            flag = 0
        a = flag*100
        b = a+100
        flag = flag+1
        batch_edges = train_all_edges[a:b,:]
        batch_y = labels[a:b]
        batch_x1 = emb[batch_edges[:,0],:]
        batch_x2 = emb[batch_edges[:,1],:]
        _,loss,preds_ = sess.run([train_op, cross_entropy,preds], feed_dict={x1: batch_x1,x2:batch_x2,y:batch_y})

#         if epoch%1000 == 0:
#             print(epoch)

    test_all_edges = np.concatenate((test_edges, test_edges_false),axis=0) 
    test_labels=np.zeros(test_all_edges.shape)
    test_labels[:int(test_all_edges.shape[0]/2),0]=1
    test_labels[int(test_all_edges.shape[0]/2):,1]=1
    test_preds=np.empty((0,2))
    flag=0
    for epoch in range(int(test_all_edges.shape[0]/100)):
        if flag*100+100 > test_all_edges.shape[0]:
            flag = 0
        a = flag*100
        b = a+100
        flag = flag+1
        batch_edges = test_all_edges[a:b,:]
        batch_y = test_labels[:100,:]
        batch_x1 = emb[batch_edges[:,0],:]
        batch_x2 = emb[batch_edges[:,1],:]
        batch_preds = sess.run(preds, feed_dict={x1: batch_x1,x2:batch_x2,y:batch_y})
        test_preds = np.vstack((test_preds,batch_preds))
    test_preds.shape
    test_labels = test_labels[:int((test_all_edges.shape[0])/100)*100,:]
    #p = np.where(test_preds>0)[1]
    p=[]
    for label in test_preds:
        if label[0]>=label[1]:
            p.append(0)
        else:
            p.append(1)
        
    l = test_labels[:,1]
    from sklearn.metrics import f1_score,accuracy_score
    acc = accuracy_score(l, p) 
    f1 = f1_score(l,p,average='macro') 
    print(acc)
    print(f1)

    f = open('./data/'+dataset +'_results.txt', 'r+')
    content = f.read()
    f.seek(0, 0)
    f.write(str(acc)+'\n')
    f.write(str(f1)+'\n'+content)
    f.close()
    return acc,f1


# In[4]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


# In[5]:


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
dataset = FLAGS.dataset 
if dataset == 'rochester':
    epochs=15000
else:
    epochs=25000
acc,f1 = pred_link(dataset,epochs)
print("ACC of link predction:"+ str(acc))
print("Macro F1 of link predction:"+ str(f1))

