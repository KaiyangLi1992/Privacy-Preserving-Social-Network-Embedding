import tensorflow as tf
import numpy as np
from model import APGE, Discriminator
from optimizer import OptimizerAE
import scipy.sparse as sp
from input_data import load_data
import inspect
from preprocessing import preprocess_graph, sparse_to_tuple, construct_feed_dict
import random
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2],
                                            name='real_distribution'),
        'sample': tf.placeholder(tf.float32)

    }

    return placeholders


def get_model(placeholders, num_features, num_nodes, features_nonzero,gender_attr):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    model = APGE(placeholders, num_features, features_nonzero,gender_attr)
    return d_real, discriminator, model



def get_optimizer(model, discriminator, placeholders, pos_weight, norm, d_real,num_nodes,attr_labels_list):

    d_fake = discriminator.construct(model.embeddings, reuse=True)
#         pred_attrs=[model.attr0_logits,model.attr1_logits,model.attr2_logits,
#                     model.attr3_logits,model.attr4_logits]
    pred_attrs = [model.attr0_logits,model.attr1_logits,model.attr2_logits]

    opt = OptimizerAE(preds=model.reconstructions,
                      labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],validate_indices=False), [-1]),
                      pos_weight=pos_weight,
                      pred_attrs = pred_attrs,
                      norm=norm,
                      d_real=d_real,
                      d_fake=d_fake,
                      attr_labels_list=attr_labels_list,
                      sample_list=model.sample)
    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary
    sampled_id = np.zeros((features[2][0],1))
    resultList=random.sample(range(features[2][0]),256)
    for i in resultList:
        sampled_id[i] = 1;
    
            
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['sample']: sampled_id})
    emb_concat,emb_long = sess.run([model.embeddings_concat,model.embeddings_long], feed_dict=feed_dict)

    z_real_dist = np.random.randn(adj.shape[0], FLAGS.hidden2)
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.O_opt_op, opt.cost], feed_dict=feed_dict)
        for m in range(10):
            _, pri_loss = sess.run([opt.A_opt_op, opt.attr2_loss], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    AE_cost = sess.run(opt.cost, feed_dict=feed_dict)

    return emb_long,emb_concat, AE_cost


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]