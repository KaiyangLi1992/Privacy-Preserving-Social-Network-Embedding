import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

                        

class OptimizerAE(object):
    def __init__(self,OorA,link_preds,link_labels,attr_preds_list,attr_labels_list,sample_list,pos_weight,norm):
 

        self.attr0_loss = tf.losses.softmax_cross_entropy(logits= tf.cast(attr_preds_list[0],tf.float32),
                                                          onehot_labels=attr_labels_list[0],
                                                          reduction = tf.losses.Reduction.NONE)
        mask_attr = np.sum(attr_labels_list[0],axis = 1)
        self.attr0_loss = tf.multiply(self.attr0_loss,sample_list)
        self.attr0_loss = tf.reduce_mean(tf.multiply(self.attr0_loss,mask_attr))

        
        self.attr1_loss = tf.losses.softmax_cross_entropy(logits= tf.cast(attr_preds_list[1],tf.float32),
                                                          onehot_labels=attr_labels_list[1],
                                                          reduction = tf.losses.Reduction.NONE)

        mask_attr = np.sum(attr_labels_list[1],axis = 1)
        self.attr1_loss = tf.multiply(self.attr1_loss,sample_list)
        self.attr1_loss = tf.reduce_mean(tf.multiply(self.attr1_loss,mask_attr))


        self.pri_loss = tf.losses.softmax_cross_entropy(logits= tf.cast(attr_preds_list[2],tf.float32),
                                                          onehot_labels=attr_labels_list[2],
                                                          reduction = tf.losses.Reduction.NONE)
        mask_attr = np.sum(attr_labels_list[2],axis = 1)
        self.pri_loss = tf.multiply(self.pri_loss,sample_list)
        self.pri_loss = tf.reduce_mean(tf.multiply(self.pri_loss,mask_attr))



        self.attr_loss = self.attr0_loss + self.attr1_loss
        self.link_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=link_preds,targets=link_labels,pos_weight=pos_weight))   
        
        if OorA == 'obfuscator':     
            self.O_cost =  self.link_loss  + self.attr_loss- self.pri_loss
            self.O_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) 
            self.O_opt_op = self.O_optimizer.minimize(self.O_cost,var_list=list(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)).difference(set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="appge/privacy_classification")))))
            self.O_grads_vars = self.O_optimizer.compute_gradients(self.O_cost)


        if OorA == 'attacker':     
            self.A_cost = self.pri_loss 
            self.A_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
            var_list = []
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="appge/privacy_classification")
            self.A_opt_op = self.A_optimizer.minimize(self.A_cost,var_list=var_list)
            self.A_grads_vars = self.A_optimizer.compute_gradients(self.A_cost)