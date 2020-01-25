import tensorflow.compat.v1 as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake,pred_attrs,attr_labels_list,sample_list):
        attr_preds_list = pred_attrs
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real,name='dclreal'))

        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake,name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))



        self.link_cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        
        
        self.attr_loss = tf.losses.softmax_cross_entropy(logits= tf.cast(attr_preds_list[0],tf.float32),
                                                          onehot_labels=attr_labels_list[0],
                                                          reduction = tf.losses.Reduction.NONE)
        mask_attr = np.sum(attr_labels_list[0],axis = 1)
        self.attr_loss = tf.multiply(self.attr_loss,sample_list)
        self.attr_loss = tf.reduce_mean(tf.multiply(self.attr_loss,mask_attr))

        
        
        
        self.pri_loss = tf.losses.softmax_cross_entropy(logits= tf.cast(attr_preds_list[1],tf.float32),
                                                             onehot_labels=attr_labels_list[1],
                                                         reduction = tf.losses.Reduction.NONE)
        mask_attr = np.sum(attr_labels_list[1],axis = 1)
        self.pri_loss = tf.multiply(self.pri_loss,sample_list)
        self.pri_loss = tf.reduce_mean(tf.multiply(self.pri_loss,mask_attr))
        
        self.attr_cost =  FLAGS.uti_attr_weight*self.attr_loss - (FLAGS.pri_weight * self.pri_loss)
        
        self.cost = self.attr_cost + FLAGS.link_weight * self.link_cost
        
        
        
        
        self.generator_loss = generator_loss + self.cost
        

        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]
        pri_var = [var for var in all_variables if 'pri_' in var.name]
        all_rm_pri = [x for x in all_variables if x not in pri_var]
        
        self.O_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.O_opt_op = self.O_optimizer.minimize(self.cost,var_list=all_rm_pri)
        #self.O_grads_vars = self.O_optimizer.compute_gradients(self.generator_loss)
        
        self.A_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.A_opt_op = self.A_optimizer.minimize(self.pri_loss,var_list=pri_var)
        #self.A_grads_vars = self.A_optimizer.compute_gradients(self.attr2_loss)

      
        with tf.variable_scope(tf.get_variable_scope()):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                             beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var) #minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)





