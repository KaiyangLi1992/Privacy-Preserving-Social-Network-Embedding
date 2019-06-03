import import_ipynb
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

    

class APPGE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(APPGE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.sample = placeholders['sample']
        self.build()


    def _build(self):
        layer1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)
        self.hidden1 = layer1(self.inputs)
        self.var_scope1 = layer1.name + '_vars'

        layer2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)

        self.embeddings =layer2(self.hidden1)
        self.var_scope2 = layer2.name + '_vars'


        self.z_mean = self.embeddings
        #print(self.z_mean)

        with tf.variable_scope("utility_classification"):

            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          logging=self.logging)(self.embeddings)
            self.attr0_preds = tf.layers.dense(inputs=self.embeddings, units=5)

            self.attr1_preds = tf.layers.dense(inputs=self.embeddings, units=2)


            #print(tf.get_variable_scope().name)


        with tf.variable_scope("privacy_classification"):
            #self.sensi_attr_ =  tf.layers.dense(inputs=self.embeddings, units=100)
            self.privacy_preds = tf.layers.dense(inputs=self.embeddings, units=6)