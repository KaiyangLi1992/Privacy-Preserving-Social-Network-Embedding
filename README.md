Adversarial Privacy Graph Embedding (APGE)
============

This is a TensorFlow implementation of the Adversarial Privacy Graph Embedding (APGE) model as described in our paper.

We borrowed part of code from T. N. Kipf, M. Welling, Variational Graph Auto-Encoders [https://github.com/tkipf/gae] and 
Shirui Pan .et al, Adversarially Regularized Graph Autoencoder for Graph Embedding [https://github.com/Ruiqi-Hu/ARGA].



## Requirements
* TensorFlow (1.8.0 or later)
* python 3.5
* networkx
* scikit-learn
* scipy
* numpy


## Run the demo

```bash
cd APGE
python train.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node)
* an N by M attribute matrix (M is the number of attibutes per node)

Here, the i-th row of feature matrix is the concatenation of i-th user’s every attribute one-hot vector. And the element in the i-th row and j-th column of attribute matrix is the label of i-th user's j-th attribute.

And You should use mask_test_edges.py to preprocess adjacency matrix to get test set and validation set.


## Models

You can choose between the following models: 
* `APGE`: Adversarial Privacy Graph Embedding
* `ADPGE`: Adversarial Privacy-Disentangled Graph Embedding
* `APPGE`: Adversarially Privacy-Purged Variational Graph Auto-Encoder 

