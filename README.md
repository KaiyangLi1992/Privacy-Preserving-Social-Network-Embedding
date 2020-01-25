Adversarial Privacy Graph Embedding (APGE)
============

This is a TensorFlow implementation of the Adversarial Privacy Graph Embedding (APGE) model as described in our paper.

We borrowed part of code from T. N. Kipf, M. Welling, Variational Graph Auto-Encoders [https://github.com/tkipf/gae].



## Requirements
* TensorFlow (1.8.0 or later)
* python 3.5
* scikit-learn
* scipy
* numpy


## Run the demo


train the model and predict utility and private attribute
```
python train.py -dataset yale
```

predict linkage
```
python link_predict.py -dataset yale 
```



## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), such as ./data/yale_adj.pkl
* an N by D feature matrix (D is the number of features per node), such as ./data/yale_feats.pkl
* an N by M attribute matrix (M is the number of attibutes per node), such as ./data/yale_label.npy

Here, the i-th row of feature matrix is the concatenation of i-th user’s every attribute one-hot vector. And the element in the i-th row and j-th column of attribute matrix is the label of i-th user's j-th attribute.

In yale_feats.pkl, elements in columns 0 - 4 correspond to the utility attribute student/faculty status, and elements in the bottom 6 columns correspond to class year, which is privacy here. 
In rochester.pkl, elements in the bottom 19 columns correspond to the utility attribute class year, and elements in the 6,7 columns correspond to gender, which is privacy here.

In yale_label.npy, elements in the first cloumn  correspond to student/faculty status, and elements in the last column correspond to class year.
In rochester_label.npy, elements in the last cloumn  correspond to student/faculty status, and elements in the seconed column correspond to gender.

And You could use mask_test_edges.py to preprocess adjacency matrix to get test set and validation set.



