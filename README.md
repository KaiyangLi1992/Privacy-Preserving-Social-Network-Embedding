Adversarial Privacy Graph Embedding (APGE)
============

This is a TensorFlow implementation of the Adversarial Privacy Graph Embedding (APGE) model as described in our paper.

We borrowed part of code from T. N. Kipf, M. Welling, Variational Graph Auto-Encoders [https://github.com/tkipf/gae].

 <p align =“center”>
    <image  src=figure.png width=500 />
 </p>

## idea

In this work, we propose the privacy-preserving approach against inference attack which is the adversary try to infer privacy information based on released data. Like traditional works defending inference attack, we regard the sensitive attributes as privacy (i.e., sexual orientation, political opinion)[1-3]. That is, we want the graph embedding contains the information of the sensitive attributes as little as possible. Since most of the private information is removed from graph embedding, our approach is suit to defend all kinds of inference attack launched on graph embedding.


In our paper, we propose two mechanisms disentanglement and purging to extrude the private information. To evaluate the performance of preserving privacy, we train MLP/SVM to predict sensitive attributes. Indeed, it is not a very sound approach, because we cannot test all possible models. But in the classical works against inference attack, authors always use this approach to evaluate the preserving privacy model [1][2].  We have assumed attackers leverage other classifier launch attacks such as logistic regression, KNN, and random frost. In these cases, APGE significantly outperforms baselines in these classifiers too. Because of the lack of space, we do not show the results in the paper.

[1] Zhipeng Cai, Zaobo He, Xin Guan, and Yingshu Li. Collective data-sanitization for preventing sensitive information inference attacks in social networks. IEEE Transactions on Dependable and Secure Computing, 15(4):577–590, 2018.

[2]Jinyuan Jia and Neil Zhenqiang Gong. Attriguard: A practical defense against attribute inference attacks via adversarial machine learning. USENIX Security 18, pages 513–529, 2018.

[3] Ahmadinejad S H, Fong P W L. Unintended disclosure of information: Inference attacks by third-party extensions to Social Network Systems[J]. Computers \& security, 2014, 44: 75-91.


We introduce a "disentanglement" work supervised AAE in the last paragraph of Sec. 2.2. The supervised AAE inspires us to propose APDGE. 






## Requirements
* TensorFlow (2.20 or later)
* python 3.5
* scikit-learn
* scipy
* numpy


## Run the demo


train the model and predict utility and private attribute
```
python train.py -dataset rochester
```

predict linkage
```
python link_predict.py -dataset rochester
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



