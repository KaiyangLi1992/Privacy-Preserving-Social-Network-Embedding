Adversarial Privacy Graph Embedding (APGE)
============

This is a TensorFlow implementation of the Adversarial Privacy Graph Embedding (APGE) model as described in our paper.

We borrowed part of code from T. N. Kipf, M. Welling, Variational Graph Auto-Encoders [https://github.com/tkipf/gae] and 
Shirui Pan .et al, Adversarially Regularized Graph Autoencoder for Graph Embedding [https://github.com/Ruiqi-Hu/ARGA]


## Installation

```bash
pip install -r requirements.txt
```

## Requirements
* TensorFlow (1.0 or later)
* python 2.7
* networkx
* scikit-learn
* scipy

## Run from

```bash
python run.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `load_data()` function in `input_data.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid

## Models

You can choose between the following models: 
* `arga_ae`: Adversarially Regularised Graph Auto-Encoder
* `arga_vae`: Adversarially Regularised Variational Graph Auto-Encoder 

## Cite

Please cite following papers if you use this code in your own work:

```
@inproceedings{pan2018adversarially,
  title={Adversarially Regularized Graph Autoencoder for Graph Embedding.},
  author={Pan, Shirui and Hu, Ruiqi and Long, Guodong and Jiang, Jing and Yao, Lina and Zhang, Chengqi},
  booktitle={IJCAI},
  pages={2609--2615},
  year={2018}
}
```
