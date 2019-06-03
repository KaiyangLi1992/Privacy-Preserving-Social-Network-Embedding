import pickle as pkl
import scipy.sparse as sp

def load_data():
    with open('../data/yale_feats_new.pkl', 'rb') as f1:
        features = pkl.load(f1)
    with open('../data/yale_adj.pkl', 'rb') as f2:
        adj = pkl.load(f2,encoding='latin1')
    return adj, features