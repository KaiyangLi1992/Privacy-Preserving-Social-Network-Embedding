from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import  svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import tensorflow as tf




def get_score(dataset,adj_orig,edges_pos, edges_neg, emb=None):
    #load attribute matrix 
    attr_label = np.load( './data/{}_label.npy'.format(dataset))

    #In different dataset, we set different utility attribute and private attribute. 
    #Find the column index corresponding to each attibute.
    if (dataset == 'yale'):
        attr_index=[0,1,5]
    elif (dataset == 'rochester'):
        attr_index=[0,5,1]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    k_fold = KFold(5)
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    y = attr_label[:,attr_index[0]]
    filter_id = np.where(y==0)
    y = np.delete(y,filter_id,0)
    X_emb_feat0 = np.delete(emb,filter_id,0)
    clf = MLPClassifier(max_iter=500)
    prec = cross_val_score(clf, X_emb_feat0, y, cv=k_fold,n_jobs=-1)
    p0_mlp = sum(prec)/len(prec)

    y = attr_label[:,attr_index[1]]
    filter_id = np.where(y==0)
    y = np.delete(y,filter_id,0)
    X_emb_feat1 = np.delete(emb,filter_id,0)
    clf = MLPClassifier(max_iter=500)
    prec = cross_val_score(clf, X_emb_feat1, y, cv=k_fold,n_jobs=-1)
    p1_mlp = sum(prec)/len(prec)
    
    y = attr_label[:,attr_index[2]]
    filter_id = np.where(y==0)
    y = np.delete(y,filter_id,0)
    X_emb_feat2 = np.delete(emb,filter_id,0)
    clf = LogisticRegression()
    prec = cross_val_score(clf, X_emb_feat2, y, cv=k_fold,n_jobs=-1)
    p2_lr = sum(prec)/len(prec) 
    clf = svm.SVC()
    prec = cross_val_score(clf, X_emb_feat2, y, cv=k_fold,n_jobs=-1)
    p2_svm = sum(prec)/len(prec)
    clf = MLPClassifier(max_iter=500)
    prec = cross_val_score(clf, X_emb_feat2, y, cv=k_fold,n_jobs=-1)
    p2_mlp = sum(prec)/len(prec)

    return roc_score,ap_score,p0_mlp,p1_mlp,p2_lr,p2_svm,p2_mlp
