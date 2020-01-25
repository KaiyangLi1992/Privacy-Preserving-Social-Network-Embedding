import pickle as pkl
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
import numpy as np


def get_attr_list(dataset,labels,features_mat):
## Remove privacy attributes from feature matrix 
# Bulid attibute labels
    if dataset == 'yale':
        #On Yale, elements in columns 0 - 4 correspond to student/faculty status, 
        #elements in columns 5,6 correspond to gender,
        #and elements in  the bottom 6 columns correspond to class year,which is privacy here.

        y=labels[:,0]
        attr_labels = np.eye(len(np.unique(y)))[y.astype(int)-1]
        attr_labels = attr_labels [:,1:]


        y=labels[:,-1]
        privacy_labels = np.eye(len(np.unique(y)))[y.astype(int)-1]
        privacy_labels = privacy_labels [:,1:]

        attr_labels_list = [attr_labels,privacy_labels]
        dim_attr = [attr_labels.shape[1], privacy_labels.shape[1]]
        features_rm_privacy =features_mat[:,:-6]

    elif dataset == 'rochester':
        #On Rochester, elements in columns 0 - 5 correspond to student/faculty status, 
        #elements in the bottom 19 columns correspond to class year,
        #and elements in  the bottom 6,7 columns correspond to gender,which is privacy here.


        y=labels[:,-1]
        attr_labels = np.eye(len(np.unique(y)))[y.astype(int)-1]
        attr_labels = attr_labels [:,1:]

        y=labels[:,1]
        privacy_labels = np.eye(len(np.unique(y)))[y.astype(int)-1]
        privacy_labels = privacy_labels [:,1:]

        attr_labels_list = [attr_labels,privacy_labels]
        dim_attr = [attr_labels.shape[1], privacy_labels.shape[1]]
        features_rm_privacy =np.hstack((features_mat[:,:6],features_mat[:,8:]))
        
        
    return attr_labels_list,dim_attr,features_rm_privacy
