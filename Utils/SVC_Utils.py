import numpy as np 
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import torch
import torchvision 

def load(dataloader):
    for data in dataloader:
        x,y=data
    x=x.view(x.shape[0],-1)
    return x,y

def hp_grid(n_components, C_range, gamma_range):
    clfs=[]
    pca=PCA(n_components=n_components)
    scaling = MinMaxScaler(feature_range=(-1,1))
    for i in C_range:
        for j in gamma_range:
            svc=svm.SVC(C=i, gamma=j)
            clf=make_pipeline(pca, scaling, svc)
            clfs.append(clf)
    return clfs

def train_grid(clfs, inputs, targets):
    fitted_clfs=[]
    for i in range(len(clfs)):
        x=clfs[i].fit(inputs, targets)
        fitted_clfs.append(x)
        print('Fitted: ', i+1, '/', len(clfs))
    return fitted_clfs

def predict_eval(clf, inputs, targets, training=False):
    preds=clf.predict(inputs)
    num_correct=torch.eq(torch.from_numpy(preds), targets).sum().item()
    acc=(num_correct/len(targets))*100
    if training:
        print('C: ', clf.get_params(deep=True)['svc__C'], 'gamma: ', clf.get_params(deep=True)['svc__gamma'])
        print('Training Accuracy: ', acc)
    else:
        print('Testing Accuracy: ', acc)
    return preds, acc

def maxacc_gen(test_accs, train_accs, clfs):
    """Finds and returns model with highest test accuracy and model with train/test accuracy ratio closest to 1."""
    
    test=np.array(test_accs)
    train=np.array(train_accs)
    
    maxacc=clfs[np.argmax(test)]
    gen=clfs[np.argmin(train-test)]
    
    return maxacc, gen

def save_proba(fn, pipe, inputs, targets):
    """Fits svm with probabilities and saves to disk."""

    params=pipe.get_params(deep=True)    
        
    pca=PCA(n_components=180)
    scaling = MinMaxScaler(feature_range=(-1,1))
    pipe_prob=make_pipeline(pca, scaling, svm.SVC(C=params['svc__C'], gamma=params['svc__gamma'], probability=True))
                        
    pipe_prob.fit(inputs, targets)
    joblib.dump(pipe_prob, fn)
