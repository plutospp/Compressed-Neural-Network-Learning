# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:29:23 2017

@author: david
"""
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product
from munkres import Munkres
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import metrics
from tabulate import tabulate
from tensorly.decomposition import tucker
import sys
eps = sys.float_info.epsilon

def cat2num(x, is_one_hot = True):
    distinct = list(set(x))
    idx = \
        np.concatenate(
            list(
                map(
                    lambda d: np.expand_dims(x==d, 1), 
                    distinct
                )
            ), 
            1
        ).astype('float')
    if not is_one_hot:
        idx = toLabels(idx)
    return idx

def dataPreprocess(x, is_categorical):
    if is_categorical:
        y = cat2num(x, False)
    else:
        y = x.astype('float')
        y[np.isnan(y)] = np.nanmean(y)
        y = y/np.sqrt(np.nanmean(y*y))    
    return y

def toLabels(one_hots):
    return np.argmax(one_hots, axis = 1) + 1

def toOneHot(labels, n = 0):
    if n==0:
        n_labels = int(np.max(labels))
    else:
        n_labels = n
    one_hot = np.eye(n_labels)[np.array(labels-1, dtype = 'int'),:]
    return one_hot
    
def clust_accuracy(labels_true, labels_pred):
    onehots_pred = toOneHot(labels_pred)
    onehots_true = toOneHot(labels_true)
    n = onehots_pred.shape[1]
    C = np.zeros([n,n])
    for i, j in product(range(n), range(n)):
        C[i, j] = np.sum(np.abs(onehots_pred[:,i] - onehots_true[:,j]))
    indexes = Munkres().compute(C)
    P = np.zeros([n,n])
    for idx in indexes:
        P[idx] = 1
    return accuracy_score(labels_true, toLabels(onehots_pred@P))   

def randIndex(labels1,labels2):
    tp,tn,fp,fn = 0.0,0.0,0.0,0.0
    for point1 in range(len(labels1)):
        for point2 in range(len(labels1)):
            tp += 1 if labels1[point1] == labels1[point2] and labels2[point1] == labels2[point2] else 0
            tn += 1 if labels1[point1] != labels1[point2] and labels2[point1] != labels2[point2] else 0
            fp += 1 if labels1[point1] != labels1[point2] and labels2[point1] == labels2[point2] else 0
            fn += 1 if labels1[point1] == labels1[point2] and labels2[point1] != labels2[point2] else 0
    return (tp+tn) /(tp+tn+fp+fn)

def kernelize(X, L, knn_ratio, is_categorical):
    if is_categorical:
        K = (X==np.transpose(L)).astype('float')
    else:
        D = cdist(X, L)
        sigsX = np.maximum(np.partition(D, int(knn_ratio*L.shape[0]), 1)[:,-1], 10**-6)
        sigsL = np.maximum(np.partition(D, int(knn_ratio*X.shape[0]), 0)[-1,:], 10**-6)
        K = np.exp(-np.divide(np.power(D,2), np.reshape(sigsX,[-1,1])@np.reshape(sigsL,[1,-1])))
    return K
    
def field_ker(X, L, knn_ratio, distfun):
    D = distfun(X, L)
    sigsX = np.maximum(np.partition(D, int(knn_ratio*L.shape[0]), 1)[:,-1], 10**-6)
    sigsL = np.maximum(np.partition(D, int(knn_ratio*X.shape[0]), 0)[-1,:], 10**-6)
    K = np.exp(-np.divide(np.power(D,2), np.reshape(sigsX,[-1,1])@np.reshape(sigsL,[1,-1])))    
    return K

def MVkernelize(X, L, knn_ratio, is_categorical):
    ker_list = \
        list(
            map(
                lambda x,l,is_cat: np.expand_dims(kernelize(x, l, knn_ratio, is_cat), 2), 
                X, L, is_categorical
            )
        )
    ret = \
        np.concatenate(
            ker_list, 
            2
        )
    if len(ret.shape)==1:
        ret = np.expand_dims(ret, 2)
    return ret

    
def normalize(K):
    Dx = np.diag(1.0/np.sqrt(K.sum(axis= 1)))
    Dy = np.diag(1.0/np.sqrt(K.sum(axis= 0)))
    return Dx.dot(K).dot(Dy)    
    
def speClust(X, Y, k, nnr):
    K = normalize(kernelize(X, Y, nnr))
    u, _, _ = np.linalg.svd(K)
    return KMeans(n_clusters = k).fit(u[:,:k]).labels_

def MVspeClust(X, Y, k, nnr):
#    K = np.reshape(MVkernalize(X, Y, nnr), [X.shape[0],-1])
#    u, _, _ = np.linalg.svd(normalize(K))
#    return KMeans(n_clusters = k).fit(u[:,:k]).labels_    
    K = MVkernelize(X, Y, nnr)
    _, factors = tucker(K, ranks = [k,k,k])
    return KMeans(n_clusters = k).fit(factors[0]).labels_    
    
def main():    
    
    dataset = 'iris'

    file = '/home/david/Dropbox/datasets/' + dataset.upper() + '.csv'
    datmat = np.loadtxt(file, dtype = 'float', delimiter = ',')
    x = dataPreprocess(datmat[:,:-1])        
    labels = datmat[:,-1] - 1
    n_clusters = int(max(labels) + 1)

    labels_pred0 = KMeans(n_clusters = n_clusters).fit(x).labels_
    labels_pred1 = speClust(x, x, n_clusters, 0.1)
    labels_pred2 = MVspeClust(x, x, n_clusters, 0.1) 

    ri0 = randIndex(labels_pred0, labels)
    ri1 = randIndex(labels_pred1, labels)
    ri2 = randIndex(labels_pred2, labels)

    nmi0 = metrics.normalized_mutual_info_score(labels_pred0,labels)
    nmi1 = metrics.normalized_mutual_info_score(labels_pred1,labels)
    nmi2 = metrics.normalized_mutual_info_score(labels_pred2,labels)

    acc0 = clust_accuracy(labels_pred0,labels)
    acc1 = clust_accuracy(labels_pred1,labels)
    acc2 = clust_accuracy(labels_pred2,labels)

    table = \
        tabulate( \
            [ \
                [dataset.upper()], \
                ['index', 'kmeans', 'spectral', 'MVspectral'], \
                ['RI', '{:05.4f}'.format(ri0), '{:05.4f}'.format(ri1), '{:05.4f}'.format(ri2)], \
                ['NMI', '{:05.4f}'.format(nmi0), '{:05.4f}'.format(nmi1), '{:05.4f}'.format(nmi2)], \
                ['ACC', '{:05.4f}'.format(acc0), '{:05.4f}'.format(acc1), '{:05.4f}'.format(acc2)] \
            ] \
        )

    print(table)

if __name__ == "__main__":
    main()