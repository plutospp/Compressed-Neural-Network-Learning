# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:48:03 2017

@author: david
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import correlation
from nn_tools import toOneHot, dataPreprocess, field_ker
from tensorflow.examples.tutorials.mnist import input_data
from nltk.cluster import KMeansClusterer, euclidean_distance, cosine_distance
from functools import reduce

def mix_dist(X, Y, is_categorical):
    X = np.transpose(np.expand_dims(X,2), [0,2,1])
    Y = np.transpose(np.expand_dims(Y,2), [2,0,1])
    is_cat = is_categorical.astype('float')
    is_cat = np.expand_dims(is_cat, 1)
    is_cat = np.expand_dims(is_cat, 2)
    is_cat = np.transpose(is_cat, [1,2,0])
    ds = np.logical_not(X==Y)*is_cat + np.abs(X-Y)*(1-is_cat)
    return np.sqrt(np.sum(ds*ds, 2))

def encoding(X_list, is_categorical_list, n_encodings):
    features = np.concatenate(X_list, 1)
    is_categorical = np.concatenate(is_categorical_list)
    ret = \
        np.concatenate(
            list(
                map(
                    lambda x, is_cat, n_enc: np.expand_dims(x, 1) if not is_cat else toOneHot(x, n_enc), 
                    np.transpose(features), 
                    is_categorical, 
                    n_encodings
                )            
            ),
            1
        )
    return np.expand_dims(ret, 2) 

def logfunc(x, x2):
    return tf.multiply( x, tf.log(tf.div(x,x2)))

def kl_div(rho, rho_hat):
    invrho = tf.subtract(tf.constant(1.), rho)
    invrhohat = tf.subtract(tf.constant(1.), rho_hat)
    logrho = tf.add(logfunc(rho,rho_hat), logfunc(invrho, invrhohat))
    return logrho

def tprod_n(Wn, X):    
    Y = tf.random_normal([int(X.shape[0]), int(Wn.shape[1]), int(X.shape[2])])
    for d in range(int(X.shape[2])):
        Y[:,:,d] = \
            tf.transpose(
                tf.sparse_tensor_dense_matmul(
                    Wn[:,:,d], tf.transpose(X[:,:,d], [1,0])
                )
            )
    return Y

def conv(X, nbrs, actfn):
    wn = tf.Variable(tf.random_normal([nbrs.size]))
    a = np.expand_dims(nbrs, 2)
    b = \
        np.expand_dims(
            np.tile(
                np.expand_dims(np.arange(nbrs.shape[0]),1),
                [1, nbrs.shape[1]]
            ), 2                    
        )
    idxs = \
        np.reshape(
            np.concatenate(
                [
                    b,
                    a
                ], 
            ), [-1,2]
        )
    dshape = [nbrs.shape[0], int(X.shape[1])]
    Wn = tf.SparseTensor(idxs, wn, dshape)
    l2_loss = tf.reduce_sum(wn*wn)
    output = actfn(tf.sparse_tensor_dense_matmul(Wn,tf.transpose(X)))
#    output = actfn(X@tf.sparse_tensor_to_dense(Wn))
    return tf.transpose(output), l2_loss

def mv_conv(X, nbrs, rate, actfn):
    wn = tf.Variable(tf.random_normal([nbrs.size]))
    a = np.expand_dims(nbrs, 3)
    b = \
        np.expand_dims(
            np.tile(
                np.expand_dims(np.arange(nbrs.shape[0]),1),
                [1, nbrs.shape[1], nbrs.shape[2]]
            ), 3                    
        )
    c = \
        np.expand_dims(
            np.tile(
                np.arange(nbrs.shape[2]),
                [nbrs.shape[0], nbrs.shape[1],1]
            ), 3                    
        )
    idxs = \
        np.reshape(
            np.concatenate(
                [
                    b,
                    a,
                    c
                ], 3
            ), [-1,2]
        )
    dshape = [nbrs.shape[0], int(X.shape[1]), nbrs.shape[2]]
    Wn = tf.SparseTensor(idxs, wn, dshape)
    Wd = tf.Variable(tf.random_normal([nbrs.shape[2], int(nbrs.shape[2]*rate)]))
    l2_loss = tf.reduce_sum(wn*wn)
    output = actfn(tf.tensordot(tprod_n(Wn, X), Wd, [2,0]))
    return l2_loss, output
        
def full(X, rate, actfn, shape_X, sparse):
    shape_Y = (rate*shape_X).astype('int')
    shape_Wn = np.concatenate([shape_X,shape_Y])
    Wn = tf.Variable(tf.random_normal(shape_Wn))
    B  = tf.Variable(tf.random_normal(shape_Y))
    Y = actfn(tf.tensordot(X,Wn,[np.arange(1,len(shape_X)+1),np.arange(len(shape_X))]) + B)
    kl_div_loss = tf.reduce_sum(kl_div(sparse, tf.reduce_mean(Y,0)))
    l1_loss = tf.reduce_sum(tf.abs(Wn))    
    l2_loss = tf.reduce_sum(Wn**2)
    return Y, shape_Y, l1_loss, l2_loss, kl_div_loss   
    
def dim_full(X, rates, actfn, shape_X, sparse):
    shape_Y = (np.array(rates)*shape_X).astype('int')
    shape_Y[shape_Y==0] = 1
    Wns = list(map(lambda dim_x, dim_y: tf.Variable(tf.random_normal([dim_x, dim_y])), shape_X, shape_Y))
    Z = \
        reduce(
            lambda T, W: tf.tensordot(T, W, [[1], [0]]),
            Wns, 
            X
        )
    B  = tf.Variable(tf.random_normal(shape_Y))
    Y = actfn(Z + B)
    kl_div_loss = tf.reduce_sum(kl_div(sparse, tf.reduce_mean(Y,0)))    
    l1_loss = reduce(lambda loss, W: loss + tf.reduce_sum(tf.abs(W)), Wns, 0)   
    l2_loss = reduce(lambda loss, W: loss + tf.reduce_sum(W**2), Wns, 0) 
    return Y, shape_Y, l1_loss, l2_loss, kl_div_loss 

def final(X, k, actfn, shape_X, sparse):
    shape_Wn = np.concatenate([shape_X,[k]],0)
    Wn = tf.Variable(tf.random_normal(shape_Wn))
    B  = tf.Variable(tf.random_normal([k]))
    idx = [np.arange(1,len(shape_X)+1),np.arange(len(shape_X))]
    Y = actfn(tf.tensordot(X,Wn,idx) + B)
    kl_div_loss = tf.reduce_sum(kl_div(sparse, tf.reduce_mean(Y,0)))
    l1_loss = tf.reduce_sum(tf.abs(Wn)) 
    l2_loss = tf.reduce_sum(Wn**2)
    return Y, l1_loss, l2_loss, kl_div_loss   
    
def sample(X, samp, nnr, metric):
    n = X.shape[0]
    samp_idx = np.random.choice(n, int(n*samp), replace = False)
    X_samp = X[samp_idx,:]
    nbrs = \
        NearestNeighbors(
            n_neighbors = int(n*nnr), 
            algorithm = 'brute',
            metric = metric
        ).fit(X).kneighbors(X[samp_idx,:])[1]
    return X_samp, nbrs
    
def train(sess, inputs, labels, training_epochs, batch_size, optimizer, cost, X, L, nnr, kerfn):
    n_points = inputs[0].shape[0]
    display_step = 1
    total_batch = int(n_points/batch_size) + 1
    init = tf.global_variables_initializer() 
    sess.run(init)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            x = list(map(lambda inp: inp[np.arange(i*batch_size, np.minimum((i+1)*batch_size, n_points)),:], inputs))
            l = labels[np.arange(i*batch_size, np.minimum((i+1)*batch_size, n_points)),:]
            _, c = \
                sess.run( 
                    [optimizer, cost], 
                    feed_dict = { X: kerfn(x), L: l } 
                )
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))  
                      
def prediction(sess, model, X_test, batch_size, X, kerfn, shape_Y):
    n = X_test[0].shape[0]
    total_batch = int(n/batch_size) + 1
    partitions = np.empty(np.concatenate([[0], shape_Y]), 'float')
    for i in range(total_batch):
        x = list(map(lambda xt: xt[np.arange(i*batch_size, np.minimum((i+1)*batch_size, n)),:], X_test))       
        partitions = \
            np.concatenate(
                [
                    partitions, 
                    sess.run(model, feed_dict = { X: kerfn(x)})
                ], 0
            )  
    return partitions
    
def euclidean(P1, P2):
    return tf.reduce_sum(tf.pow(P1 - P2,2))
    
def kl_divergence(P1, P2):
    eps = np.finfo(float).eps
    return tf.reduce_sum(P1*tf.log((P1+eps)/(P2+eps)))
    
def group_by_columns(mat, groups):
    def group_fn(alp):
        idx = groups==alp
        ret = mat[:,idx]
        if len(idx)==1:
            ret = np.expand_dims(ret,1)
        return ret
    distinct = list(set(groups))
    return list(map(group_fn, distinct))

def group_by(arr, groups):
    distinct = list(set(groups))
    return list(map(lambda alp: arr[groups==alp], distinct))
    
def getData(datadir, test_rate, lm_rate, is_preprocess, is_categorical_default, is_grouped_default, max_cat_num):
    data_mat = np.loadtxt(datadir, dtype='str', delimiter=',')
    dim = data_mat.shape[1]-1
    true_arr = data_mat[0,:] == 'T'
    false_arr = data_mat[0,:] == 'F'
    if any(true_arr) or any(false_arr):
        is_categorical = true_arr
        data_mat = data_mat[1:,:]
    else:
        is_categorical = np.concatenate([np.repeat(is_categorical_default, dim), [True]])    
    groups = data_mat[0,:]
    if all(np.in1d(groups, np.vectorize(chr)(np.arange(65, 65 + max_cat_num)))): 
        data_mat = data_mat[1:,:]
    elif is_grouped_default:
        groups = np.repeat('a', dim)
    else:
        groups = np.vectorize(chr)(np.arange(65, 65+dim))
    if is_preprocess:
        data_mat = \
            np.concatenate(
                list(
                    map(
                        lambda x, is_cat: np.expand_dims(dataPreprocess(x, is_cat), 1),
                        np.transpose(data_mat),
                        is_categorical
                    )                
                ),
                1
            )
    features = data_mat[:,:-1]
    labels = data_mat[:,-1].astype('int')
    n = features.shape[0]
    n_test = int(n*test_rate)
    n_train = n - n_test
    lm_idx = np.random.choice(n_train, int(n_train*lm_rate))
    test_idx = np.zeros(dtype = 'bool', shape = [n])
    test_idx[np.random.choice(n, n_test)] = True
    train_idx = ~ test_idx
    X_all = group_by_columns(features, groups)
    X_train = list(map(lambda x: x[train_idx], X_all))
    X_test  = list(map(lambda x: x[test_idx], X_all))
    X_lm = list(map(lambda x: x[train_idx][lm_idx], X_all))                    
    L_train = toOneHot(labels[train_idx])
    L_test  = toOneHot(labels[test_idx])
    is_categorical_list = group_by(is_categorical[:-1], groups)
    n_encodings = \
        list(
            map(
                lambda x, is_cat: x.shape[1] if not is_cat else np.max(x),
                np.transpose(X_all),
                is_categorical[:-1]                
            )            
        )  
    return X_train, L_train, X_test, L_test, X_lm, is_categorical_list, n_encodings
    
def main():
    test_rate = 0.1
    landmark_rate = 0.0005
    is_mnist = False
    is_categorical_default = True
    is_grouped_default = False
    max_cat_num = 100
    if is_mnist:
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        X_train = [mnist.train.images]
        X_test = [mnist.test.images]
        L_train = mnist.train.labels
        L_test = mnist.test.labels
        is_categorical = [False]        
    else:            
        datadir = '/home/david/Dropbox/datasets/CON.csv'
        is_preprocess = True
        X_train, L_train, X_test, L_test, X_lm, is_categorical, n_encodings = \
            getData(
                datadir, 
                test_rate, 
                landmark_rate, 
                is_preprocess, 
                is_categorical_default, 
                is_grouped_default, 
                max_cat_num
            )
    initial_learning_rate = 0.01
    n_clusters = L_train.shape[1]
    samp_rate = 1.5
    nnr = 0.1
    alpha = 10**2
    beta = 10**-9
    gamma = 10**-6
    sparse = 0.05
    n_landmarks = X_lm[0].shape[0]
    training_epochs = 50
    batch_size = 200
    actfn = tf.nn.sigmoid
    hidden_node_rates = [[0.25, 0.5], [1.0, 1.0]] 
    kernalization = 'ker'#{'raw', 'ker'}     
    
    network = dim_full  
    if kernalization=='raw':
        kerfn = lambda x: encoding(x, is_categorical, n_encodings) 
        input_shape = [np.sum(n_encodings), 1]
    elif kernalization=='ker':
        input_shape = [n_landmarks, len(is_categorical)]
        kerfn = \
            lambda X: \
                np.concatenate(
                    list(
                        map(
                            lambda x, x_lm, is_cats: 
                                np.expand_dims(
                                    field_ker(
                                        x, 
                                        x_lm, 
                                        nnr, 
                                        lambda x1, x2: mix_dist(x1, x2, is_cats)
                                    ),
                                    2
                                ),
                            X, X_lm, is_categorical
                        )                    
                    ),
                    2
                )
    
    sess = tf.Session()
    X = tf.placeholder("float", np.concatenate([[None], input_shape]))
    L = tf.placeholder("float", [None,n_clusters])
    
    model = X    
    shape_X = np.array(X.shape[1:].as_list(), 'int')
    L1_loss = 0
    L2_loss = 0
    KL_loss = 0
    for rates in hidden_node_rates:
        model, shape_X, l1_loss, l2_loss, kl_loss = network(model, rates, actfn, shape_X, sparse)
        L2_loss += l2_loss 
        KL_loss += kl_loss
    model, l1_loss, l2_loss, kl_loss = final(model, n_clusters, actfn, shape_X, sparse)
    L1_loss += l1_loss    
    L2_loss += l2_loss
    KL_loss += kl_loss
    
    cost = tf.norm(model - L)**2 + alpha*L1_loss #+ beta*L2_loss + gamma*KL_loss
    optimizer = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer() 
    sess.run(init)    

    train(sess,X_train,L_train,training_epochs,batch_size,optimizer,cost,X,L,nnr,kerfn)
    partitions = prediction(sess, model, X_test, batch_size, X, kerfn, [n_clusters])
    print(accuracy_score(np.argmax(L_test,1), np.argmax(partitions,1))) 
    

if __name__ == "__main__":
    main()