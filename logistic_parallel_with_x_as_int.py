#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import math
import os
import sys
import decimal
from collections import defaultdict
from collections import Counter
from sklearn.metrics import accuracy_score
import functools
import numpy as np
import math
import os
import scipy.io as sio
import time
from sklearn.metrics import accuracy_score

from mpi4py import MPI
import datetime

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
# Init MPI
comm = MPI.COMM_WORLD

Matrix_dot = np.dot

Input_layer_size=400
def compress(theta_grad):
    
    shape_theta = theta_grad.shape

    #print("shape :", shape_theta)
    loops = shape_theta[0]
    itr = shape_theta[1]
    reslist = []
    
    while(loops > 0):

        res = 0
        itr = shape_theta[1]
        i = 0
        while(itr > 0):
            
            x = int(theta_grad[loops-1][itr-1])
            if(x == -1):
                res =  res + 2 * pow(3,i)
            elif(x == 0):
                res =  res 
            elif(x == 1):
                res =  res + 1 * pow(3, i)
            else :
                print("Error in signSGD compression")
                exit(0)

            itr = itr - 1
            i = i + 1

        reslist.append(res)
        loops = loops - 1

    return shape_theta[1], reslist

def decompress(t_c, l):
    
    t_len = len(t_c)
    reslist = []
    i = 0

    while (i < t_len):

        res = []
        t = t_c[i]

        while t:
            t, r = divmod(t, 3)
            if(r == 2):
                res.append(-1)
            else:
                res.append(r)

        lenlist = len(res)

        if(lenlist != l):
            dif = l - lenlist
            while dif:
                res.append(0)
                dif = dif-1

        res.reverse()
        i = i+1
        reslist.append(res)

    reslist.reverse()
    resnp = np.array(reslist)
    return resnp

def load_test_train_data(training_file='mnistdata.mat'):
    training_data = sio.loadmat(training_file)


    inputs = training_data['X'].astype('f8')   
    labels = training_data['y'].reshape(training_data['y'].shape[0])

    xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.1)

    # xtrain = convert_memory_ordering_f2c(xtrain)
    # xtest = convert_memory_ordering_f2c(xtest)
    # ytrain = convert_memory_ordering_f2c(ytrain)
    # ytest = convert_memory_ordering_f2c(ytest)
    return (xtrain, xtest, ytrain, ytest)

def get_unique_labels(list1):
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set))
    return unique_list

 
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def h_theta(X, Theta): 
    Z = np.dot(Theta, X.T)        # as we take X as a horizontal vector here and not a vertical vector, hence we dont do 

    res = sigmoid(Z)              # np.dot(Theta.T, X) as written in general but np.dot(Theta, X.T)
    
    return res                    # Theta : 1 X d  X.T : d X n  res = 1 X n  

def get_J_dash_theta(X, Theta, alpha, Y_actual, threshold):
    
    Y_pred = h_theta(X, Theta)  # must have same dimensions as Y_actual : 1 X n
    Y_pred[Y_pred >= threshold] = 1 
    Y_pred[Y_pred < threshold] = 0
    Y_actual_minus_pred = np.subtract(Y_actual, Y_pred)  
    J_dash_theta = np.dot(Y_actual_minus_pred, X)  # or the direction that is the derivitive : 1 X d
    J_dash_theta_sgn=np.sign(J_dash_theta)
    l1, J_dash_theta_cmp = compress(J_dash_theta_sgn)
    return l1,J_dash_theta_cmp
    

def perform_gradient_descent(X, Theta, alpha, Y_actual, itr, threshold):
    
    iterations = itr
    m = X.shape[0]


    if comm.rank == 0:
        theta = Theta
    else:
        theta = np.zeros_like(Theta)

    comm.Barrier()
    
    if comm.rank == 0:
        time_bcast_start = time.time()
    comm.Bcast([theta, MPI.DOUBLE])
    comm.Barrier()
    
    if comm.rank == 0:
        time_bcast_end = time.time()
        #print('\tBcast theta uses {} secs.'.format(time_bcast_end - time_bcast_start))


    
    q = comm.size
    n, r = divmod(X.shape[0], q)
    if(r != 0):
        d = n*q
        X = X[0:d]
        Y_actual=Y_actual[0:d]
    X = X.astype('uint8')
    Y_actual = Y_actual.astype('uint8')
    #print(Y_actual.shape)

    for i in range(iterations):

        time_iter_start = time.time()
        #print(len(X),comm.size)
        sliced_inputs = np.asarray(np.split(X, comm.size))
        sliced_labels = np.asarray(np.split(Y_actual, comm.size))
        inputs_buf = np.zeros((len(X)//comm.size, Input_layer_size))
        labels_buf = np.zeros((len(Y_actual)//comm.size),)
        # inputs_buf = bytearray(1<<24)
        # labels_buf = bytearray(1<<24)
        #print(sliced_inputs.shape,inputs_buf.shape)
        #print(sliced_labels.shape,labels_buf.shape)
        comm.Barrier()
        if comm.rank == 0:
            time_scatter_start = time.time()
        comm.Scatter(sliced_inputs, inputs_buf)
        if comm.rank == 0:
            time_scatter_end = time.time()
            #print('\tScatter inputs uses {} secs.'.format(time_scatter_end - time_scatter_start))

        comm.Barrier()
        if comm.rank == 0:
            time_scatter_start = time.time()
        comm.Scatter(sliced_labels, labels_buf)
        if comm.rank == 0:
            time_scatter_end = time.time()
            #print('\tScatter labels uses {} secs.'.format(time_scatter_end - time_scatter_start))

        # Calculate distributed costs and gradients of this iteration
        # by cost function.
        comm.Barrier()

        
        #J_dash_theta = get_J_dash_theta(X, Theta, alpha, Y_actual, threshold)
        l1,J_dash_theta_c=get_J_dash_theta(inputs_buf,Theta,alpha,labels_buf,threshold)
        J_dash_theta_grad = decompress(J_dash_theta_c, l1)


        comm.Barrier()
        J_dash_theta_buf = np.asarray([np.zeros_like(J_dash_theta_grad)] * comm.size)
        comm.Barrier()
        if comm.rank == 0:
            time_gather_start = time.time()
        comm.Gather(J_dash_theta_grad, J_dash_theta_buf)
        if comm.rank == 0:
            time_gather_end = time.time()
            #print('\tGather theta uses {} secs.'.format(time_gather_end - time_gather_start))
        comm.Barrier()
        J_dash_theta_grad = functools.reduce(np.add, J_dash_theta_buf) / comm.size


        #diff = J_dash_theta*(alpha/m)
        #Theta = np.add(Theta, diff)

        diff = J_dash_theta_grad*(alpha/m)
        Theta = np.add(Theta, diff)

        comm.Bcast([Theta, MPI.DOUBLE])
        comm.Barrier()
        time_iter_end = time.time()
        '''
        if comm.rank == 0:
            print('Iteration {0} (learning rate {1}, iteration {2}), time: {3}'.format(
                i+1, alpha, iterations, time_iter_end - time_iter_start)
            )
        '''
    return Theta

def fit_logistic_regression_multiclass_one_vs_one(unique_labels, X, Y_label, alpha=0.01, threshold=0.5, itr=1000):
    
    # Fitting the model means calculating Theta values for all the classes vs all the other classes
    # created a map for storing all the respective theta values
    num_of_classes = len(unique_labels)
    Theta = {}
    
    for idx in range(0, num_of_classes):
        for jdx in range(idx+1, num_of_classes):
            
            i = unique_labels[idx]
            j = unique_labels[jdx]
            
            X_train_ij = []
            Y_train_actual_ij = []
            
            # seperate out the data according to the label
            
            for k in range(len(Y_label)):
                if(Y_label[k] == i or Y_label[k] == j):
                
                    X_train_ij.append(X[k]) 
                    Y_train_actual_ij.append(Y_label[k])
                    
            X_train_ij = np.array(X_train_ij)
            Y_train_actual_ij = np.array(Y_train_actual_ij)

            # done picking out data for the required labels i and j

            #  Now take i as the positive label and put it as "1" and j becomes "0"       

            Y_train_actual_ij = (Y_train_actual_ij == i).astype(int)

            # pick random values initially for Theta

            dim = X_train_ij.shape[1]
            theta_init_ij = np.random.rand(1,dim)
            theta_ij = perform_gradient_descent(X_train_ij, theta_init_ij, alpha, Y_train_actual_ij, itr, threshold)

            Theta[(i,j)] = theta_ij

    return Theta
    
def predict_logistic_regression_multiclass_one_vs_one(X_test, Theta, unique_labels, threshold=0.5):
    
    test_sample_size = X_test.shape[0]
    num_of_classes = len(unique_labels)
    pred_vec = []
    
    for k in range(test_sample_size):
        
        majority_class_list = []
        
        for idx in range(0, num_of_classes):
            for jdx in range(idx+1, num_of_classes):
            
                i = unique_labels[idx]
                j = unique_labels[jdx]
                prob = h_theta( X_test[k], Theta[(i,j)])
                
                if prob >= threshold:
                    prob = i
                else:
                    prob = j
                    
                majority_class_list.append(prob)
    
        pred_vec.append(Counter(majority_class_list).most_common(1)[0][0])
        
    return pred_vec

def run(X_train, X_test, actuallabels_num_train):

    unique_labels = get_unique_labels(actuallabels_num_train)
    print("Training..... : ")

    Theta = fit_logistic_regression_multiclass_one_vs_one(unique_labels = unique_labels, X = X_train, Y_label = actuallabels_num_train, alpha = 0.8, threshold = 0.5, itr=1000)
    print("Predicting..... : ")

    pred_labels_num = predict_logistic_regression_multiclass_one_vs_one(X_test, Theta, unique_labels)

    return pred_labels_num

xtrain, xtest, ytrain, ytest = load_test_train_data()
op = run(xtrain, xtest, ytrain)

print("Accuracy is : ",accuracy_score(ytest, op))