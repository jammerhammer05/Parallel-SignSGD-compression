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
from numpy.random import RandomState
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
    random1=RandomState()

    inputs = training_data['X'].astype('f8')   
    labels = training_data['y'].reshape(training_data['y'].shape[0])

    xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.1,random_state=random1)

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

def get_J_dash_theta(X_train, theta, alpha, Y_train, threshold):
    Y_pred = h_theta(X_train, theta)
    #cost = 1 / X_Train.shape[0] * numpy.sum(-binary_y_train * numpy.log(h) - (1 - binary_y_train) * numpy.log(1 - h))
    Y_pred[Y_pred >= threshold] = 1 
    Y_pred[Y_pred < threshold] = 0
    Y_actual_minus_pred = np.subtract(Y_train, Y_pred)  
    J_dash_theta = np.dot(Y_actual_minus_pred, X_train)  # or the direction that is the derivitive : 1 X d
    #J_dash_theta_sgn=np.sign(J_dash_theta)
    #l1, J_dash_theta_cmp = compress(J_dash_theta_sgn)
    return J_dash_theta
def perform_gradient_descent(X_train, theta1, alpha, Y_train, itr, threshold):
    
    iterations = itr
    m = X_train.shape[0]


    if comm.rank == 0:
        theta = theta1
    else:
        theta = np.zeros_like(theta1)

    comm.Barrier()
    
    if comm.rank == 0:
        time_bcast_start = time.time()
    comm.Bcast([theta, MPI.DOUBLE])
    comm.Barrier()
    
    if comm.rank == 0:
        time_bcast_end = time.time()
        #print('\tBcast theta uses {} secs.'.format(time_bcast_end - time_bcast_start))
    Y_train = Y_train.astype('uint8')
    for i in range(iterations):

        time_iter_start = time.time()
        #print(len(X),comm.size)
        sliced_inputs = np.asarray(np.split(X_train, comm.size))
        sliced_labels = np.asarray(np.split(Y_train, comm.size))
        inputs_buf = np.zeros((len(X_train)//comm.size, Input_layer_size+1))
        labels_buf = np.zeros((len(Y_train)//comm.size),)
        # inputs_buf = bytearray(1<<24)
        # labels_buf = bytearray(1<<24)
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
        #l1,J_dash_theta_c=get_J_dash_theta(inputs_buf,Theta,alpha,labels_buf,threshold)
        #J_dash_theta_grad = decompress(J_dash_theta_c, l1)


        J_dash_theta_grad=get_J_dash_theta(inputs_buf,theta,alpha,labels_buf,threshold)

        #l1,J_dash_theta_c=get_J_dash_theta(inputs_buf,theta,alpha,labels_buf,threshold)
        #J_dash_theta_grad = decompress(J_dash_theta_c, l1)


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


        #diff = (alpha * J_dash_theta_grad)/m
        #theta -= diff

        diff = J_dash_theta_grad*(alpha/m)
        theta = np.add(theta, diff)

        comm.Bcast([theta, MPI.DOUBLE])
        comm.Barrier()
        time_iter_end = time.time()
    return theta

def fit_logistic_regression_multiclass_one_vs_all(classes, X_train, Y_train, alpha=0.01, threshold=0.5, itr=1000):
    
    # Fitting the model means calculating Theta values for all the classes vs all the other classes
    # created a map for storing all the respective theta values
    temp = np.ones(X_train.shape[0])
    X_train=np.insert(X_train,0,temp,axis=1)
    thetas=[]
    for c in classes:
        binary_y_train = np.where(Y_train == c, 1, 0)
        #theta = np.random.rand(1,X_train.shape[1])
        theta=np.random.rand(X_train.shape[1])
        theta = perform_gradient_descent(X_train, theta, alpha,binary_y_train, itr, threshold)
        thetas.append(theta)
    return thetas
    
def predict_logistic_regression_multiclass_one_vs_all(X_test, Theta, classes, threshold=0.5):
    print(type(X_test))
    temp = np.ones(X_test.shape[0])
    X_test=np.insert(X_test,0,temp,axis=1)
    h=[sigmoid(theta.dot(X_test.T)) for theta in Theta]
    h=np.array(h)
    preds = np.argmax(h, axis=0)
    return [classes[p] for p in preds]

def run(X_train, X_test, Y_train):

    classes = get_unique_labels(Y_train)
    #print(classes)
    print("Training..... : ")

    Theta = fit_logistic_regression_multiclass_one_vs_all(classes = classes, X_train = X_train, Y_train = Y_train, alpha = 0.001, threshold = 0.5, itr=1000)
    print("Predicting..... : ")

    pred_labels_num = predict_logistic_regression_multiclass_one_vs_all(X_test, Theta, classes)

    return pred_labels_num

xtrain, xtest, ytrain, ytest = load_test_train_data()
op = run(xtrain, xtest, ytrain)

print("Accuracy is : ",accuracy_score(ytest, op))