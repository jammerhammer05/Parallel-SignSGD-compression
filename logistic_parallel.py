#!/usr/bin/python

import functools
import numpy as np
import math
import os
import scipy.io as sio
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
import pandas as pd
import math
import cv2
import sys
import decimal
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import train_test_split

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from mpi4py import MPI
import datetime
a=datetime.datetime.now()

Gpu_mode = False
Distributed = False

# Init MPI
comm = MPI.COMM_WORLD

Matrix_dot = np.dot


def convert_memory_ordering_f2c(array):
    if np.isfortran(array) is True:
        return np.ascontiguousarray(array)
    else:
        return array


def load_training_data(training_file='mnistdata.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).
    inputs: numpy array with size (5000, 400).
    labels: numpy array with size (5000, 1).
    The training data is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4data1.mat).
    '''
    training_data = sio.loadmat(training_file)
    inputs = training_data['X'].astype('f8')
    inputs = convert_memory_ordering_f2c(inputs)
    labels = training_data['y'].reshape(training_data['y'].shape[0])
    labels = convert_memory_ordering_f2c(labels)
    return (inputs, labels)

def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def compress(theta_grad):
    
    shape_theta = theta_grad.shape

    print("shape :", shape_theta)
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

def compare(t1, t2):
    shape = t1.shape
    y = shape[0]
    
    while(y > 0):
        x = shape[1]
        
        while(x > 0):
            if(t1[y-1][x-1] != t2[y-1][x-1]):
                return False
            x = x-1
        y = y - 1
    
    return True

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

    J_dash_theta=np.sign(J_dash_theta)
    l1, J_dash_theta_cmp = compress(J_dash_theta)
    return J_dash_theta_cmp
    

def perform_gradient_descent(X, Theta, alpha, Y_actual, itr, threshold):
    
    iterations = itr
    m = X.shape[0]
                    
    for i in range(iterations):
        if Distributed is True:
            sliced_inputs = np.asarray(np.split(X, comm.size))
            sliced_labels = np.asarray(np.split(Y_actual, comm.size))
            inputs_buf = np.zeros((len(inputs)//comm.size), dtype='uint8')
            labels_buf = np.zeros((len(labels)//comm.size), dtype='uint8')

            comm.Barrier()
            if comm.rank == 0:
                time_scatter_start = time.time()
            comm.Scatter(sliced_inputs, inputs_buf)
            if comm.rank == 0:
                time_scatter_end = time.time()
                print('\tScatter inputs uses {} secs.'.format(time_scatter_end - time_scatter_start))

            comm.Barrier()
            if comm.rank == 0:
                time_scatter_start = time.time()
            comm.Scatter(sliced_labels, labels_buf)
            if comm.rank == 0:
                time_scatter_end = time.time()
                print('\tScatter labels uses {} secs.'.format(time_scatter_end - time_scatter_start))

            # Calculate distributed costs and gradients of this iteration
            # by cost function.
            comm.Barrier()
            cost, (J_dash_theta_c) = get_J_dash_theta(inputs_buf,Theta,alpha,labels_buf,threshold)
            J_dash_theta_grad = decompress(theta1_grad_c, l1)
            comm.Barrier()
            J_dash_theta_buf = np.asarray([np.zeros_like(J_dash_theta_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(J_dash_theta_grad, J_dash_theta_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta1 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            J_dash_theta_grad = functools.reduce(np.add, J_dash_theta_buf) / comm.size

        diff = J_dash_theta*(alpha/m)
        Theta = np.add(Theta, diff)

        #J_dash_theta = get_J_dash_theta(X, Theta, alpha, Y_actual, threshold)
        #diff = J_dash_theta*(alpha/m)
        #Theta = np.add(Theta, diff)
        
    return Theta

def fit_logistic_regression_multiclass_one_vs_one(unique_labels, X, Y_label, alpha=0.01, threshold=0.5, itr=1000):
    
    # Fitting the model means calculating Theta values for all the classes vs all the other classes
    # created a map for storing all the respective theta values
    num_of_classes = len(unique_labels)
    Theta = {}
    if Distributed is True:
        for idx in range(0, num_of_classes):
            for jdx in range(idx+1, num_of_classes):
                if comm.rank == 0:
                    i = unique_labels[idx]
                    j = unique_labels[jdx]
                    X_train_ij = []
                    Y_train_actual_ij = []
                    
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
                comm.Barrier()
                if comm.rank == 0:
                    dim = X_train_ij.shape[1]
                    theta_init_ij = np.random.rand(1,dim)
                comm.Bcast([theta_init_ij, MPI.DOUBLE])
                comm.Barrier()
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


def train(inputs, labels, learningrate=0.1, iteration=200)
    ul = get_unique_labels(labels)
    model = fit_logistic_regression_multiclass_one_vs_one(ul, inputs, labels, alpha=learningrate, itr=iteration)
    return model

if __name__ == '__main__':
    
    Matrix_dot = np.dot

    # Note: There are 10 units which present the digits [1-9, 0]
    # (in order) in the output layer.
    inputs, labels = load_data()

    # train_ip, train_lb, test_ip, test_labels = 

    # train the model from scratch and predict based on it
    model = train(inputs, labels, learningrate=0.1, iteration=200)

    outputs = predict(model, inputs)

    acc = accuracy_score(labels, outputs)

    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i]:
            correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    print('accuracy: ',acc)
    print('precision: {}'.format(precision))

time = datetime.datetime.now()-a
print("EXECUTION_TIME:",time)