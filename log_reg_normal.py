
#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import math
import cv2
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
    return J_dash_theta
    

def perform_gradient_descent(X, Theta, alpha, Y_actual, itr, threshold):
    
    iterations = itr
    m = X.shape[0]
                    
    for i in range(iterations):
        
        J_dash_theta = get_J_dash_theta(X, Theta, alpha, Y_actual, threshold)
        diff = J_dash_theta*(alpha/m)
        Theta = np.add(Theta, diff)
        
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
    print("unique_labels : ",unique_labels)

    Theta = fit_logistic_regression_multiclass_one_vs_one(unique_labels = unique_labels, X = X_train, Y_label = actuallabels_num_train, alpha = 0.01, threshold = 0.5, itr=2000)
    pred_labels_num = predict_logistic_regression_multiclass_one_vs_one(X_test, Theta, unique_labels)

    return pred_labels_num

xtrain, xtest, ytrain, ytest = load_test_train_data()
op = run(xtrain, xtest, ytrain)

print(accuracy_score(ytest, op))

