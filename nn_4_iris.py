import functools
import numpy as np
import math
import os
import scipy.io as sio
import time
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from mpi4py import MPI
import datetime


Gpu_mode = True
Distributed = True

# Init MPI
comm = MPI.COMM_WORLD

# Structure of the 4-layer neural network.
Input_layer_size = 4
hidden_layer_1_size = 50
hidden_layer_2_size = 25
Output_layer_size = 3


Matrix_dot = np.dot

def convert_memory_ordering_f2c(array):
    if np.isfortran(array) is True:
        return np.ascontiguousarray(array)
    else:
        return array


def load_training_data(training_file='mnistdata.mat'):
    training_data = sio.loadmat(training_file)
    inputs = training_data['X'].astype('f8')
    inputs = convert_memory_ordering_f2c(inputs)
    labels = training_data['y'].reshape(training_data['y'].shape[0])
    labels = convert_memory_ordering_f2c(labels)
    return (inputs, labels)

def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def compress(theta_grad):
    
    shape_theta = theta_grad.shape

#     print("shape :", shape_theta)
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

def cost_function(theta1, theta2, theta3, input_layer_size, hidden_layer_1_size, output_layer_size, inputs, labels, regular=0):
  
    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401

    hidden_layer_1 = Matrix_dot(input_layer, theta1.T)
    hidden_layer_1 = sigmoid(hidden_layer_1)
    hidden_layer_1 = np.insert(hidden_layer_1, 0, 1, axis=1)  # add bias, 5000x101

    hidden_layer_2 = Matrix_dot(hidden_layer_1, theta2.T)
    hidden_layer_2 = sigmoid(hidden_layer_2)
    hidden_layer_2 = np.insert(hidden_layer_2, 0, 1, axis=1)  # add bias, 5000x26
    
    output_layer = Matrix_dot(hidden_layer_2, theta3.T)  # 5000x10
    output_layer = sigmoid(output_layer)

    # forward propagation: calculate cost
 
    cost = 0.0
    for training_index in range(len(inputs)):
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1

        for k in range(output_layer_size):
            error = -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
            cost += error
    cost /= len(inputs)

    
    # back propagation: calculate gradiants

    
    # 400 -> 10
    theta1_grad = np.zeros_like(theta1)  # 100x401
    theta2_grad = np.zeros_like(theta2)  # 25x101 
    theta3_grad = np.zeros_like(theta3)  # 10x26
    
    for index in range(len(inputs)):
        
        # transform label y[i] from a number to a vector : one hot
        outputs = np.zeros((1, output_layer_size))  # (1,10)
        outputs[0][labels[index]-1] = 1

        # calculate delta4
        delta4 = (output_layer[index] - outputs).T  # (10,1)

        # calculate delta3
        z3 = Matrix_dot(theta2, hidden_layer_1[index:index+1].T)  # (25,101) x (101,1)
        z3 = np.insert(z3, 0, 1, axis=0)  # add bias, (26,1)
        delta3 = np.multiply(
            Matrix_dot(theta3.T, delta4),  # (26,10) x (10,1)
            sigmoid_gradient(z3)  # (26,1)
        )
        delta3 = delta3[1:] 
        # delta3 =  delta3[1:] # (25,1)

        # calculate delta2
        z2 = Matrix_dot(theta1, input_layer[index:index+1].T)  # (100,401) x (401,1)
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (101,1)
        delta2 = np.multiply(
            Matrix_dot(theta2.T, delta3),  # (101,25) x (25,1)
            sigmoid_gradient(z2)  # (101,1)
        )
        delta2 = delta2[1:]  # (100,1) removing the bias part

        
        # (100,401) = (100,1) x (1,401)
        theta1_grad += Matrix_dot(delta2, input_layer[index:index+1])
        # (25,101) = (25,1) x (1,101)
        theta2_grad += Matrix_dot(delta3, hidden_layer_1[index:index+1])
        # (10,26) = (10,1) x (1,26)
        theta3_grad += Matrix_dot(delta4, hidden_layer_2[index:index+1])

        
    
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)
    theta3_grad /= len(inputs)
    
    theta1_grad=np.sign(theta1_grad)
    theta2_grad=np.sign(theta2_grad)
    theta3_grad=np.sign(theta3_grad)

    l1, theta1_grad_cmp = compress(theta1_grad)
    l2, theta2_grad_cmp = compress(theta2_grad)
    l3, theta3_grad_cmp = compress(theta3_grad)

    # return cost, (theta1_grad, theta2_grad)
    return cost, (theta1_grad_cmp, l1, theta2_grad_cmp, l2, theta3_grad_cmp, l3)


def gradient_descent(inputs, labels, learningrate=0.8, iteration=50):

    if Distributed is True:
        if comm.rank == 0:
            theta1 = rand_init_weights(Input_layer_size, hidden_layer_1_size)
            theta2 = rand_init_weights(hidden_layer_1_size, hidden_layer_2_size)
            theta3 = rand_init_weights(hidden_layer_2_size, Output_layer_size)
        else:
            theta1 = np.zeros((hidden_layer_1_size, Input_layer_size + 1))
            theta2 = np.zeros((hidden_layer_2_size, hidden_layer_1_size + 1))
            theta3 = np.zeros((Output_layer_size, hidden_layer_2_size + 1))
        comm.Barrier()
        
        comm.Bcast([theta1, MPI.DOUBLE])
        comm.Barrier()
        comm.Bcast([theta2, MPI.DOUBLE])
        comm.Barrier()
        comm.Bcast([theta3, MPI.DOUBLE])
        
    else:
        
        theta1 = rand_init_weights(Input_layer_size, hidden_layer_1_size)
        theta2 = rand_init_weights(hidden_layer_1_size, hidden_layer_2_size)
        theta3 = rand_init_weights(hidden_layer_2_size, Output_layer_size)

    labels= labels.astype('uint8')
    cost = 0.0
    for i in range(iteration):

        if Distributed is True:
            # Scatter training data and labels.
            sliced_inputs = np.asarray(np.split(inputs, comm.size))
            sliced_labels = np.asarray(np.split(labels, comm.size))
            inputs_buf = np.zeros((len(inputs)//comm.size, Input_layer_size))
            labels_buf = np.zeros((len(labels)//comm.size), dtype='uint8')

            comm.Barrier()
            comm.Scatter(sliced_inputs, inputs_buf)

            comm.Barrier()
            
            comm.Scatter(sliced_labels, labels_buf)

            # Calculate distributed costs and gradients of this iteration
            # by cost function.
            comm.Barrier()
            # cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            cost, (theta1_grad_c, l1, theta2_grad_c, l2, theta3_grad_c, l3) = cost_function(theta1, theta2, theta3,
                Input_layer_size, hidden_layer_1_size, Output_layer_size,
                inputs_buf, labels_buf, regular=0)

            theta1_grad = decompress(theta1_grad_c, l1)
            theta2_grad = decompress(theta2_grad_c, l2)
            theta3_grad = decompress(theta3_grad_c, l3)

            # Gather distributed costs and gradients.
            comm.Barrier()
            cost_buf = [0] * comm.size
            try:
                cost_buf = comm.gather(cost)
                cost = sum(cost_buf) / len(cost_buf)
            except TypeError as e:
                print('[{0}] {1}'.format(comm.rank, e))

            theta1_grad_buf = np.asarray([np.zeros_like(theta1_grad)] * comm.size)
            comm.Barrier()
            
            comm.Gather(theta1_grad, theta1_grad_buf)
            comm.Barrier()
            theta1_grad = functools.reduce(np.add, theta1_grad_buf) / comm.size

            theta2_grad_buf = np.asarray([np.zeros_like(theta2_grad)] * comm.size)
            comm.Barrier()
            
            comm.Gather(theta2_grad, theta2_grad_buf)
            comm.Barrier()
            theta2_grad = functools.reduce(np.add, theta2_grad_buf) / comm.size

            theta3_grad_buf = np.asarray([np.zeros_like(theta3_grad)] * comm.size)
            comm.Barrier()
            comm.Gather(theta3_grad, theta3_grad_buf)
            comm.Barrier()
            theta3_grad = functools.reduce(np.add, theta3_grad_buf) / comm.size

        else:

            cost, (theta1_grad_c, l1, theta2_grad_c, l2, theta3_grad_c, l3) = cost_function(theta1, theta2, theta3,
                Input_layer_size, hidden_layer_1_size, Output_layer_size,
                inputs_buf, labels_buf, regular=0)

            theta1_grad = decompress(theta1_grad_c, l1)
            theta2_grad = decompress(theta2_grad_c, l2)
            theta3_grad = decompress(theta3_grad_c, l3)

        theta1 -= learningrate * theta1_grad
        theta2 -= learningrate * theta2_grad
        theta3 -= learningrate * theta3_grad

        if Distributed is True:
           # Sync-up weights for distributed worknodes.
            comm.Bcast([theta1, MPI.DOUBLE])
            comm.Bcast([theta2, MPI.DOUBLE])
            comm.Bcast([theta3, MPI.DOUBLE])
            comm.Barrier()

    return cost, (theta1, theta2, theta3)


def train(inputs, labels, learningrate=0.8, iteration=50):
    cost, model = gradient_descent(inputs, labels, learningrate, iteration)
    return model

# 400 X 100 X 25 -> 10
def predict(model, inputs):
    theta1, theta2, theta3 = model
    a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
    a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,100)
    a2 = sigmoid(a2)    # (5000,100)
    a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,101)
    a3 = np.dot(a2, theta2.T)  # (5000,101) x (101,25)
    a3 = sigmoid(a3)  # (5000,25)
    a3 = np.insert(a3, 0, 1, axis=1)  # add bias, (5000,26)
    a4 = np.dot(a3, theta3.T) # (5000,26) x (26,10)
    a4 = sigmoid(a4) 
    return [i.argmax()+1 for i in a4]

# main

X, Y = load_iris(return_X_y=True)
Y = np.add(Y, 1)
xtrain, xtest, ytrain, ytest  = train_test_split(X, Y, test_size=0.2)
xtrain = convert_memory_ordering_f2c(xtrain)
xtest = convert_memory_ordering_f2c(xtest)
ytrain = convert_memory_ordering_f2c(ytrain)
ytest = convert_memory_ordering_f2c(ytest)


Matrix_dot = np.dot

model = train(xtrain, ytrain, learningrate=0.01, iteration=100)

outputs = predict(model, xtest)
acc = accuracy_score(ytest, outputs)

print('accuracy: ',acc)


