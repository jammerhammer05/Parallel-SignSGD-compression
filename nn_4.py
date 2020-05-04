#!/usr/bin/python

import functools
import numpy as np
import math
import os
import scipy.io as sio
import time
from sklearn.metrics import accuracy_score

from mpi4py import MPI
import datetime
a=datetime.datetime.now()

Gpu_mode = False
Distributed = False

# Init MPI
comm = MPI.COMM_WORLD

# Structure of the 4-layer neural network.
Input_layer_size = 400
hidden_layer_1_size = 100
hidden_layer_2_size = 25
Output_layer_size = 10


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


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

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

def cost_function(theta1, theta2, theta3, input_layer_size, hidden_layer_1_size, output_layer_size, inputs, labels, regular=0):
# def cost_function(theta1, theta2, input_layer_size, hidden_layer_1_size, output_layer_size, inputs, labels, regular=0):
    
    '''
    Note: theta1, theta2, inputs, labels are numpy arrays:
        theta1: (100, 401)
        theta2: (25, 101)
        theta3: (10, 26)
        inputs: (5000, 400)
        labels: (5000, 1)
    '''
    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401

    time_start = time.time()
    hidden_layer_1 = Matrix_dot(input_layer, theta1.T)
    hidden_layer_1 = sigmoid(hidden_layer_1)
    hidden_layer_1 = np.insert(hidden_layer_1, 0, 1, axis=1)  # add bias, 5000x101
    time_end = time.time()
    if comm.rank == 0:
        print('\tconstruction: hidden layer 1 dot costs {} secs'.format(time_end - time_start))

    time_start = time.time()
    hidden_layer_2 = Matrix_dot(hidden_layer_1, theta2.T)
    hidden_layer_2 = sigmoid(hidden_layer_2)
    hidden_layer_2 = np.insert(hidden_layer_2, 0, 1, axis=1)  # add bias, 5000x26
    time_end = time.time()
    if comm.rank == 0:
        print('\tconstruction: hidden layer 2 dot costs {} secs'.format(time_end - time_start))


    time_start = time.time()
    output_layer = Matrix_dot(hidden_layer_2, theta2.T)  # 5000x10
    output_layer = sigmoid(output_layer)
    time_end = time.time()
    if comm.rank == 0:
        print('\tconstruction: output layer dot costs {} secs'.format(time_end - time_start))

    # forward propagation: calculate cost
    time_start = time.time()
    cost = 0.0
    for training_index in range(len(inputs)):
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1

        for k in range(output_layer_size):
            error = -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
            cost += error
    cost /= len(inputs)
    time_end = time.time()
    if comm.rank == 0:
        print('\tforward prop: costs {} secs'.format(time_end - time_start))

    
    # back propagation: calculate gradiants
    time_start = time.time()
    
    # 400 -> 10
    theta1_grad = np.zeros_like(theta1)  # 100x401
    theta2_grad = np.zeros_like(theta2)  # 25x101 
    theta3_grad = np.zeros_like(theta3)  # 10x26
    
    for index in range(len(inputs)):
        
        # transform label y[i] from a number to a vector : one hot
        outputs = np.zeros((1, output_layer_size))  # (1,10)
        outputs[0][labels[index]-1] = 1

        # calculate delta3
        delta3 = (output_layer[index] - outputs).T  # (10,1)

        # calculate delta2
        z2 = Matrix_dot(theta1, input_layer[index:index+1].T)  # (25,401) x (401,1)
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (26,1)
        delta2 = np.multiply(
            Matrix_dot(theta2.T, delta3),  # (26,10) x (10,1)
            sigmoid_gradient(z2)  # (26,1)
        )
        delta2 = delta2[1:]  # (25,1)

        # calculate gradients of theta1 and theta2
        # (25,401) = (25,1) x (1,401)
        theta1_grad += Matrix_dot(delta2, input_layer[index:index+1])
        # (10,26) = (10,1) x (1,26)
        theta2_grad += Matrix_dot(delta3, hidden_layer_1[index:index+1])
    
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)
    theta3_grad /= len(inputs)
    
    theta1_grad=np.sign(theta1_grad)
    theta2_grad=np.sign(theta2_grad)
    theta3_grad=np.sign(theta3_grad)

    l1, theta1_grad_cmp = compress(theta1_grad)
    l2, theta2_grad_cmp = compress(theta2_grad)
    l3, theta3_grad_cmp = compress(theta3_grad)

    # theta1_grad_d = decompress(theta1_grad_cmp, l1)
    # theta2_grad_d = decompress(theta2_grad_cmp, l2)
    # theta3_grad_d = decompress(theta3_grad_cmp, l3)

    comp1 = compare(theta1_grad_d, theta1_grad)
    comp2 = compare(theta2_grad_d, theta2_grad)
    comp3 = compare(theta3_grad_d, theta3_grad)

    # if(comp1):
    #     print("gradients 1 compressed and decompressed correctly")

    # if(comp2):
    #     print("gradients 2 compressed and decompressed correctly")

    # if(comp3):
    #     print("gradients 3 compressed and decompressed correctly")

    time_end = time.time()
    if comm.rank == 0:
        print('\tback prop: costs {} secs'.format(time_end - time_start))

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
        
        if comm.rank == 0:
            time_bcast_start = time.time()
        comm.Bcast([theta1, MPI.DOUBLE])
        comm.Barrier()
        comm.Bcast([theta2, MPI.DOUBLE])
        
        if comm.rank == 0:
            time_bcast_end = time.time()
            print('\tBcast theta1 and theta2 uses {} secs.'.format(time_bcast_end - time_bcast_start))
    else:
        
        theta1 = rand_init_weights(Input_layer_size, hidden_layer_1_size)
        theta2 = rand_init_weights(hidden_layer_1_size, hidden_layer_2_size)
        theta3 = rand_init_weights(hidden_layer_2_size, Output_layer_size)

    cost = 0.0
    for i in range(iteration):
        time_iter_start = time.time()

        if Distributed is True:
            # Scatter training data and labels.
            sliced_inputs = np.asarray(np.split(inputs, comm.size))
            sliced_labels = np.asarray(np.split(labels, comm.size))
            inputs_buf = np.zeros((len(inputs)//comm.size, Input_layer_size))
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
            # cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            cost, (theta1_grad_c, l1, theta2_grad_c, l2, theta3_grad_c, l3) = cost_function(theta1, theta2, theta3
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
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta1_grad, theta1_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta1 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta1_grad = functools.reduce(np.add, theta1_grad_buf) / comm.size

            theta2_grad_buf = np.asarray([np.zeros_like(theta2_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta2_grad, theta2_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta2 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta2_grad = functools.reduce(np.add, theta2_grad_buf) / comm.size

            theta3_grad_buf = np.asarray([np.zeros_like(theta3_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta3_grad, theta3_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta3 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta3_grad = functools.reduce(np.add, theta3_grad_buf) / comm.size

        else:
            # cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            cost, (theta1_grad_c, l1, theta2_grad_c, l2, theta3_grad_c, l3) = cost_function(theta1, theta2, theta3
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

        time_iter_end = time.time()
        if comm.rank == 0:
            print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}, time: {4}'.format(
                i+1, cost, learningrate, iteration, time_iter_end - time_iter_start)
            )
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


if __name__ == '__main__':
    
    Matrix_dot = np.dot

    # Note: There are 10 units which present the digits [1-9, 0]
    # (in order) in the output layer.
    inputs, labels = load_training_data()

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
