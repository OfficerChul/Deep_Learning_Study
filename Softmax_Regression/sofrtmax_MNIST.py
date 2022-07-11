"""
* FILENAME : softmax_MNIST.py
*
* DESCRIPTION : softmax regressing using MNIST dataset and tensorflow v2.9.1
*
* AUTHOR : Kyochul Jang
* EMAIL: jang128@purdue.edu
* START DATE : 4 July 2022
"""
#%% Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

#%% One-hot Encoding for lables
def one_hot_encoding(labels, class_num):
    data_num = len(labels[1]) # 60000, 10000
    zeroes = np.zeros((data_num, 10))
    
    for idx in range(data_num):
        zeroes[idx][labels[1][idx]] = 1
        
    return zeroes
    
#%% Initialize Variables

# Load MNIST dataset from implementing Keras API
mnist_train, mnist_test = mnist.load_data() # 60000 training set, 10000 testing set

# Reshape train data and test data
re_mnist_train = mnist_train[0].reshape((60000, 28*28)) # (60000, 784)
re_mnist_test = mnist_test[0].reshape((10000, 28*28)) # (10000, 784)

# X, W, and b
X = re_mnist_train
W = np.zeros((784, 10))
b = 0

#%% Get labels by using one_hot encoding function
labels = one_hot_encoding(mnist_train, class_num)

#%% Use Softmax Func




