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

#%% Version check tensorflow-cpu
print('$ tensorflow version:', tf.__version__)
print('$ Eagerly execution: ', tf.executing_eagerly())

#%% Load MNIST dataset from implementing Keras API
mnist = tf.keras.datasets.mnist
a = 6
mnist

