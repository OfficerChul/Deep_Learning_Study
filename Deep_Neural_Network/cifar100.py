"""
* FILENAME : cifar100.ipynb
*
* DESCRIPTION : classify cifar100 dataset in 100 classes using MLP (Multi-Layer Perceptron)
*
* AUTHOR : Kyochul Jang
* EMAIL: jang128@purdue.edu
* START DATE : 20 July 2022
"""

#%% Import Library
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#%% Import CIFAR100 Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train, x_test = x_train/255, x_test/255

print(f'$ x_train: {x_train.shape}')
print(f'$ y_train: {y_train.shape}')
print(f'$ x_test: {x_test.shape}')
print(f'$ y_test: {y_test.shape}')

#%% Define Functions

def draw_image():
    plt.figure(figsize=(10,10))
    for idx, img in enumerate(np.random.randint(0, 9999, 25)):
        plt.subplot(5, 5, idx+1)
        plt.imshow(x_train[img])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    plt.show()
    
#%% Implement

# Draw Images
draw_image()

# One-hot Encoding



