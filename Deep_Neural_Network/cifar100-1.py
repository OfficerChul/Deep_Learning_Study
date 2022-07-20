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
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras 
from keras.layers import Dense

#%% Import CIFAR100 Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
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
    
#%% Reshape

# Draw Images
draw_image()

dim2 = 32*32*3
x_train = x_train.reshape(len(x_train), dim2)
x_test = x_test.reshape(len(x_test), dim2)
    
#%% Implement



# One-hot Encoding
y_test = pd.get_dummies(i for i in y_test.T[0])
y_train = pd.get_dummies(i for i in y_train.T[0])

# Initialize weight and b
W = np.full((3072, 1), 0.0)
b = np.full((100, 1), 0.0)

#%% Build Model
model = keras.Sequential()

# Add first layer
model.add(Dense(512, activation='relu', input_shape=(3072, )))


# Add second layer
model.add(Dense(100, activation='softmax'))

#%% Model Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%% model Fit
model_fit = model.fit(x_train, y_train, batch_size=500, epochs=100, verbose=1)

#%% model evaluate
score = model.evaluate(x_test, y_test, verbose=0)
pred = model.predict(x_test)
np.argmax(pred[0])
np.argmax(y_test.T[0])
















