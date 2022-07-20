"""
* FILENAME : XOR_MLP.py
*
* DESCRIPTION : Implement XOR gates using MLP
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

#%% Initialize Variables

# X
x_data = [
    
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
    
    ]

y_data = [
    
    [0],
    [1],
    [1],
    [0]
    
    ]

#%% 

