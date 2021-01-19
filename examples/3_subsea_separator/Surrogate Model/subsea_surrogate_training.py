# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:26:21 2021

@author: wilhe
"""

# Import general scientific computing packages into enviroment.
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import pickle
import os.path
from os import path
import csv
import statistics
import matplotlib.pyplot as plt

import sklearn.preprocessing as preprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

initial_path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Raw Training Data\\Subsea Surrogate\\TrainingData.csv"

col_names = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
flabel = ["x1", "x2", "x3", "x4"]
tlabel = ["x5", "x6", "x7", "x8"]

split_ratio = 0.15
test_ratio = 0.15

training_path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Test Train Storage\\Subsea Surrogate\\training_set.csv"
test_path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Test Train Storage\\Subsea Surrogate\\test_set.csv"

split_exists = path.exists(training_path) & path.exists(test_path)

if split_exists: 
    train_set = pd.read_csv(training_path).astype(float)
    test_set = pd.read_csv(test_path).astype(float)
else:
    df = pd.read_csv(initial_path, names = col_names).astype(float)   # load data frame
    df = df.sample(frac = 1)
    scaler = preprocess.MinMaxScaler()                      # scale data
    df[col_names] = scaler.fit_transform(df[col_names])

    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= (1.0 - test_ratio)
    train_set = df[msk]
    test_set = df[~msk]
    train_set.to_csv(training_path)
    test_set.to_csv(test_path)
    
x_train = pd.DataFrame(train_set, columns=flabel)
y_train = pd.DataFrame(train_set, columns=tlabel)
    
x_test = pd.DataFrame(test_set, columns=flabel)
y_test = pd.DataFrame(test_set, columns=tlabel)

# Create basic sequential model
model = tf.keras.Sequential()
                
# Add input layer
model.add(layers.Input(shape = (4,)))           
      
for q in range(0, 2):
    model.add(layers.Dense(12, tf.nn.silu))
    #model.add(layers.Dropout(0.8))

# Add a dense output layer of neurons
#model.add(layers.Dense(1))
model.add(layers.Dense(4, 'sigmoid'))
                
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-1), loss = 'mse')

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 1e-1
	drop = 0.5
	epochs_drop = 100.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = keras.callbacks.LearningRateScheduler(step_decay)
callbacks_list = [lrate]

fr = model.fit(x_train, y_train, epochs = 1000, callbacks=callbacks_list, validation_data = (x_test, y_test))

weight_calced = model.get_weights()