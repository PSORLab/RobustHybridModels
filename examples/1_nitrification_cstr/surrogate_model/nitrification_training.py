# Copyright (c) 2021: Matthew Wilhelm, Chenyu Wang, Matthew Stuber, 
# and the University of Connecticut (UConn).
# This code is licensed under the MIT license (see LICENSE.md for full details).
#################################################################################
# RobustHybridModels
# Examples from the paper Semi-Infinite Optimization with Hybrid Models
# https://github.com/PSORLab/RobustHybridModels
#################################################################################
# examples/1_nitrification_cstr/surrogate_model/nitrification_training.py
# This file uses Keras to create a neural network for the ammonia oxidation rate
# for use in examples/1_nitrification_cstr/nitrification_surrogate.jl.
#################################################################################
"""
Created on Sun Jan 10 09:26:21 2021

@author: Matthew Wilhelm
"""

# Import general scientific computing packages into environment
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

folder_path = "examples/1_nitrification_cstr/surrogate_model"
data_path = "nitrification_training_data"

col_names = ['x1', 'x2', 'x3']
flabel = ['x1', 'x2']
tlabel = ['x3']

split_ratio = 0.15
test_ratio = 0.15

name = data_path+".csv"
training_path = folder_path+"test_train_split/"+data_path+"_training_set.csv"
test_path = folder_path+"test_train_split/"+data_path+"_test_set.csv"

split_exists = path.exists(training_path) & path.exists(test_path)

if split_exists: 
    train_set = pd.read_csv(training_path).astype(float)
    test_set = pd.read_csv(test_path).astype(float)
else:
    # Load data frame
    df = pd.read_csv(folder_path+name, names = col_names).astype(float)
    df = df.sample(frac = 1)
    # Scale data
    scaler = preprocess.MinMaxScaler()
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
model.add(layers.Input(shape = (2,)))           
      
for q in range(0, 2):
    model.add(layers.Dense(8, tf.nn.tanh))

# Add a dense output layer of neurons
model.add(layers.Dense(1, 'sigmoid'))
                
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-1), loss = 'mse')

# Learning rate schedule
def step_decay(epoch):
	initial_lrate = 1e-2
	drop = 0.5
	epochs_drop = 100.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = keras.callbacks.LearningRateScheduler(step_decay)
callbacks_list = [lrate]

fr = model.fit(x_train, y_train, epochs = 500, callbacks=callbacks_list, validation_data = (x_test, y_test))

def Ammonia_oxidation_rate(x):
    c_NH = x['x1'].to_numpy()
    c_O = x['x2'].to_numpy()
    r_AOmax = 0.67/60
    K_OAO = 0.3
    K_SAO = 0.24
    K_IAO = 6200
    r_AO = r_AOmax
    r_AO = r_AO*(c_NH/(K_SAO + c_NH + c_NH**2/K_IAO))
    r_AO = r_AO*(c_O/(K_OAO + c_O))
    return r_AO

y_all_predicted = np.squeeze(model.predict(x_test))
x_test_scaled = pd.DataFrame(test_set, columns=flabel)
x_test_scaled['x1'] = x_test['x1']*40.0
x_test_scaled['x2'] = x_test['x2']*9.1
 
y_all_true = Ammonia_oxidation_rate(x_test_scaled)/0.638966464124928
y_all_dev =  y_all_predicted - y_all_true

plt.scatter(y_all_true, y_all_predicted)
plt.show()

max_dev = max(y_all_dev)
avg_dev = statistics.mean(y_all_dev)
min_dev = min(y_all_dev)

y_true_ll = []
y_pred_ll = []
for i in range(0, len(y_all_predicted)):
    y_true_ll.append([y_all_true[i]])
    y_pred_ll.append([y_all_predicted[i]])

mse_loss = tf.keras.losses.MeanSquaredError()
percent_rel_dev = max(100*abs(y_all_dev)/y_all_true)

weight_calced = model.get_weights()

print(f"""Maximum deviation of rate = {max_dev}, \n
          Minimum deviation of rate = {min_dev}, \n
          Average deviation of rate = {avg_dev}, \n 
          Percent relative deviation = {percent_rel_dev}""")