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

col_names = ['x1', 'x2', 'x3','x4']
flabel = ['x1', 'x2', 'x3']
tlabel = ['x4']

split_ratio = 0.15
test_ratio = 0.15

training_path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Test Train Storage\\Polymer\\training_set.csv"
test_path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Test Train Storage\\Polymer\\test_set.csv"
initial_path = "C:\\Users\\wilhe\\Dropbox\\My PC (DESKTOP-P6322LG)\\Desktop\\Final Training Work\\Raw Training Data\\Polymer\\TrainingData.csv"

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
model.add(layers.Input(shape = (3,)))           
      
for q in range(0, 3):
    model.add(layers.Dense(6, tf.nn.gelu))

# Add a dense output layer of neurons
#model.add(layers.Dense(1))
model.add(layers.Dense(1, 'sigmoid'))
                
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-1), loss = 'mse')

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 1e-2
	drop = 0.5
	epochs_drop = 100.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = keras.callbacks.LearningRateScheduler(step_decay)
callbacks_list = [lrate]

fr = model.fit(x_train, y_train, epochs = 500, callbacks=callbacks_list, validation_data = (x_test, y_test))

"""
def Ammonia_oxidation_rate(x):
    c_NH = x['x1'].to_numpy()#*40.0
    c_O = x['x2'].to_numpy()#*9.1
    r_AOmax = 0.67
    K_OAO = 0.3
    K_SAO = 0.34
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

plt.scatter(y_all_true,y_all_predicted)
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
"""

weight_calced = model.get_weights()


#print(f"""Maximum deviation of rate = {max_dev}, \n
       #   Minimum deviation of rate = {min_dev}, \n
       #   Average deviation of rate = {avg_dev}, \n 
       #   Percent relative deviation = {percent_rel_dev}""")