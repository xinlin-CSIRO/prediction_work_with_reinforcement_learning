import numpy as np
import pandas as pd
import new_bank
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import io
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
from attention import Attention
from keras import Input
from keras.models import load_model, Model

##data for forecasting##
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCI_labeled_.csv"
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\European.csv"
Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCSD_dataset.csv"
ratings = pd.read_csv(Path_sorce_, header=0, usecols=[0])

dataset = ratings.values
all_length = len(dataset)
source=Path_sorce_.split('\\')[-1]
if('UCI' in source):
    From_i = 62520
    data_source='UCI_'
elif('European' in source):
    From_i = 39689
    data_source = 'European_'
elif ('UCSD' in source):
    From_i = 62520
    data_source = 'UCSD_'
ratings = pd.read_csv(Path_sorce_, header=0, usecols=[0])



dataset = ratings.values
dataset_initial = ratings.values
all_length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

compensation=0
warning_level=0
look_days_ = 1
base_window_prediction = 96
analysis_back_steps=10
n_days=49
# SQL-based
#nLevels = 10
lei_ji=0
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\LSTM_att_"
now = datetime.now()
current_time = str(now.strftime("%Y_%m_%d"))
result_location_step = result_location +  data_source +  '_49days.csv'

for mix in range(5000):
    if (mix == 0):
        f_one_step = open(result_location_step, "w+")
    else:
        f_one_step = open(result_location_step, "a+")
    t_start = time.time()
    current_24_h, ahead_set = dataset[(From_i - base_window_prediction):From_i], \
                                                     dataset[(From_i ):(From_i + base_window_prediction)]
    trainX, trainY = new_bank.moving_window(From_i, dataset, base_window_prediction,n_days)  # format=list
    #trainX, trainY, trainX_ross, trainY_ross, extracted_data, labels, break_index, Scores = _3rd_bank.training_data_selector_(
        #current_24_h, From_i, dataset, base_window_prediction, n_days)


    back_real_data = scaler.inverse_transform(current_24_h)
    real_data_current = back_real_data[base_window_prediction - 1][0]

    real_data_observation = dataset_initial[From_i]
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    look_forward=1

    n_samples = trainX.shape[0]
    time_step = trainX.shape[1]
    n_layers = n_samples

    n_layers = n_samples
    loss_ = 'mean_squared_error'
    neurals = time_step
    model_input = Input(batch_input_shape=(n_samples, time_step, 1))
    x = LSTM(neurals, return_sequences=True)(model_input)
    x = Attention(units=time_step)(x)
    x = Dense(100*look_forward)(x)
    x = Dense(10 * look_forward)(x)
    x = Dense(look_forward)(x)
    model = Model(model_input, x)
    model.compile(loss=loss_, optimizer='adam')
    #print(model.summary())
    model.fit(trainX, trainY, epochs=100, batch_size=n_samples, verbose=0)

    # re-define model
    model_input_new = Input(batch_input_shape=(1, time_step, 1))
    x_new = LSTM(neurals, return_sequences=True)(model_input_new)
    x_new = Attention(units=time_step)(x_new)
    x_new = Dense(100 * look_forward)(x_new)
    x_new = Dense(10 * look_forward)(x_new)
    x_new = Dense(look_forward)(x_new)
    model_new = Model(model_input_new, x_new)

    # copy weights
    old_weights = model.get_weights()
    model_new.set_weights(old_weights)
    model_new.compile(loss=loss_, optimizer='adam')
    current_24_h = np.reshape(current_24_h, (current_24_h.shape[1], current_24_h.shape[0], 1))
    pred1 = model_new.predict(current_24_h)
    prediction_result = scaler.inverse_transform(pred1)

    # if(anomaly_!=1):
    t_record = time.time() - t_start

    s_result =str(prediction_result[0]) + ',' + str(real_data_observation) + ', ' + str(t_record) +'\n'
    f_one_step.write(s_result.replace('[', '').replace(']', ''))
    f_one_step.close()
    print('this is the ', mix)
    From_i = From_i + 1
