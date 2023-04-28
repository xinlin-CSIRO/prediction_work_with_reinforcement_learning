import new_bank
import numpy as np
import pandas as pd
import math

import io
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

from new_bank import prediction_environment_new

##data for forecasting##
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCI_labeled_.csv"
Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\European.csv"
ratings = pd.read_csv(Path_sorce_, header=0, usecols=[0])

dataset = ratings.values
all_length = len(dataset)
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
From_i = 39689
#European from 39689
#UCI from 62520 to 67520
num=0
last_setp_prediction_acceptable = 1
compensation=0
warning_level=0
look_days_ = 1
train_window =  96 #96 #int((look_days_ * 24 * 60) / 15)  # every 15 mins, so 7 days will be (24*7*60)/15=672-->96
window_prediction =  96 #96
analysis_back_steps=96
n_days= 49
# SQL-based
#nLevels = 10
lei_ji = 0
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\"
now = datetime.now()
current_time = str(now.strftime("%m_%d_%h_%m"))
# result_location_day_pred = result_location + 'classification_' + current_time + '.csv'
result_location_step = result_location + 'european_' + current_time + '_.csv'


# states = env_prediction.reset()
lower_p = 0
upper_p = 0

tran_std = 0

for mix in range(1):
    t_start = time.time()
    current_24_h, ahead_set, for_fig= dataset[(From_i - train_window):From_i], \
                                                     dataset[(From_i ):(From_i + 10)], \
                                                     dataset[(From_i-10 ):(From_i)]
    # test_forPlot = scaler.inverse_transform(ahead_set)
    real_data_observation = ahead_set[0][0]
    # back_real_data = scaler.inverse_transform(current_24_h)
    real_data_current =  current_24_h[window_prediction - 1][0]
    real_data_last =     current_24_h[window_prediction - 2][0]
    real_data_last_X_2 = current_24_h[window_prediction - 3][0]

    # trainX_listed, trainY_listed = _3rd_bank.listed_data_selector_opt(current_24_h, From_i, dataset, train_window, window_prediction)  # format=list
    trainX_listed, trainY_listed = new_bank.listed_data_selector_opt_new (current_24_h, From_i, dataset, train_window,
                                                                      window_prediction)  # format=list
    trainX_ = np.reshape(trainX_listed, (48, 96))
    trainX = np.array(trainX_listed)  # 7/96/1
    trainY = np.array(trainY_listed)

    lower, upper = new_bank.ONE_STEP_AHEAD_predictor (current_24_h, trainX, trainY,  train_window)

    # real_d = np.ones((2, len(for_fig)))
    for_fig1=np.reshape(for_fig, (1, 10))
    real_d=np.concatenate(( for_fig1,  for_fig1), axis=0)
    pred=np.array([lower, upper])
    da=np.concatenate((real_d, pred), axis=1)
    low=da[0,:]
    high=da[1,:]
    fig, ax = plt.subplots()
    x1 = np.linspace(1, 10,10)
    x2 = np.linspace(1, 11, 11)
    ax.plot(x1, for_fig,  linewidth=3, color='b')
    ax.fill_between(x2, low,high,  color='violet', alpha=.5)
    min_=8000
    max_=max(for_fig+100)
    ax.set_ylim(min_, max_)
    plt.xlabel('Time step')
    plt.ylabel('Power (kW)')
    plt.show()

    print('this is ', mix)
    From_i = From_i + 1