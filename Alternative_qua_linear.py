import numpy as np
import pandas as pd
import new_bank
from keras.models import Sequential
from keras.layers import  Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import io
import time
from sklearn.metrics import accuracy_score
from sklearn.utils.fixes import sp_version, parse_version
import sqlite3
from datetime import datetime
from attention import Attention
from keras import Input
from keras.models import load_model, Model
from quantile_forest import RandomForestQuantileRegressor
from keras.layers import Input, LSTM, Dense, Bidirectional
from sklearn.linear_model import QuantileRegressor


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
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\Qutail_linear_"
now = datetime.now()
current_time = str(now.strftime("%Y_%m_%d"))
result_location_step = result_location +  data_source +  '_49days.csv'


scenario = 0
lei_ji=0
res_=0
alpha_p=0
alpha_max=0.7
for mix in range(5000):
    if (mix == 0):
        f_one_step = open(result_location_step, "w+")
        s_ =  'low , high, real, t_record\n'
        f_one_step.write(s_)
    else:
        f_one_step = open(result_location_step, "a+")
    t_start = time.time()
    current_24_h, ahead_set = dataset[(From_i - base_window_prediction):From_i], \
                                                     dataset[(From_i ):(From_i + base_window_prediction)]
    trainX, trainY = new_bank.moving_window(From_i, dataset, base_window_prediction,n_days)  # format=list
    #trainX, trainY, trainX_ross, trainY_ross, extracted_data, labels, break_index, Scores = _3rd_bank.training_data_selector_(
        #current_24_h, From_i, dataset, base_window_prediction, n_days)

    look_forward = 1
    back_real_data = scaler.inverse_transform(current_24_h)
    real_data_current = back_real_data[base_window_prediction - 1][0]

    real_data_observation = dataset_initial[From_i]
    trainX = np.reshape(np.array(trainX),[n_days,base_window_prediction])
    trainY = np.reshape(np.array(trainY),[n_days,look_forward])

    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    reg = QuantileRegressor(quantile=0.7, solver=solver).fit(trainX, trainY)
    current_24_h = np.reshape(current_24_h, (current_24_h.shape[1], current_24_h.shape[0]))
    y_high = reg.predict(current_24_h)

    reg2 = QuantileRegressor(quantile=0.3, solver=solver).fit(trainX, trainY)
    y_low = reg2.predict(current_24_h)



    prediction_result = scaler.inverse_transform([y_low, y_high])
    # prediction_result=np.reshape(prediction_result,(prediction_result.shape[1], prediction_result.shape[0]))
    t_record = time.time() - t_start

    s_result =str(prediction_result[0]) + ',' +str(prediction_result[1]) + ',' + str(real_data_observation) + ', ' + str(t_record) +'\n'
    f_one_step.write(s_result.replace('[', '').replace(']', ''))
    f_one_step.close()
    print('this is the ', mix)
    From_i = From_i + 1
