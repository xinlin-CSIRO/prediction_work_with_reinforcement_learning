import new_bank
import numpy as np
import pandas as pd
import math
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
from tensorforce.execution import Runner
import tensorforce
import tensorflow
from new_bank import prediction_environment_new
from tensorforce import Agent, Environment
##data for forecasting##
Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCI_labeled_.csv"
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\European.csv"
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCSD_dataset.csv"
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
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\new_approach_"
now = datetime.now()
current_time = str(now.strftime("%H_%M_%S"))
# result_location_day_pred = result_location + 'classification_' + current_time + '.csv'
result_location_step = result_location + data_source + current_time + '_AC_.csv'


new_states_address='C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\db\\mydb.db'
env_prediction = prediction_environment_new(new_states_address)
env = Environment.create(environment=env_prediction, max_episode_timesteps=10000)
agent = Agent.create(agent='ac', environment=env, memory=60000, batch_size=10, learning_rate=1e-3)
# states = env_prediction.reset()
lower_p = 0
upper_p = 0

tran_std = 0


for mix in range(5000):
    t_start = time.time()
    current_24_h, ahead_set = dataset[(From_i - train_window):From_i], \
                                                     dataset[(From_i ):(From_i + window_prediction)]
    # test_forPlot = scaler.inverse_transform(ahead_set)
    real_data_observation = ahead_set[0][0]
    # back_real_data = scaler.inverse_transform(current_24_h)
    real_data_current =  current_24_h[window_prediction - 1][0]
    real_data_last =     current_24_h[window_prediction - 2][0]
    real_data_last_X_2 = current_24_h[window_prediction - 3][0]

    # trainX_listed, trainY_listed = _3rd_bank.listed_data_selector_opt(current_24_h, From_i, dataset, train_window, window_prediction)  # format=list
    trainX_listed, trainY_listed = new_bank.listed_data_selector_opt_new (current_24_h, From_i, dataset, train_window,
                                                                      window_prediction)  # format=list
    Previous_predicted_states = np.reshape(np.array([lower_p, upper_p]),[1,2])
    current_observation = np.reshape(np.array([real_data_last, real_data_current]), [1, 2])
    trainX_ = np.reshape(trainX_listed, (48, 96))
    train_diff = trainX_[:,[-2,-1]]

    Frist_diff = train_diff[:, 1] - train_diff[:, 0]
    q1, q3 = np.percentile(Frist_diff, [25, 75])


    new_states = np.array([lower_p, upper_p, real_data_last, real_data_current, q1, q3])
    # new_states= np.concatenate((train_diff, current_observation,Previous_predicted_states ), axis=0)

    if (mix == 0):
        f_one_step = open(result_location_step, "w+")
        f_one_step.write('Raw_low,Raw_high, r_lower, r_high, real, fb_svr, action, reward, time,case \n')


        # f_states = open(new_states_address, "w+")
        # # f_states.write(s_)
        # # f_states.write(s_)
        # f_states.close()

    else:
        f_one_step = open(result_location_step, "a+")

        # f_states = open(new_states_address, "a+")
        # # f_states.write(s_)
        # f_states.close()

    connection = sqlite3.connect(new_states_address)
    cur = connection.cursor()
    cur.execute("DROP TABLE IF EXISTS new_state")
    # print("Table dropped... ")
    sql_create_projects_table = "CREATE TABLE new_state ( lower_p integer, higher_p  integer,  real_last integer, real_now  integer,  q_1 integer, q_3  integer); "
    cur.execute(sql_create_projects_table)
    connection.commit()
    # print("Table generated... ")
    # for x in range (len(new_states)):
    #     last,now =new_states[x,0],new_states[x,1]
    sqlite_insert_ = """INSERT OR IGNORE INTO new_state (lower_p, higher_p,  real_last, real_now,  q_1, q_3) VALUES (?, ?, ?,?,?,?);"""
    data_tuple = (lower_p, upper_p, real_data_last, real_data_current,q1, q3)  # (t-1, t, t)--> diff at t
    cur.execute(sqlite_insert_, data_tuple)
    connection.commit()

    # print (1)
    trainX, trainY, actions,reward, case = new_bank. RL_BASED_TRAINING_DATA_SELECTOR (trainX_listed, trainY_listed, env_prediction, agent, new_states)

    trainX=np.array(trainX) # 7/96/1
    trainY=np.array(trainY)




    lower, upper = new_bank.ONE_STEP_AHEAD_predictor(current_24_h, trainX, trainY,  train_window)
    fb_svr = new_bank.Evaluation_3(real_data_current, real_data_last, real_data_last_X_2, lower_p, upper_p, lower, upper, mix)
    lower_p = lower[0]
    upper_p = upper[0]
    if (mix > analysis_back_steps):
        lower_record_svr = lower + fb_svr
        upper_record_svr = upper + fb_svr
    else:
        lower_record = lower
        upper_record = upper
        lower_record_svr = lower
        upper_record_svr = upper

    t_record = time.time() - t_start
    s_result = str(lower) + ',' + str(upper) + ',' + str(lower_record_svr) + ',' + str(upper_record_svr)  + ',' + str(real_data_observation) + ',' + str(fb_svr) + ',' + str(actions) + ',' + str(reward) +',' + str(t_record) +','+str(case)+ '\n'
    f_one_step.write(s_result.replace('[', '').replace(']', ''))
    f_one_step.close()
    print('this is ', mix)
    From_i = From_i + 1