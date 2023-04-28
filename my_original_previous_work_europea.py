import original_3rd_bank
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

##data for forecasting##
# Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\UCI_labeled_.csv"
Path_sorce_ = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\European.csv"
ratings = pd.read_csv(Path_sorce_, header=0, usecols=[0])
label_=pd.read_csv(Path_sorce_, header=0, usecols=[0])
label_ = label_.values

dataset = ratings.values
dataset_initial = ratings.values
all_length = len(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
From_i = 39689
#European from 39689
#UCI from 62520 to 67520
num=0
n_repeat_times =200
last_setp_prediction_acceptable = 1
compensation=0
warning_level=0
look_days_ = 1
base_size_of_window =  4 #96 #int((look_days_ * 24 * 60) / 15)  # every 15 mins, so 7 days will be (24*7*60)/15=672-->96
base_window_prediction =  96
analysis_back_steps=10
n_days=7
# SQL-based
#nLevels = 10
lei_ji=0
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\"
now = datetime.now()
current_time = str(now.strftime("%m_%d"))
result_location_day_pred = result_location + 'classification_european' + current_time + '.csv'
result_location_step = result_location + 'original_Europea_' + current_time + '.csv'

connection = sqlite3.connect('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\db\\mydb.db')
cur = connection.cursor ()
cur.execute("DROP TABLE IF EXISTS prediction2")
print("Table dropped... ")
sql_create_projects_table =  "CREATE TABLE prediction2 ( id integer PRIMARY KEY,   real_load string, X_ross string, Y_ross string,  X string, Y string,  Extracted string, Labels string, break_index integer, scenario integer); "
cur.execute(sql_create_projects_table)
connection.commit()
print("Table generated... ")

scenario = 0
lei_ji=0
res_=0
initial_alpha=0.6
previous_result=np.zeros(base_size_of_window)
alpha_p=0
alpha_max=0.7
for mix in range(5000):

    t_start = time.time()
    anomaly_ = 0
    acc_ = 0
    daily_max = -100000
    daily_min  =  100000
    current_24_h_initial=dataset_initial[(From_i - base_size_of_window):From_i]
    current_24_h, ahead_set = dataset[(From_i - base_window_prediction):From_i], \
                                                     dataset[(From_i ):(From_i + base_window_prediction)]
    trainX, trainY,trainX_ross, trainY_ross,extracted_data, labels, break_index, Scores = original_3rd_bank.training_data_selector_(current_24_h, From_i, dataset, base_window_prediction,n_days)  # format=list
    #train_list_r, test_list_r, train_list, test_list, index, Scores
    Labels=np.array2string(labels)
    test_forPlot = scaler.inverse_transform(ahead_set)
    Mid=''
    X=''
    Y = ''
    X_ross=''
    Y_ross=''
    for x in trainX:
        x=str(x)
        X =X+x+'-'
    for x in trainY:
        x=str(x)
        Y = Y+x+'-'
    for x in trainX_ross:
        x=str(x)
        X_ross = X_ross+x+'-'
    for x in trainY_ross:
        x=str(x)
        Y_ross = Y_ross+x+'-'
    for x in extracted_data:
        x=str(x)
        Mid = Mid+x+'-'
    back_real_data = scaler.inverse_transform(current_24_h)
    real_data_current = back_real_data[base_window_prediction - 1][0]
    current_label_24_h= label_[(From_i - base_window_prediction):From_i]
    real_label_current=current_label_24_h[base_window_prediction - 1][0]
    real_data_observation = test_forPlot[0][0]
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    a = trainY.shape[0] * trainY.shape[1]
    tem = np.reshape(trainY, (a, 1))
    trainY_reshape = scaler.inverse_transform(tem)
    trainY_reshape = np.reshape(trainY_reshape, (trainY.shape[0], trainY.shape[1]))
    tranY_min  = np.zeros(n_days)
    tranY_max = np.zeros(n_days)
    tranY_location_max=np.zeros([n_days])
    tranY_location_min = np.zeros([n_days])
    for x in range(0, trainY.shape[0]):
        for y in range(0, trainY.shape[1]):
            if (daily_max < trainY_reshape[x][y]):
                daily_max = trainY_reshape[x][y]
                tranY_location_max[x]=y
            if (daily_min  > trainY_reshape[x][y]):
                daily_min  = trainY_reshape[x][y]
                tranY_location_min[x] = y
            if(y==trainY.shape[0]-1):
                daily_max = -100000
                daily_min = 100000
        tranY_min[x] =daily_min
        tranY_max[x] = daily_max


    max_diff=np.zeros(n_days)
    min_diff = np.zeros(n_days)
    for x in range(0, trainY.shape[0]):
        if(tranY_location_max[x]!=95) and (tranY_location_max[x]!=0):
                diff_1=tranY_max[x]- trainY_reshape[x][int(tranY_location_max[x]-1)]
                diff_2 = tranY_max[x] - trainY_reshape[x][int(tranY_location_max[x] + 1)]
                max_diff[x]=max(diff_1,diff_2)
        elif(tranY_location_max[x]==95):
            diff_1 = tranY_max[x] - trainY_reshape[x][int(tranY_location_max[x] - 1)]
            max_diff[x] = diff_1
        elif(tranY_location_max[x]==0):
            diff_1 = tranY_max[x] - trainY_reshape[x][int(tranY_location_max[x] + 1)]
            max_diff[x] = diff_1
        if (tranY_location_min[x] != 95) and (tranY_location_min[x]!= 0):
                diff_1 = -tranY_min[x] + trainY_reshape[x][int(tranY_location_min[x] - 1)]
                diff_2 = -tranY_min[x] +trainY_reshape[x][int(tranY_location_min[x] + 1)]
                min_diff[x] = max(diff_1, diff_2)
        elif(tranY_location_min[x]==95):
            diff_1 = -tranY_min[x] + trainY_reshape[x][int(tranY_location_min[x] - 1)]
            min_diff[x] = diff_1
        elif (tranY_location_min[x] == 0):
            diff_1 = tranY_min[x] - trainY_reshape[x][int(tranY_location_max[x] + 1)]
            min_diff[x] = diff_1
    max_diff_final=np.max(max_diff)
    min_diff_final = np.max(min_diff)
    tranY_mean_max=np.mean(tranY_max)
    tranY_mean_min = np.mean(tranY_min)
    tranY_std_max=np.std(tranY_max)
    tranY_std_min = np.std(tranY_min)
    Final_max=tranY_mean_max+3*tranY_std_max
    Final_min=tranY_mean_min-3*tranY_std_min
    #Final_max =  daily_max
    #Final_min  =  daily_min
    if (mix == 0):
        f_one_step = open(result_location_step, "w+")
        f_load = open(result_location_day_pred, "w+")
        _x_=int( -100*np.log((1-initial_alpha)/initial_alpha))
        _x_initial=_x_
    else:
        f_one_step = open(result_location_step, "a+")
        f_load = open(result_location_day_pred, "a+")
        #previous results check, so start from the 2nd step
        # important prediction part
        scenario, compensation=original_3rd_bank.previous_prediction_compensation(predictor_lower,predictor_upper, real_data_current,Final_max, Final_min)
        #print ('compensation= ',compensation)

        if(scenario==1):
                if(_x_>-500):
                    if(mix>analysis_back_steps):
                       sqlite_query__ = """select scenario from  prediction2 where id<=(?) and id> (?)"""
                       cur.execute(sqlite_query__, (mix,(mix-analysis_back_steps)))
                       records = cur.fetchmany(analysis_back_steps)
                       records=np.array(records)
                       time_s=1
                       for a_i in range(len(records) - 1, -1, -1):
                           if(records[a_i]==scenario):
                               time_s=time_s+1
                           else:
                               break
                       _x_=_x_-time_s

                    else:
                        _x_ = _x_ - 1
        elif  (scenario == 2):
            if (alpha_p < alpha_max):
                if (mix > analysis_back_steps):
                    sqlite_query__ = """select scenario from  prediction2 where id<=(?) and id> (?)"""
                    cur.execute(sqlite_query__, (mix, mix - analysis_back_steps))
                    records = cur.fetchmany(analysis_back_steps)
                    records = np.array(records)
                    time_s = 0
                    for a_i in range(len(records) - 1, -1, -1):
                        if  (records[a_i] == scenario):
                            time_s = time_s + 1
                        else:
                            break
                    _x_ = _x_ + time_s  # turns back to
            else:
                _x_ = _x_
        elif(scenario==3) or (scenario==4):
                if (_x_ > -500):
                    if (mix > analysis_back_steps):
                        sqlite_query__ = """select scenario from  prediction2 where id<=(?) and id> (?)"""
                        cur.execute(sqlite_query__, (mix, mix -analysis_back_steps))
                        records = cur.fetchmany(analysis_back_steps)
                        records = np.array(records)
                        time_s = 1
                        for a_i in range(len(records) - 1, -1, -1):
                            if (records[a_i] ==3) or (records[a_i] == 4):
                                time_s = time_s + 1
                        _x_ = _x_ - time_s*10
                    else:
                        _x_ = _x_ - 10
                else:
                    _x_ = _x_
        elif(scenario==5):
            sqlite_query__ = """select real_load from  prediction2 where id=?"""
            cur.execute(sqlite_query__, ((mix),))
            records = cur.fetchone()
            records = np.array(records)
            records = records[0]
            diff_current=real_data_current-records

            if(diff_current>max_diff_final):
                  anomaly_ = 1
                  _x_ = _x_initial

        elif (scenario == 6):
            sqlite_query__ = """select real_load from  prediction2 where id=?"""
            cur.execute(sqlite_query__, ((mix),))
            records = cur.fetchone()
            records = np.array(records)
            records = records[0]
            diff_current = real_data_current - records
            if (diff_current > min_diff_final):
                anomaly_ = 1
                _x_ = _x_initial

    t_r = time.time() - t_start
    print('time=mind', t_r)

    #X_set= _3rd_bank.previous_data_generator_(current_24_h_initial, From_i, dataset_initial, base_size_of_window,n_days)
    #diff_class = _3rd_bank.diversity_(X_set, current_24_h_initial)  #
    #print('Now, dis_class result = ', diff_class)
    #s_result = str(From_i+2) + ',' + str(real_data_observation) + ',' + str(diff_class)   + ','  + '\n'
    #f_load.write(s_result.replace('[', '').replace(']', ''))
    #f_load.close()
    # important prediction part
    lower, upper,  alpha= original_3rd_bank.ONE_STEP_AHEAD_predictor(current_24_h, trainX, trainY ,_x_,initial_alpha)
    alpha_p=alpha
    res_ = np.array([lower, upper])
    res_ = scaler.inverse_transform(res_)
    lower = res_[0, 0]
    upper = res_[1, 0]
    predictor_upper = upper
    predictor_lower = lower
    if(scenario==1) or (scenario==0):
      lower_record = lower
      upper_record = upper
    elif(scenario==2) or (scenario==5) or (scenario==6):
        lower_record =  lower-compensation
        upper_record = upper-compensation
    elif (scenario == 3):
        lower_record = daily_min
        upper_record = upper - compensation
    elif (scenario == 4):
        lower_record = lower - compensation
        upper_record = daily_max

    t_record = time.time() - t_start
    print('time=end',t_record)
    difference_=upper_record-lower_record
    if(difference_<0):
        u_=lower_record
        l_=upper_record
        upper_record=u_
        lower_record=l_
        difference_=-difference_
    # if(anomaly_!=1):
    s_result = str(scenario) + ',' + str(alpha_p) + ',' + str(predictor_lower) + ',' + str(
                    predictor_upper) + ', ' + str(lower_record) + ',' + str(upper_record) + ', ' + str(
                    real_data_observation) + ', ' + str(_x_) + ','+str(anomaly_)+','+str(real_label_current)+','+str(difference_)\
                    +','+str(Scores)+ ','+ str(t_record)+',' +str(Final_min)+','+str(Final_max)+'\n'
    f_one_step.write(s_result.replace('[', '').replace(']', ''))
    f_one_step.close()
    print('this is the ', mix)
    scenario_perious=scenario
    X=X.replace('[','')
    X = X.replace(']', '')
    X_ross = X_ross.replace('[', '')
    X_ross = X_ross.replace(']', '')
    Y = Y.replace('[', '')
    Y = Y.replace(']', '')
    Y_ross = Y_ross.replace('[', '')
    Y_ross = Y_ross.replace(']', '')
    Mid = Mid.replace('[', '')
    Mid = Mid.replace(']', '')
    #mystring = mystring.replace(' ', ';')

    sqlite_insert_with_param = """INSERT INTO prediction2
                              (id,  real_load,  X_ross , Y_ross,  X,   Y,   Extracted,   Labels,  break_index, scenario)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""
    uu=mix+1
    data_tuple = (uu,  real_data_current, X_ross,  Y_ross,   X,     Y,   Mid,   Labels,  break_index, scenario)
    cur.execute(sqlite_insert_with_param, data_tuple)
    connection.commit()
    #print('data inserted')
    From_i = From_i + 1
