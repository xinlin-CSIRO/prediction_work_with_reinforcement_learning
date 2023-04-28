# import pandas as pd
# import numpy as np
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error as mae
# import time
# import os
# import math
# import matplotlib.pyplot as plt
# from datetime import datetime
# from tslearn.barycenters import  dtw_barycenter_averaging
#
# def rmse(predictions, targets):
#     return np.sqrt(np.mean((predictions - targets) ** 2))
# def mape(actual, pred):
#     actual, pred = np.array(actual), np.array(pred)
#     return np.mean(np.abs((actual - pred) / actual)) * 100
# def smape(act,forc):
#     return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))
# def tic (act,forc):
#     leng=len(act)
#     useless=np.zeros(leng)
#     upper= mae(act,forc)
#     lower=mae(act,useless)+mae(forc,useless)
#     return upper/lower
# now = datetime.now()
#
# result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\"
# now = datetime.now()
# current_time = str(now.strftime("%Y_%m_%d"))
# result_location_step = result_location + 'one_step_2023_03_04_.csv'
#
#
#
#
# data_= pd.read_csv(result_location_step, header=0, encoding='windows-1252')
# data_ = np.array(data_)
# data_real = np.array(data_[:,4]).reshape(-1,1)
# record_range=data_[:, 2: 4]
# pred_range=data_[:, 0: 2]
# record_range=record_range.transpose(1,0)
# pred_range=pred_range.transpose(1,0)
# prediction_ave_ = dtw_barycenter_averaging(pred_range)
# record_ave_ = dtw_barycenter_averaging(record_range)
# prediction_ave_=np.array(prediction_ave_).reshape(-1,1)
# record_ave_=np.array(record_ave_).reshape(-1,1)
# print('rmse, r_2, mape, mae, smape, tic \n')
#
# # plt.plot(record_ave_)
# # plt.plot(prediction_ave_)
# # plt.plot(data_real)
# # plt.show()
#
# rmse_each = rmse(prediction_ave_, data_real)
# r_2_each = r2_score(prediction_ave_, data_real)
# mape_each = mape(prediction_ave_, data_real)
# mae_each = mae(prediction_ave_, data_real)
# smape_each = smape(prediction_ave_, data_real)
# tic_each = tic(prediction_ave_, data_real)
# s = str(rmse_each) + ',' + str(r_2_each) + ',' + str(mape_each)+ ',' + str(mae_each) + ',' + str(smape_each)+ ',' + str(tic_each)
# print (s)
#
# rmse_each = rmse(record_ave_, data_real)
# r_2_each = r2_score(record_ave_, data_real)
# mape_each = mape(record_ave_, data_real)
# mae_each = mae(record_ave_, data_real)
# smape_each = smape(record_ave_, data_real)
# tic_each = tic(record_ave_, data_real)
# s = str(rmse_each) + ',' + str(r_2_each) + ',' + str(mape_each)+ ',' + str(mae_each) + ',' + str(smape_each)+ ',' + str(tic_each)
# print (s)
#
#
#
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
import time
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
def smape(act,forc):
    return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))
def tic (act,forc):
    leng=len(act)
    useless=np.zeros(leng)
    upper= mae(act,forc)
    lower=mae(act,useless)+mae(forc,useless)
    return upper/lower
now = datetime.now()
current_time = str(now.strftime("%Y_%m_%d"))
result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\"
# result_location_hour = result_location + 'one_step_03_21_Mar_03_.csv'  #european_03_21_Mar_03_.csv

result_location_hour = result_location + 'new_approach_UCI_12_15_04_AC_.csv'  #

data_= pd.read_csv(result_location_hour, header=1, encoding='windows-1252')
data_ = np.array(data_)
aa=5001
data_real = data_[0:aa,4]
data_pred_lower=data_[0:aa,2]
data_pred_upper = data_[0:aa,3]
length_real = len(data_real)
data_pred = np.zeros(length_real)
now = datetime.now()

print('rmse, r_2, mape, mae, smape, tic')
for i in range(length_real):
    if(data_pred_upper [i] !=0):
        if((data_pred_upper [i]>data_real[i]) and (data_pred_lower [i]<data_real[i] )) or((data_pred_upper [i]<data_real[i]) and (data_pred_lower [i]>data_real[i] )):
            data_pred[i]=data_real[i]
        else:
            upper_diff =abs(data_pred_upper [i]-data_real[i])
            lower_diff = abs(data_pred_lower[i] - data_real[i])
            if(upper_diff<lower_diff):
                data_pred[i] = data_pred_upper [i]
            else:
                data_pred[i] = data_pred_lower [i]
    else:
        data_pred[i] = data_pred_lower[i]

rmse_each = rmse(data_pred, data_real)
r_2_each = r2_score(data_pred, data_real)
mape_each = mape(data_pred, data_real)
mae_each = mae(data_pred, data_real)
smape_each = smape(data_pred, data_real)
tic_each = tic(data_pred, data_real)
s = str(rmse_each) + ',' + str(r_2_each) + ',' + str(mape_each)+ ',' + str(mae_each) + ',' + str(smape_each)+ ',' + str(tic_each)+ '\n'
print (s)


raw_lower=data_[:,0]
raw_upper = data_[:,1]
raw_pred = np.zeros(length_real)

print('rmse, r_2, mape, mae, smape, tic')
for i in range(length_real):
    if(raw_upper [i] !=0):
        if((raw_upper [i]>data_real[i]) and (raw_lower [i]<data_real[i] )) or((raw_upper [i]<data_real[i]) and (raw_lower [i]>data_real[i] )):
            raw_pred[i]=data_real[i]
        else:
            upper_diff =abs(raw_upper [i]-data_real[i])
            lower_diff = abs(raw_lower[i] - data_real[i])
            if(upper_diff<lower_diff):
                raw_pred[i] = raw_upper [i]
            else:
                raw_pred[i] = raw_lower [i]
    else:
        raw_pred[i] = data_pred_lower[i]
rmse_each = rmse(raw_pred, data_real)
r_2_each = r2_score(raw_pred, data_real)
mape_each = mape(raw_pred, data_real)
mae_each = mae(raw_pred, data_real)
smape_each = smape(raw_pred, data_real)
tic_each = tic(raw_pred, data_real)
s = str(rmse_each) + ',' + str(r_2_each) + ',' + str(mape_each)+ ',' + str(mae_each) + ',' + str(smape_each)+ ',' + str(tic_each)+ '\n'
print (s)


real_max= max(data_real)
real_min= min(data_real)
real_mean=np.average(data_real)

intervals=data_pred_upper- data_pred_lower

pred_interval_max= max(intervals)
pred_interval_min=min(intervals)
pred_interval_mean=np.average(intervals)

print ('real max,', real_max )
print ('real min,', real_min )
print ('real min-max,', real_max-real_min )
print ('real average,', real_mean )

print ('pred max,', pred_interval_max )
print ('pred min,', pred_interval_min )
print ('pred average,', pred_interval_mean )

print ('mean percantage,', pred_interval_mean/(real_mean))






