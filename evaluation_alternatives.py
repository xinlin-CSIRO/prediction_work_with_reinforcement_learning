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

result_location="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\"
now = datetime.now()
result_location_step = result_location + 'Bi_LSTM_att_UCSD__49days.csv'




data_= pd.read_csv(result_location_step, header=0, encoding='windows-1252')
data_ = np.array(data_)
data_real = np.array(data_[:,1]).reshape(-1,1)
pred_=data_[:, 0].reshape(-1,1)

print('rmse, r_2, mape, mae, smape, tic \n')

# plt.plot(record_ave_)
# plt.plot(prediction_ave_)
# plt.plot(data_real)
# plt.show()

rmse_each = rmse(pred_, data_real)
r_2_each = r2_score(pred_, data_real)
mape_each = mape(pred_, data_real)
mae_each = mae(pred_, data_real)
smape_each = smape(pred_, data_real)
tic_each = tic(pred_, data_real)
s = str(rmse_each) + ',' + str(r_2_each) + ',' + str(mape_each)+ ',' + str(mae_each) + ',' + str(smape_each)+ ',' + str(tic_each)
print (s)







