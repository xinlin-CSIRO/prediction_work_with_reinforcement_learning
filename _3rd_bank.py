#!/usr/local/bin python
from __future__ import division
import numpy as np
import pandas as pd
import math
import io
import time
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import euclidean
from sklearn.ensemble import GradientBoostingRegressor
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import os
import sys
import os.path
from sklearn.svm import SVR
import copy
from sklearn.metrics import accuracy_score, precision_recall_curve,f1_score
from numpy import linalg as li
import random
import pickle
from math import log, ceil, floor
from sklearn.cluster import KMeans
import tensorflow as tf
import warnings
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import random
import sqlite3
from tensorforce import Agent, Environment
import numpy as np
from tensorforce.execution import Runner
import tensorforce
import tensorflow


warnings.filterwarnings("ignore")


scaler = MinMaxScaler(feature_range=(0, 1))
def get_training_data(path):
    ratings_= pd.read_csv(path, header=None)
    dataset_ = ratings_.values
    dataset_ = scaler.fit_transform(dataset_)
    dataset_ = dataset_[:,0:96]
    all_length_ = len(dataset_)
    return ( dataset_, all_length_)

def get_single_data_for_test(path,i):
    ratings_= pd.read_csv(path, header=None)
    dataset_ = ratings_.values
    dataset_ = scaler.fit_transform(dataset_)
    dataset_ = dataset_[i,0:96]
    return ( dataset_)

def distance(w1, w2):
    d = abs(w2 - w1)
    return d
# DTW计算序列s1,s2的最小距离
def DTW_opt(s1, s2):
    m = len(s1)
    n = len(s2)
    # 构建二位dp矩阵,存储对应每个子问题的最小距离
    dp = [[0] * n for _ in range(m)]
    # 起始条件,计算单个字符与一个序列的距离
    for i in range(m):
        dp[i][0] = distance(s1[i], s2[0])
    for j in range(n):
        dp[0][j] = distance(s1[0], s2[j])

    # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
    length_opt=min(m,n)
    weigth_0=1
    diff_=(2-length_opt)/(length_opt*(length_opt-1))
    for i in range(1, m):
        for j in range(1, n):
            #dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + distance(s1[i], s2[j])
            dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])*(1+(j-1)*diff_) + distance(s1[i], s2[j])
            #dp=np.array(dp)
    return dp[-1][-1]

def DTW_old(s1, s2):
    m = len(s1)
    n = len(s2)
    # 构建二位dp矩阵,存储对应每个子问题的最小距离
    dp = [[0] * n for _ in range(m)]
    # 起始条件,计算单个字符与一个序列的距离
    for i in range(m):
        dp[i][0] = distance(s1[i], s2[0])
    for j in range(n):
        dp[0][j] = distance(s1[0], s2[j])

    # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
    length_opt=min(m,n)
    weigth_0=1
    diff_=(2-length_opt)/(length_opt*(length_opt-1))
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + distance(s1[i], s2[j])
    return dp[-1][-1]

def listed_data_selector_opt(for_test, i_tern,dataset,train_window, prediction_window):
    data_capacility=49
    distance = np.ones(data_capacility)*1000000
    train_one_year=[]
    test_one_year=[]
    train_list=[]
    test_list=[]
    one_week = 96 * 7
    for i in range(1, data_capacility):  # one year has 52 weeks
         test_head = int(i_tern - (one_week*i))
         test_rear = int(i_tern- (one_week*i)+ prediction_window)
         train_rear = int(i_tern - (one_week*i) - 1)
         train_head = int(i_tern  -1- (one_week*i)-train_window)
         a = dataset[train_head:train_rear]
         train_one_year.append(a)
         b = dataset[test_head:test_rear]
         test_one_year.append(b)
         distance[i-1] = DTW_opt(a, for_test)
    here_rank = np.argsort(distance)
    for i in range(0, data_capacility-1):  # find the most close 7 series
        # print(i)
        location = here_rank[i]
        candidate_train = train_one_year[location]
        candidate_test = test_one_year[location]
        # plt.plot(candidate_train)
        train_list.append(candidate_train)
        # print (candidate_train)
        test_list.append(candidate_test)
    return (train_list, test_list)

def listed_data_selector_opt_new (for_test, i_tern,dataset,train_window, prediction_window):
    data_capacility=49
    distance = np.ones(data_capacility)*1000000
    train_one_year=[]
    test_one_year=[]
    train_list=[]
    test_list=[]
    one_week = 96 * 7
    for i in range(1, data_capacility):  # one year has 52 weeks
         test_head = int(i_tern - (one_week*i))
         test_rear = int(i_tern- (one_week*i)+ prediction_window)
         train_rear = int(i_tern - (one_week*i) - 1)
         train_head = int(i_tern  -1- (one_week*i)-train_window)
         a = dataset[train_head:train_rear]
         train_one_year.append(a)
         b = dataset[test_head:test_rear]
         test_one_year.append(b)
         distance[i-1] = DTW_opt(a, for_test)
    here_rank = np.argsort(distance)
    ranked_distance=[]
    for i in range(0, data_capacility-1):  # find the most close 7 series
        # print(i)
        location = here_rank[i]
        candidate_train = train_one_year[location]
        candidate_test = test_one_year[location]
        # plt.plot(candidate_train)
        train_list.append(candidate_train)
        # print (candidate_train)
        test_list.append(candidate_test)
        ranked_distance.append(distance[location])
    return (train_list, test_list,ranked_distance)

def training_data_selector_(for_test,i_tern,dataset,size_of_window, n_days):
    distance = np.zeros(52)
    train_one_year=[]
    test_one_year=[]
    train_list=[]
    test_list=[]
    for i in range(1, 53):  # one year has 52 weeks
         test_head = int(i_tern - (672*i))
         test_rear = int(i_tern- (672*i)+ size_of_window)
         train_rear = int(i_tern - (672*i) - 1)
         train_head = int(i_tern  -1- (672*i)-size_of_window)
         a = dataset[train_head:train_rear]
         train_one_year.append(a)
         b = dataset[test_head:test_rear]
         test_one_year.append(b)
         if(len(a)==96):
             distance[i-1], path = fastdtw(a, for_test, dist=euclidean)
         else: distance[i-1]=10000
    here_rank = np.argsort(distance)
    gross_n_days=n_days+1
    for i in range(0, gross_n_days):  # find the most close 7 series
        location = here_rank[i]
        candidate_train = train_one_year[location]
        candidate_test  = test_one_year[location]
        #plt.plot(candidate_train)
        train_list.append(candidate_train)
        # print (candidate_train)
        test_list.append(candidate_test)
    train_list_=list(np.reshape(np.array(train_list),[8, 96]))
    test_data_gross=np.array(test_list)
    exetreme_value=np.zeros([gross_n_days,2])
    exetreme_location = np.zeros([gross_n_days, 2])
    mean_array = np.zeros([gross_n_days])
    for i in range(0, gross_n_days):  # find the most close 7 series
            exetreme_value[i,0]  =     np.min(test_data_gross[i])
            exetreme_value[i, 1] =     np.max(test_data_gross[i])
            exetreme_location[i, 0] = np.argmin(test_data_gross[i])
            exetreme_location[i, 1] = np.argmax(test_data_gross[i])
            mean_array[i]=                 np.mean(test_data_gross[i])
    #extected_data will be the newdataset for classification
    mid_data = np.zeros([gross_n_days, size_of_window * 1])
    extected_data=np.zeros([gross_n_days,size_of_window*1])
    for x in range(0, gross_n_days):  #
        for y in range(0, size_of_window*1):
            if(y!=exetreme_location[x, 0]) and (y!=exetreme_location[x, 1]):
                 mid_data[x,y]=mean_array[x]
            elif (y==exetreme_location[x, 0]) and (y!=exetreme_location[x, 1]):
                mid_data[x, y] = exetreme_value[x,0]
            elif (y!=exetreme_location[x, 0]) and (y==exetreme_location[x, 1]):
                mid_data[x, y] = exetreme_value[x,1]

    for z in range(0, gross_n_days):
              #print (z)
              if((exetreme_location[z, 0]==0) and (exetreme_location[z, 1]==size_of_window)):
                  #print("scenario=1")
                  diff_=(exetreme_value[z,1] -exetreme_value[z,0])/size_of_window
                  for y in range(0, size_of_window * 1):
                      extected_data[z, y] = exetreme_value[z,0]+diff_*y
              elif (exetreme_location[z, 1] == 0) and (exetreme_location[z, 0] == size_of_window):
                          #print("scenario=2")
                          diff_ = (exetreme_value[z, 1] - exetreme_value[z, 0]) / size_of_window
                          for y in range(0, size_of_window * 1):
                              extected_data[z, y] = exetreme_value[z, 1] - diff_*y
             # 3 points
              elif (exetreme_location[z, 1] != 0) and (exetreme_location[z, 0] == size_of_window):
                   diff_ = (exetreme_value[z, 1] - mean_array[z] )/ exetreme_location[z, 1]
                   #print("scenario=3")
                   for y in range(0, int(exetreme_location[z, 1])):
                       extected_data[z, y]=mean_array[z] +diff_*y
                   diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (size_of_window-exetreme_location[z, 1])
                   for y in range(int(exetreme_location[z, 1]), size_of_window):
                       extected_data[z, y] = exetreme_value[z, 1]  - diff_2*(y-int(exetreme_location[z, 1]))
             # 3 points
              elif (exetreme_location[z, 1] != 0) and (exetreme_location[z, 0] == 0):
                  diff_ = (exetreme_value[z, 1] - exetreme_value[z, 0]) / exetreme_location[z, 1]
                  #print("scenario=4")
                  for y in range(0, int(exetreme_location[z, 1])):
                      extected_data[z, y] = exetreme_value[z, 0]+ diff_*y
                  diff_2 = (exetreme_value[z, 1] - mean_array[z]) / (size_of_window - exetreme_location[z, 1])
                  for y in range(int(exetreme_location[z, 1]), size_of_window):
                      extected_data[z, y] = exetreme_value[z, 1] - diff_2*(y-int(exetreme_location[z, 1]))
              # 3 points
              elif (exetreme_location[z, 1] == 0) and (exetreme_location[z, 0] != 0):
                      diff_ = (exetreme_value[z, 1] - exetreme_value[z, 0]) / exetreme_location[z, 0]
                      #print("scenario=5")
                      for y in range(0, int(exetreme_location[z, 0])):
                          extected_data[z, y] = exetreme_value[z, 1] - diff_*y
                      diff_2 = (mean_array[z]-exetreme_value[z, 0] ) / (size_of_window - exetreme_location[z, 0])
                      for y in range(int(exetreme_location[z, 0]), size_of_window):
                          extected_data[z, y] = exetreme_value[z, 0] + diff_2*(y-int(exetreme_location[z, 0]))
             # 3 points
              elif (exetreme_location[z, 1] == size_of_window) and (exetreme_location[z, 0] != 0):
                  #print("scenario=6")
                  diff_ = (mean_array[z] - exetreme_value[z, 0]) / (exetreme_location[z, 0])
                  diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (size_of_window-exetreme_location[z, 0])
                  for y in range(0, int(exetreme_location[z, 0])):
                      extected_data[z, y] = mean_array[z] - diff_*y
                  for y in range(int(exetreme_location[z, 0]), size_of_window):
                      extected_data[z, y] = exetreme_value[z, 0] + diff_2*(y-int(exetreme_location[z, 0]))
              # 4 points
              elif (exetreme_location[z, 1] != size_of_window) and (exetreme_location[z, 0] != 0):

                  if(exetreme_location[z, 1]>exetreme_location[z, 0]):#min shows first
                      #print("scenario=7")
                      diff_ = (mean_array[z] - exetreme_value[z, 0]) / (exetreme_location[z, 0])
                      diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (exetreme_location[z, 1] - exetreme_location[z, 0])
                      diff_3 = (exetreme_value[z, 1] - mean_array[z]) / (size_of_window - exetreme_location[z, 1])
                      for y in range(0, int(exetreme_location[z, 0])):
                          extected_data[z, y] = mean_array[z] - diff_*y
                      for y in range(int(exetreme_location[z, 0]), int(exetreme_location[z, 1])):
                          extected_data[z, y] = exetreme_value[z, 0] + diff_2*(y-int(exetreme_location[z, 0]))
                      for y in range(int(exetreme_location[z, 1]), size_of_window):
                          extected_data[z, y] = exetreme_value[z, 1] - diff_3*(y-int(exetreme_location[z, 1]))
                  else:
                      #print("scenario=8") #the bigger shows first
                      diff_ = ( exetreme_value[z, 1]-mean_array[z] ) / (exetreme_location[z, 1])
                      diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (
                                  -exetreme_location[z, 1] + exetreme_location[z, 0])
                      diff_3 = (-exetreme_value[z, 0] + mean_array[z]) / (size_of_window - exetreme_location[z, 0])
                      for y in range(0, int(exetreme_location[z, 1])):
                          extected_data[z, y] = mean_array[z] + diff_*y
                      for y in range(int(exetreme_location[z, 1]), int(exetreme_location[z, 0])):
                          extected_data[z, y] = exetreme_value[z, 1] - diff_2*(y-int(exetreme_location[z, 1]))
                      for y in range(int(exetreme_location[z, 0]), size_of_window):
                          extected_data[z, y] = exetreme_value[z, 0] + diff_3*(y-int(exetreme_location[z, 0]))
              #plt.plot(extected_data[z,:])
    estimator = KMeans(n_clusters=2)
    estimator.fit(mid_data)
    Scores=silhouette_score(mid_data, estimator.labels_, metric='euclidean')
    if(Scores>0.3):
        which_one_is_majority=1
        one=zero=0
        for i_dex_ in estimator.labels_:
            if(i_dex_==1):
                one +=  1
            else:
                zero=zero+1
        if(one<zero): which_one_is_majority=1
        else: which_one_is_majority=0
        break_index=1000
        for i_dex_ in range(0, n_days):
            if (estimator.labels_[i_dex_]== which_one_is_majority):
                break_index=i_dex_
                break
        if(break_index<=n_days):
            train_np=np.reshape(np.array(train_list),[gross_n_days,size_of_window])
            test_np = np.reshape(np.array(test_list),[gross_n_days,size_of_window])
            train_list_re=np.delete(train_np, break_index,0)
            test_list_re = np.delete(test_np, break_index, 0)
            train_list_reduced=train_list_re.tolist()
            test_list_reduced =test_list_re.tolist()
        else:
            train_list_reduced=[]
            test_list_reduced=[]
            for i in range(0, n_days):  # find the most close 7 series
                candidate_train = train_list[i]
                candidate_test = test_list[i]
                train_list_reduced.append(candidate_train)
                test_list_reduced.append(candidate_test)
    else:
        break_index=10000
        train_list_reduced = []
        test_list_reduced = []
        for i in range(0, n_days):  # find the most close 7 series
            candidate_train = train_list[i]
            candidate_test = test_list[i]
            train_list_reduced.append(candidate_train)
            test_list_reduced.append(candidate_test)
    #plt.show()
    mid_data=mid_data.tolist()
    # return (train_list_reduced, test_list_reduced, train_list,test_list, mid_data, estimator.labels_, break_index, Scores)
    return (train_list_reduced, test_list_reduced, train_list,test_list, mid_data, estimator.labels_, break_index, Scores)

def training_data_selector_2(for_test,i_tern,dataset,train_window, window_prediction,n_days):
    distance = np.zeros(52)
    train_one_year=[]
    test_one_year=[]
    train_list=[]
    test_list=[]
    one_week=24*7
    for i in range(1, 53):  # one year has 52 weeks
         test_head = int(i_tern - (one_week*i))
         test_rear = int(i_tern- (one_week*i)+ window_prediction)
         train_rear = int(i_tern - (one_week*i) - 1)
         train_head = int(i_tern  -1- (one_week*i)-train_window)
         a = dataset[train_head:train_rear]
         train_one_year.append(a)
         b = dataset[test_head:test_rear]
         test_one_year.append(b)
         if(len(a)==train_window):
             distance[i-1], path = fastdtw(a, for_test, dist=euclidean)
         else: distance[i-1]=10000
    here_rank = np.argsort(distance)
    gross_n_days=n_days+1
    for i in range(0, gross_n_days):  # find the most close 7 series
        location = here_rank[i]
        candidate_train = train_one_year[location]
        candidate_test  = test_one_year[location]
        # plt.plot(candidate_train)
        train_list.append(candidate_train)
        # print (candidate_train)
        test_list.append(candidate_test)
    # train_list_=list(np.reshape(np.array(train_list),[8, 96]))
    test_data_gross=np.array(test_list)
    exetreme_value=np.zeros([gross_n_days,2])
    exetreme_location = np.zeros([gross_n_days, 2])
    mean_array = np.zeros([gross_n_days])
    for i in range(0, gross_n_days):  # find the most close 7 series
            exetreme_value[i,0]  =     np.min(test_data_gross[i])
            exetreme_value[i, 1] =     np.max(test_data_gross[i])
            exetreme_location[i, 0] = np.argmin(test_data_gross[i])
            exetreme_location[i, 1] = np.argmax(test_data_gross[i])
            mean_array[i]=                 np.mean(test_data_gross[i])
    #extected_data will be the newdataset for classification
    mid_data = np.zeros([gross_n_days, train_window * 1])
    extected_data=np.zeros([gross_n_days,train_window*1])
    for x in range(0, gross_n_days):  #
        for y in range(0, train_window*1):
            if(y!=exetreme_location[x, 0]) and (y!=exetreme_location[x, 1]):
                 mid_data[x,y]=mean_array[x]
            elif (y==exetreme_location[x, 0]) and (y!=exetreme_location[x, 1]):
                mid_data[x, y] = exetreme_value[x,0]
            elif (y!=exetreme_location[x, 0]) and (y==exetreme_location[x, 1]):
                mid_data[x, y] = exetreme_value[x,1]

    for z in range(0, gross_n_days):
              #print (z)
              if((exetreme_location[z, 0]==0) and (exetreme_location[z, 1]==train_window)):
                  #print("scenario=1")
                  diff_=(exetreme_value[z,1] -exetreme_value[z,0])/train_window
                  for y in range(0, train_window * 1):
                      extected_data[z, y] = exetreme_value[z,0]+diff_*y
              elif (exetreme_location[z, 1] == 0) and (exetreme_location[z, 0] == train_window):
                          #print("scenario=2")
                          diff_ = (exetreme_value[z, 1] - exetreme_value[z, 0]) / train_window
                          for y in range(0, train_window * 1):
                              extected_data[z, y] = exetreme_value[z, 1] - diff_*y
             # 3 points
              elif (exetreme_location[z, 1] != 0) and (exetreme_location[z, 0] == train_window):
                   diff_ = (exetreme_value[z, 1] - mean_array[z] )/ exetreme_location[z, 1]
                   #print("scenario=3")
                   for y in range(0, int(exetreme_location[z, 1])):
                       extected_data[z, y]=mean_array[z] +diff_*y
                   diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (train_window-exetreme_location[z, 1])
                   for y in range(int(exetreme_location[z, 1]), train_window):
                       extected_data[z, y] = exetreme_value[z, 1]  - diff_2*(y-int(exetreme_location[z, 1]))
             # 3 points
              elif (exetreme_location[z, 1] != 0) and (exetreme_location[z, 0] == 0):
                  diff_ = (exetreme_value[z, 1] - exetreme_value[z, 0]) / exetreme_location[z, 1]
                  #print("scenario=4")
                  for y in range(0, int(exetreme_location[z, 1])):
                      extected_data[z, y] = exetreme_value[z, 0]+ diff_*y
                  diff_2 = (exetreme_value[z, 1] - mean_array[z]) / (train_window - exetreme_location[z, 1])
                  for y in range(int(exetreme_location[z, 1]), train_window):
                      extected_data[z, y] = exetreme_value[z, 1] - diff_2*(y-int(exetreme_location[z, 1]))
              # 3 points
              elif (exetreme_location[z, 1] == 0) and (exetreme_location[z, 0] != 0):
                      diff_ = (exetreme_value[z, 1] - exetreme_value[z, 0]) / exetreme_location[z, 0]
                      #print("scenario=5")
                      for y in range(0, int(exetreme_location[z, 0])):
                          extected_data[z, y] = exetreme_value[z, 1] - diff_*y
                      diff_2 = (mean_array[z]-exetreme_value[z, 0] ) / (train_window - exetreme_location[z, 0])
                      for y in range(int(exetreme_location[z, 0]), train_window):
                          extected_data[z, y] = exetreme_value[z, 0] + diff_2*(y-int(exetreme_location[z, 0]))
             # 3 points
              elif (exetreme_location[z, 1] == train_window) and (exetreme_location[z, 0] != 0):
                  #print("scenario=6")
                  diff_ = (mean_array[z] - exetreme_value[z, 0]) / (exetreme_location[z, 0])
                  diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (train_window-exetreme_location[z, 0])
                  for y in range(0, int(exetreme_location[z, 0])):
                      extected_data[z, y] = mean_array[z] - diff_*y
                  for y in range(int(exetreme_location[z, 0]), train_window):
                      extected_data[z, y] = exetreme_value[z, 0] + diff_2*(y-int(exetreme_location[z, 0]))
              # 4 points
              elif (exetreme_location[z, 1] != train_window) and (exetreme_location[z, 0] != 0):

                  if(exetreme_location[z, 1]>exetreme_location[z, 0]):#min shows first
                      #print("scenario=7")
                      diff_ = (mean_array[z] - exetreme_value[z, 0]) / (exetreme_location[z, 0])
                      diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (exetreme_location[z, 1] - exetreme_location[z, 0])
                      diff_3 = (exetreme_value[z, 1] - mean_array[z]) / (train_window - exetreme_location[z, 1])
                      for y in range(0, int(exetreme_location[z, 0])):
                          extected_data[z, y] = mean_array[z] - diff_*y
                      for y in range(int(exetreme_location[z, 0]), int(exetreme_location[z, 1])):
                          extected_data[z, y] = exetreme_value[z, 0] + diff_2*(y-int(exetreme_location[z, 0]))
                      for y in range(int(exetreme_location[z, 1]), train_window):
                          extected_data[z, y] = exetreme_value[z, 1] - diff_3*(y-int(exetreme_location[z, 1]))
                  else:
                      #print("scenario=8") #the bigger shows first
                      diff_ = ( exetreme_value[z, 1]-mean_array[z] ) / (exetreme_location[z, 1])
                      diff_2 = (exetreme_value[z, 1] - exetreme_value[z, 0]) / (
                                  -exetreme_location[z, 1] + exetreme_location[z, 0])
                      diff_3 = (-exetreme_value[z, 0] + mean_array[z]) / (train_window - exetreme_location[z, 0])
                      for y in range(0, int(exetreme_location[z, 1])):
                          extected_data[z, y] = mean_array[z] + diff_*y
                      for y in range(int(exetreme_location[z, 1]), int(exetreme_location[z, 0])):
                          extected_data[z, y] = exetreme_value[z, 1] - diff_2*(y-int(exetreme_location[z, 1]))
                      for y in range(int(exetreme_location[z, 0]), train_window):
                          extected_data[z, y] = exetreme_value[z, 0] + diff_3*(y-int(exetreme_location[z, 0]))
              #plt.plot(extected_data[z,:])
    estimator = KMeans(n_clusters=2)
    estimator.fit(mid_data)
    Scores=silhouette_score(mid_data, estimator.labels_, metric='euclidean')
    if(Scores>0.3):
        which_one_is_majority=1
        one=zero=0
        for i_dex_ in estimator.labels_:
            if(i_dex_==1):
                one +=  1
            else:
                zero=zero+1
        if(one<zero): which_one_is_majority=1
        else: which_one_is_majority=0
        break_index=1000
        for i_dex_ in range(0, n_days):
            if (estimator.labels_[i_dex_]== which_one_is_majority):
                break_index=i_dex_
                break
        if(break_index<=n_days):
            train_np=np.reshape(np.array(train_list),[gross_n_days,train_window])
            test_np = np.reshape(np.array(test_list),[gross_n_days,window_prediction])
            train_list_re=np.delete(train_np, break_index,0)
            test_list_re = np.delete(test_np, break_index, 0)
            train_list_reduced=train_list_re.tolist()
            test_list_reduced =test_list_re.tolist()
        else:
            train_list_reduced=[]
            test_list_reduced=[]
            for i in range(0, n_days):  # find the most close 7 series
                candidate_train = train_list[i]
                candidate_test = test_list[i]
                train_list_reduced.append(candidate_train)
                test_list_reduced.append(candidate_test)
    else:
        break_index=10000
        train_list_reduced = []
        test_list_reduced = []
        for i in range(0, n_days):  # find the most close 7 series
            candidate_train = train_list[i]
            candidate_test = test_list[i]
            train_list_reduced.append(candidate_train)
            test_list_reduced.append(candidate_test)
    # for_see=np.reshape(train_list_reduced, (24, n_days))
    # plt.plot(for_see)
    # plt.show()
    mid_data=mid_data.tolist()
    # return (train_list_reduced, test_list_reduced, train_list,test_list, mid_data, estimator.labels_, break_index, Scores)
    return (train_list_reduced, test_list_reduced, train_list,test_list, mid_data, estimator.labels_, break_index, Scores)


def moving_window(i_tern,dataset,size_of_window, n_days):
    train_list = []
    test_list = []
    one_week=24*7
    for i in range(1, (n_days+1)):  # one year has 52 weeks
        test_head = int(i_tern - (one_week * i))
        test_rear = int(i_tern - (one_week * i) + 1)
        train_rear = int(i_tern - (one_week * i))
        train_head = int(i_tern - (one_week * i) - size_of_window)
        a = dataset[train_head:train_rear]
        train_list.append(a)
        b = dataset[test_head:test_rear]
        test_list.append(b)
    return ( train_list,test_list)

def initial_training_data_selector_(for_test, i_tern, dataset, size_of_window, n_days):
    distance = np.zeros(52)
    train_one_year = []
    test_one_year = []
    train_list = []
    test_list = []
    for i in range(1, 53):  # one year has 52 weeks
        test_head = int(i_tern - (672 * i))
        test_rear = int(i_tern - (672 * i) + size_of_window)
        train_rear = int(i_tern - (672 * i) - 1)
        train_head = int(i_tern - 1 - (672 * i) - size_of_window)
        a = dataset[train_head:train_rear]
        train_one_year.append(a)
        b = dataset[test_head:test_rear]
        test_one_year.append(b)
        distance[i - 1], path = fastdtw(a, for_test, dist=euclidean)
    here_rank = np.argsort(distance)
    one_year_ = 52
    # dif_ = np.zeros((n_days, (size_of_window- 1)))
    for i in range(0, n_days):  # find the most close 7 series
        location = here_rank[i]
        candidate_train = train_one_year[location]
        candidate_test = test_one_year[location]
        # plt.plot(candidate_train)
        train_list.append(candidate_train)
        test_list.append(candidate_test)
        '''
        estimator = KMeans(n_clusters=2)
    estimator.fit(dif_)
    clustering=estimator.labels_
    Scores=silhouette_score(dif_, estimator.labels_, metric='euclidean')
    maj,mat=0,0
    for j in range(0, n_days+1):
        if(clustering[j]==0):
            maj=maj+1
        else:
            mat=mat+1
    index=0
    if((mat==1) and (Scores>0.4)) :
        for j in range(0, n_days + 1):
            if(clustering[j]==1):
                index=j
    elif((maj==1) and (Scores>0.4)):
        for j in range(0, n_days + 1):
            if(clustering[j]==0):
                index=j
    else:
        index=10000
    print("index= ",index)
    #print ("len(train_list)= ",len(train_list))
    if(index<=n_days):
        train_list=np.array(train_list)
        train_list=np.reshape(train_list,((n_days+1),size_of_window))
        train_list=np.delete(train_list, index, axis=0)
        test_list = np.array(test_list)
        test_list = np.reshape(test_list, ((n_days + 1), size_of_window))
        test_list = np.delete(test_list, index, axis=0)
    else:
        train_list = np.array(train_list)
        train_list = np.reshape(train_list, ((n_days + 1), size_of_window))
        train_list = np.delete(train_list, n_days, axis=0)

        test_list = np.array(test_list)
        test_list = np.reshape(test_list, ((n_days + 1), size_of_window))
        test_list = np.delete(test_list, n_days, axis=0)
    train_list=list(train_list)
    test_list=list(test_list)

        '''

    return (train_list, test_list)

def diversity_dtw (X_set,for_test):
    Bary_dtw_ave_ = dtw_barycenter_averaging(X_set)
    X_set=np.array(X_set)
    X_set=np.reshape(X_set,(X_set.shape[0],X_set.shape[1]))
    variances=np.zeros(X_set.shape[1])
    for lie in range(0, X_set.shape[1]):
         for hang in range (0, X_set.shape[0]):
             dis_ =(X_set[hang,lie]-Bary_dtw_ave_[lie])**2
             dis_+=dis_
         aa=dis_/(X_set.shape[0]+1)
         variances[lie]=aa**0.5
    tolerance_X_set=np.zeros([2, X_set.shape[1]])
    for lie in range(0, X_set.shape[1]):
        tolerance_X_set [0, lie]  = Bary_dtw_ave_[lie]+ 3*variances[lie]
        tolerance_X_set [1, lie]  = Bary_dtw_ave_[lie] - 3*variances[lie]
    security = 0
    current_load=for_test [len(for_test)-1]
    for lie in range(0, X_set.shape[1]):
        if(( current_load< tolerance_X_set [0,lie]) and (current_load > tolerance_X_set [1,lie])):
             security += 1
    result=0
    if(security==0):
        result=1
    return (result)

def previous_data_generator_(for_test,i_tern,dataset,size_of_window,n_days ):
    distance = np.zeros(52)
    train_one_year = []
    train_one_year_use=[]
    train_list = []
    for i in range(1, 53):  # one year has 52 weeks
        train_rear = int(i_tern - (672 * i) + 1 * (size_of_window))
        train_head = int(i_tern - (672 * i) - 1 * (size_of_window))
        a = dataset[train_head:train_rear]
        #plt.plot(a)
        train_rear_use = int(i_tern - (672 * i) + 2 * (size_of_window))
        train_head_use = int(i_tern - (672 * i) - 2 * (size_of_window))
        b = dataset[train_head_use:train_rear_use]
        # plt.plot(a)
        train_one_year_use.append(b)
        distance[i - 1], path = fastdtw(a, for_test, dist=euclidean)
    #plt.show()
    here_rank = np.argsort(distance)
    for i in range(0, n_days):  # find the most close 7 series
        location = here_rank[i]
        candidate_train = train_one_year_use[location]
        #plt.plot(candidate_train)
        train_list.append(candidate_train)
    #plt.show()
    return (train_list)


def ONE_STEP_AHEAD_predictor (for_test, trainX, trainY,train_window):
    len_=train_window
    # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    # trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))
    alpha = 0.7
    trainX_2D = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
    trainY_2D  = np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
    trainY_2D = (trainY_2D[:, 0]).reshape(-1, 1)
    model_ML = GradientBoostingRegressor(loss='quantile',  n_estimators=5, alpha=alpha)
    model_ML.fit(trainX_2D, trainY_2D)
    # make a one-step prediction
    for_test_1D = np.reshape(for_test, (1, len_))
    y_upper = model_ML.predict(for_test_1D)
    #################################
    model_ML.set_params(alpha=1.0 - alpha)
    model_ML.fit(trainX_2D, trainY_2D)
    y_lower = model_ML.predict(for_test_1D)
    if(y_upper>y_lower):
        o_upper=y_upper
        o_lower=y_lower
    else:
        o_upper = y_lower
        o_lower = y_upper
    return (o_lower, o_upper)


def ONE_STEP_moving_window (for_test, trainX, trainY,_x_,initial_alpha):
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))
    #print('_x_ ', _x_)
    #alpha = (1)/(1+math.exp(-_x_))
    alpha =  1 / (1 + (math.exp(-_x_)) ** 0.01)
    #print('alpha_before= ',alpha)
    alpha=max(alpha, 0.1)
    alpha=min (alpha, 0.7)
    #print('alpha_after= ', alpha)
##################################################
    trainX_2D = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
    trainY_2D  = np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
    trainY_2D = (trainY_2D[:, 0]).reshape(-1, 1)
    model_ML = GradientBoostingRegressor(loss='quantile',  n_estimators=5, alpha=alpha)
    model_ML.fit(trainX_2D, trainY_2D)
    # make a one-step prediction
    for_test_1D = np.reshape(for_test, (1, 96*7))
    y_upper = model_ML.predict(for_test_1D)
    #################################
    model_ML.set_params(alpha=1.0 - alpha)
    model_ML.fit(trainX_2D, trainY_2D)
    # Make the prediction on the meshed x-axis
    y_lower = model_ML.predict(for_test_1D)
    if(y_upper>y_lower):
        o_upper=y_upper
        o_lower=y_lower
    else:
        o_upper = y_lower
        o_lower = y_upper
    return (o_lower, o_upper,alpha)


def previous_prediction_result_check (lower,upper, real):
    last_setp_prediction_acceptable=0
    tolence=0
    upper_tolerance=upper*(1+tolence)
    lower_tolerance = lower * (1-tolence)
    if ((real < upper) and (real > lower)) or ((real<upper_tolerance) and (real > lower)) or ((real < upper) and (real>lower_tolerance)):
        last_setp_prediction_acceptable = 1
    return( last_setp_prediction_acceptable)

def denoising_(X_set,for_test):
    Bary_dtw_ave_ = dtw_barycenter_averaging(X_set)
    #plt.plot(Bary_dtw_ave_,'y-')
    #plt.show()
    X_set=np.array(X_set)
    X_set=np.reshape(X_set,(X_set.shape[0],X_set.shape[1]))
    variances=np.zeros(X_set.shape[1])
    for lie in range(0, X_set.shape[1]):
         for hang in range (0, X_set.shape[0]):
             dis_ =(X_set[hang,lie]-Bary_dtw_ave_[lie])**2
             dis_+=dis_
         aa=dis_/(X_set.shape[0]+1)
         variances[lie]=aa**0.5
    #plt.plot(variances)
    #plt.show()
    tolerance_X_set=np.zeros([2, X_set.shape[1]])
    for lie in range(0, X_set.shape[1]):
        tolerance_X_set [0, lie]  = Bary_dtw_ave_[lie]+ 3*variances[lie]
        tolerance_X_set [1, lie]  = Bary_dtw_ave_[lie] - 3*variances[lie]
    security = 0
    x = np.arange(X_set.shape[1])
    #plt.fill_between(x, tolerance_X_set [0,:], tolerance_X_set [1, :], fc='r', label='3*variance')
    #plt.plot(for_test)
    #plt.show()
    current_load=for_test [len(for_test)-1]
    for lie in range(0, X_set.shape[1]):
        if(( current_load< tolerance_X_set [0,lie]) and (current_load > tolerance_X_set [1,lie])):
             security += 1
    result=0
    if(security==0):
        result=1
    return (result)

def RL_BASED_TRAINING_DATA_SELECTOR (trainX_listed, trainY_listed, env_prediction, agent,new_states):

    actions = agent.act(states=new_states)
    print('action is %f',actions)
    states, terminal, reward,case = env_prediction.execute(actions=actions)
    agent.observe(terminal=terminal, reward=reward)
    steps=1*(actions+1)
    print('rewards ',reward)
    trainX_final=[]
    trainY_final=[]
    for x in range (0, 7):
        x=steps*x
        trainX_final.append(trainX_listed[x])
        trainY_final.append(trainY_listed[x])
    return (trainX_final,trainY_final, actions, reward,case )



class prediction_environment (Environment):
    def __init__(self,new_states):
        super().__init__()
        self.path=new_states


    def states(self):
        return dict(type='float', shape=(4,))

    def actions(self):
        return dict(type='int', num_values=7)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(4,))
        return state


    def execute(self, actions):
        ratings = pd.read_csv(self.path, header=0)

        state_=np.array(ratings)[-1]

        # state_=  0 #observation_states
        terminal = False  # Always False if no "natural" terminal state
        reward, case = reward_function (state_)
        return state_, terminal, reward,case

class prediction_environment2 (Environment):
    def __init__(self,new_states):
        super().__init__()
        self.path=new_states


    def states(self):
        return dict(type='float', shape=(4,))

    def actions(self):
        return dict(type='float', shape=(7,), min_value=1, max_value=47)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        state = np.random.random(size=(4,))
        return state


    def execute(self, actions):
        ratings = pd.read_csv(self.path, header=0)

        state_=np.array(ratings)[-1]

        # state_=  0 #observation_states
        terminal = False  # Always False if no "natural" terminal state
        reward, case = reward_function_new (state_)
        return state_, terminal, reward,case
def reward_function(state):
    low=state[0]
    high=state[1]
    real=state[2]
    last_real = state[3]

    dynamic_chang_load=abs(last_real-real)/last_real

    interval = max(abs(high - low),1)
    dynmiac_change_interval=interval/real
    mid = (low+high)/2
    rewards=0
    case=0
    feasible_index=0.15
    load_changing_index=0.04
    if(low <= real <=high):
        if(dynmiac_change_interval< feasible_index): #accurate + flecible
            a= 10* abs(mid - real)/real
            rewards = a
            case = 1
        else: #accurate + inflecible
            a = 1 * abs(mid - real)/real
            rewards = a
            case = 2
    else:
        if   (dynamic_chang_load >= load_changing_index) and (dynmiac_change_interval >= feasible_index):# inevitable--> change is so big and the interval has already been big enough
            rewards = 0
            case = 3
        elif (dynamic_chang_load > load_changing_index) and (dynmiac_change_interval < feasible_index):  #  evitable--> change is so big  but interval has not been big enough
            a = abs(mid - real)/real
            rewards = -a
            case = 4
        elif (dynamic_chang_load < load_changing_index) and (dynmiac_change_interval > feasible_index):  #  evitable--> change isnot so big but interval has been big enough
            a = 10* abs(mid - real) / real
            rewards = -a
            case = 5
        elif (dynamic_chang_load < load_changing_index) and (dynmiac_change_interval < feasible_index):  #  evitable--> change isnot so big but interval also has not been big enough
            a = abs(mid - real) / real
            rewards = -a
            case = 6

    return (rewards,case)

def reward_function_new(state):
    low=state[0]
    high=state[1]
    real=state[2]
    last_real = state[3]

    dynamic_chang_load=abs(last_real-real)/last_real

    interval = max(abs(high - low),1)
    dynmiac_change_interval=interval/real
    mid = (low+high)/2
    rewards=0
    case=0
    feasible_index=0.10
    load_changing_index=0.04
    if(low <= real <=high):
        if(dynmiac_change_interval< feasible_index): #accurate + flecible
            a= 10* abs(mid - real)/real
            rewards = a
            case = 1
        else: #accurate + inflecible
            a = 1 * abs(mid - real)/real
            rewards = a
            case = 2
    else:
        if   (dynamic_chang_load >= load_changing_index) and (dynmiac_change_interval >= feasible_index):# inevitable--> change is so big and the interval has already been big enough
            rewards = 0
            case = 3
        elif (dynamic_chang_load > load_changing_index) and (dynmiac_change_interval < feasible_index):  #  evitable--> change is so big  but interval has not been big enough
            a = abs(mid - real)/real
            rewards = -a
            case = 4
        elif (dynamic_chang_load < load_changing_index) and (dynmiac_change_interval > feasible_index):  #  evitable--> change is not big but interval has been big enough
            a = 10* abs(mid - real) / real
            rewards = -a
            case = 5
        elif (dynamic_chang_load < load_changing_index) and (dynmiac_change_interval < feasible_index):  #  evitable--> change isnot so big but interval also has not been big enough
            a = abs(mid - real) / real
            rewards = -a
            case = 6

    return (rewards,case)

def function(x):
    result = -((2) / (1 + math.exp(-x / 100))) + 2
    return (result)

def min_function(x):
    result = -((2) / (1 + math.exp(-x / 10))) + 1
    return (result)

def Evaluation_3 (real_data_current, real_data_last, real_data_last_X_2, lower_p, upper_p, lower, upper, mix):
    analysis_back_steps=96
    feedback=0
    current_load_diff=real_data_current-real_data_last
    previous_load_diff=real_data_last-real_data_last_X_2
    connection = sqlite3.connect('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\3rd_predictor\\db\\mydb.db')
    cur = connection.cursor()
    diff =real_data_current-(upper_p+lower_p)/2
    if(mix==0):
        cur.execute("DROP TABLE IF EXISTS progress_3")
        print("Table dropped... ")
        sql_create_projects_table = "CREATE TABLE progress_3 ( id integer PRIMARY KEY,  p_load_1 integer, previous_diff  integer, up integer, low integer, difference integer); "
        cur.execute(sql_create_projects_table)
        connection.commit()
        print("Table generated... ")
        # login the new data
        sqlite_insert_ = """INSERT OR IGNORE INTO progress_3
                                              (id,  p_load_1,  previous_diff, up, low, difference)
                                              VALUES (?, ?, ?, ?, ?, ?);"""
        data_tuple = (mix, real_data_last, previous_load_diff, upper_p, lower_p, diff) #(t-1, t, t)--> diff at t
        cur.execute(sqlite_insert_, data_tuple)
        connection.commit()
    else:
        # login the new data
        sqlite_insert_ = """INSERT OR IGNORE INTO progress_3
                                                      (id,  p_load_1,  previous_diff, up, low, difference)
                                                      VALUES (?, ?, ?,?, ?, ?);"""
        data_tuple = (mix, real_data_last, previous_load_diff, upper_p[0], lower_p[0], diff[0])  # (t-1, t, t)--> diff at t
        cur.execute(sqlite_insert_, data_tuple)
        connection.commit()
        if (mix > analysis_back_steps):
            #now begin to query
            sqlite_query__ = """select * from  progress_3 where id<=(?) and id> (?)"""
            cur.execute(sqlite_query__, (mix, (mix - analysis_back_steps-1)))
            records = cur.fetchmany(analysis_back_steps)
            records = np.array(records)
            trainX=records[:,1:5]
            trainY=records[:,5]
            samples=trainX.shape[0]
            features=trainX.shape[1]
            regressor = SVR(kernel = 'linear')
            regressor.fit(trainX, trainY)
            #current_ = np.array([real_data_current, current_load_diff])
            current_ = np.array([real_data_current, current_load_diff, upper, lower])
            current_ = np.reshape(current_, (features, 1))
            feedback = regressor.predict(current_.reshape(1, -1))# t, t+1, t+1 --> predict diff for t+1

    return (feedback)



def RL_BASED_TRAINING_DATA_SELECTOR_new (trainX_listed, trainY_listed, env_prediction, agent,new_states):

    actions = agent.act(states=new_states)
    print('action is %f',actions)
    states, terminal, reward,case = env_prediction.execute(actions=actions)
    agent.observe(terminal=terminal, reward=reward)

    print('rewards ',reward)
    trainX_final=[]
    trainY_final=[]
    actions_=''
    for x in range (0, 7):
        location=round(actions[x])

        trainX_final.append(trainX_listed[location])
        trainY_final.append(trainY_listed[location])
        if(x<6):
            actions_ += str(actions[x]) + ','
        else:
            actions_ += str(actions[x])
    return (trainX_final,trainY_final, actions_, reward,case )