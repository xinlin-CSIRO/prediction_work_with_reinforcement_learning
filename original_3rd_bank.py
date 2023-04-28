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
import struct
import copy
from numpy import linalg as li
from sklearn.cluster import KMeans
import tensorflow as tf
import warnings
import xgboost as xgb
from xgboost import XGBRegressor

import random

import sqlite3


warnings.filterwarnings("ignore")
levelHV_bank=[]
training_key_bank=[]
baseVal = -1
n_times=100
n_layers=50
D = 10000#10K
tolerance_threshold=4.5
threshold_a_day = 0.8
#number of level hypervectors

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




######################HDC##################################################
def diversity_(X_set,for_test):
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

######################HDC##################################################

def HDC (near_,far_,for_test, nLevels):
    Bary_dtw_near_ = dtw_barycenter_averaging(near_, max_iter=len(near_))
    Bary_dtw_far_ = dtw_barycenter_averaging(far_, max_iter=len(far_))
    data_set=[]
    data_set.append(Bary_dtw_near_)
    data_set.append(Bary_dtw_far_)
    data_set.append(for_test)
    boundarys = predict_getlevelList(data_set, nLevels)
    dict_levelHVs = predictor_genLevelHVs(nLevels, D)
    hypervector_near= predictor_EncodeToHV(Bary_dtw_near_, D, dict_levelHVs, boundarys)
    hypervector_far = predictor_EncodeToHV(Bary_dtw_far_, D, dict_levelHVs, boundarys)
    hypervector_test = predictor_EncodeToHV(for_test, D, dict_levelHVs, boundarys)
    similar_test=inner_product(hypervector_far, hypervector_test)
    similar_data = inner_product(hypervector_far, hypervector_near)
    return (similar_test, similar_data)

def random_vector_selector (hypervectors):
    num=len(hypervectors)
    the_one=random.randrange(0, num, 1)
    return(the_one)

def Dispersion_(similar_differences):
    a=0
    for i in range (0, len(similar_differences)):
        a+=similar_differences[i]
    mean=a/len(similar_differences)
    #print ('mean',mean)
    s_i = 0
    for i in range(0, len(similar_differences)):
        s_i+=(similar_differences[i]-mean)**2
    result=s_i/len(similar_differences)
    #print('result ', result)
    #print('len(similar_differences) ',len(similar_differences))
    return (result)

def inner_product(x, y):
    return np.dot(x, y) / (li.norm(x) * li.norm(y) + 0.0)

def predict_getlevelList(buffers, totalLevel):
    minimum = buffers[0][0]
    maximum = buffers[0][0]
    levelList = []
    for buffer in buffers:
        localMin = min(buffer)
        localMax = max(buffer)
        if (localMin < minimum):
            minimum = localMin
        if (localMax > maximum):
            maximum = localMax
    length = maximum - minimum
    gap = length / totalLevel
    for lv in range(totalLevel):
        levelList.append(minimum + lv * gap)
    levelList.append(maximum)
    return levelList

def predictor_genLevelHVs(totalLevel, D):
    # print ('generating level HVs')
    levelHVs = dict()
    indexVector = range(D)
    nextLevel = int((D / 2 / totalLevel))
    change = int(D / 2)
    for level in range(totalLevel):
        name = level
        if (level == 0):
            base = np.full(D, baseVal)  # 10.000di--> [-1,-1,...,-1]
            toOne = np.random.permutation(indexVector)[:change] #随机产生 50个数， 变化范围都是 0-100
        else:
            toOne = np.random.permutation(indexVector)[:nextLevel] #随机产生12个数
        for index in toOne:
            base[index] = base[index] * -1
        levelHVs[name] = copy.deepcopy(base)
    return levelHVs

def predictor_EncodeToHV(inputBuffer, D, levelHVs, levelList):
    sumHV = np.zeros(D, dtype=np.int)
    for keyVal in range(len(inputBuffer)):
        key = numToKey(inputBuffer[keyVal], levelList)
        levelHV = levelHVs[key]
        sumHV = sumHV + np.roll(levelHV, keyVal)
    #print (sumHV)
    return sumHV

def numToKey(value, levelList):
    upperIndex = len(levelList) - 1
    lowerIndex = 0
    keyIndex = 0
    levelList=np.array(levelList).reshape(-1,1)
    if (upperIndex > lowerIndex):
        keyIndex = int((upperIndex + lowerIndex) / 2)
        a= levelList[keyIndex]
        b=levelList[(keyIndex + 1)]
        if (levelList[keyIndex] <= value and levelList[(keyIndex + 1)] > value):
            return keyIndex
        if (levelList[keyIndex, 0] > value):
            upperIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex) / 2)
        else:
            lowerIndex = keyIndex
            keyIndex = int((upperIndex + lowerIndex) / 2)
    return keyIndex

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
        test_list.append(candidate_test)
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
                one=one+1
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
    return (train_list_reduced, test_list_reduced, train_list,test_list, mid_data, estimator.labels_, break_index, Scores)
    # return (train_list_reduced, test_list_reduced, train_list,test_list, mid_data, estimator.labels_, break_index, Scores)


def moving_window(i_tern,dataset,size_of_window, n_):
    train_list = []
    test_list = []
    for i in range(1, (n_+1)):  # one year has 52 weeks
        test_head = int(i_tern - (672 * i))
        test_rear = int(i_tern - (672 * i) + 1)
        train_rear = int(i_tern - (672 * i))
        train_head = int(i_tern - (672 * i) - size_of_window)
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


def ONE_STEP_AHEAD_predictor (for_test, trainX, trainY,_x_,initial_alpha):
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
    for_test_1D = np.reshape(for_test, (1, 96))
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


def previous_prediction_compensation(lower, upper,real,max,min):
        dist=upper-lower
        case = 0
        mean = (lower + upper) / 2
        error = mean - real
        lower_2 = lower - error
        upper_2 = upper - error
        if(dist>10):
            if(real<=upper) and (real>=lower):
                case=1 # nothing to do
            elif (real< max) and (real> min) and (upper_2< max)  and (lower_2 > min):
                case=2
            elif  (real< max) and (real> min) and (upper_2> max)  and (lower_2 > min):
                case=4 # change higher
            elif (real< max) and (real> min) and (upper_2< max)  and (lower_2 < min):
                case=3 #change lower
            elif (real> max)  :
                    case = 5  # change higher
            elif  (real < min):
                    case = 6
            else:
                case = 5  # change higher
                print('neglected')
        else:
            mean = (lower + upper) / 2
            error = mean - real
            case = 2
        print('Current pattern= ', case)
        return (case, error)

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