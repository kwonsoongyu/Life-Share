import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, losses, callbacks
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import Callback
from collections import deque
import matplotlib.pyplot as plt
import math
import random
import time
import os

seed = 33
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

data = pd.read_csv('D:/공유경제/공유경제.csv',encoding = 'utf-8', thousands = ',')
data = data.replace(' - ','0')
data = data.astype({'매출건수(건)':'float'})
data = data.astype({'매출금액(천원)':'float'})
print(data.info())
print(data.head())

categories = ['1.가전렌탈', '2.카셰어링', '3.라스트마일모빌리티']

#매출건수
#나이대
age_c = {
    1 : [],
    2 : [],
    3 : []
}

age_q = data.groupby(['기준년도','기준월','연령대_10','카테고리']).agg({'매출건수(건)':'sum'})

for a in [20,30,40,50]:
    for idx, category in enumerate(categories, start = 1):
        single_list = age_q.query('연령대_10 == "'+str(a)+'대" and 카테고리 =="'+category+'"')
        single_list.sort_values(by=['기준년도','기준월'], inplace = True)
        age_c[idx].append(single_list)

print(age_c)

#성별
gender_c = {
    1 : [],
    2 : [],
    3 : []
}

gender_q = data.groupby(['기준년도','기준월','성별','카테고리']).agg({'매출건수(건)':'sum'})

for g in ['남', '여']:
    for idx, category in enumerate(categories, start = 1):
        single_list = gender_q.query('성별 == "'+ g+'" and 카테고리 =="'+category+'"')
        single_list.sort_values(by=['기준년도','기준월'], inplace = True)
        gender_c[idx].append(single_list)
        
print(gender_c)

#매출금액
#나이대
age_s = {
    1 : [],
    2 : [],
    3 : []
}

age_q = data.groupby(['기준년도','기준월','연령대_10','카테고리']).agg({'매출금액(천원)':'sum'})

for a in [20,30,40,50]:
    for idx, category in enumerate(categories, start = 1):
        single_list = age_q.query('연령대_10 == "'+str(a)+'대" and 카테고리 =="'+category+'"')
        single_list.sort_values(by=['기준년도','기준월'], inplace = True)
        age_s[idx].append(single_list)

print(age_s)

#성별
gender_s = {
    1 : [],
    2 : [],
    3 : []
}

gender_q = data.groupby(['기준년도','기준월','성별','카테고리']).agg({'매출금액(천원)':'sum'})

for g in ['남', '여']:
    for idx, category in enumerate(categories, start = 1):
        single_list = gender_q.query('성별 == "'+ g+'" and 카테고리 =="'+category+'"')
        single_list.sort_values(by=['기준년도','기준월'], inplace = True)
        gender_s[idx].append(single_list)
        
print(gender_s)

class TimingCallback(Callback):
    def __init__(self):
        self.starttime = None
        self.timelogs = []
    def on_train_begin(self, logs={}):
        self.starttime= time.time()
        print("--start Time : ", self.starttime)
    def on_epoch_end(self, epoch, logs={}):
        midTime = time.time()-self.starttime
        self.timelogs.append(midTime)
        print("--Middle Time : ", midTime)
    def on_train_end(self, logs={}):
        LogsInNP = np.array(self.timelogs)
        print(LogsInNP)
        print(LogsInNP.shape)

TimeCB = TimingCallback()

#9개월 이용 18개월 예측
class LSTMprediction:
    def __init__(self, data, col, name):
        time_steps = 10
        time_prediction_length = 1
        self.time_series = data[col].to_numpy()
        X_data = []
        Y_data = []
        for idx in range(time_steps, len(self.time_series) - time_prediction_length + 1):
            X_data.append(self.time_series[idx - time_steps:idx].reshape(time_steps,1))
            Y_data.append(self.time_series[idx:idx + time_prediction_length].reshape(time_prediction_length,1))
        self.X_data = np.array(X_data, dtype=np.float64)
        self.Y_data = np.array(Y_data, dtype=np.float64)
#         print(self.Y_data[-1])
#         self.X_previous = np.array(time_series[27:27+time_steps], dtype=np.float64)
#         print(self.X_previous)
        self.name = name
        self.result_path = 'D:/공유경제/result/'
        self.checkpoint_filepath = self.result_path + 'models/'+name + '.hdf5'
    
    def getModel(self):
        self.model = Sequential()
        self.model.add(LSTM(128,return_sequences = True, input_shape = (self.X_data.shape[1],1), activation='relu'))
#         self.model.add(LSTM(128,activation='relu'))
        self.model.add(Dense(1))
        print(self.model.summary())
    
    def training(self):
        early_stopper = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100,
                                            verbose=1)
        save_best = callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath, 
                                              monitor='val_loss', 
                                              verbose=1, save_best_only=True,
                                              save_weights_only=True, mode='auto')
        self.getModel()
        self.model.compile(optimizer = optimizers.Adam(), loss = losses.MeanSquaredError())
        hist = self.model.fit(self.X_data,self.Y_data,epochs = 1000, batch_size = 5, verbose = 2, validation_split = 0.2,
                  callbacks=[early_stopper, save_best,TimeCB])
        
        training_loss = min(hist.history['val_loss'])
        print('final_loss = ', training_loss)
        
        return training_loss
    
    def predict(self, prediction_length = 77):
        self.getModel()
        self.model.load_weights(self.checkpoint_filepath)
        pre_source = deque(self.X_data[-1])
        prediction = []
        for idx in range(prediction_length):
#             print(idx)
            x = np.array([pre_source])
#             print(x)
            pre_test = (self.model.predict(x))
#             print(x,pre_test)
            pre_test = pre_test[0][-1]
#             print(pre_test)
            prediction.append(pre_test[0])
            pre_source.popleft()
            pre_source.append(pre_test)
            
        print(prediction)
        ts_list = self.time_series.tolist()
        print(len(ts_list), len(prediction))
        ts = ts_list + prediction
        plt.plot(ts)
        plt.legend(['ts_pre'], loc='best')
        plt.show()
        print (len(ts),ts)
        return ts

#age_c gender_c
#age_s gender_s
#123
#'매출건수(건)' , '매출금액(천원)'

result = {}
loss_results = {}

for idx, ages in enumerate(['20대','30대','40대','50대']):
    for i in [1,2,3]:
        model_name = ages+'_카테고리'+str(i)+'_매출건수(건)'
        print(model_name)
        LSTMmodel = LSTMprediction(age_c[i][idx],'매출건수(건)',model_name)
        loss_results[model_name] = LSTMmodel.training()
        result[model_name] = LSTMmodel.predict()
        
        
for idx, gens in enumerate(['남', '여']):
    for i in [1,2,3]:
        model_name = gens+'_카테고리'+str(i)+'_매출건수(건)'
        print(model_name)
        LSTMmodel = LSTMprediction(gender_c[i][idx],'매출건수(건)',model_name)
        loss_results[model_name] = LSTMmodel.training()
        result[model_name] = LSTMmodel.predict()
        
for idx, ages in enumerate(['20대','30대','40대','50대']):
    for i in [1,2,3]:
        model_name = ages+'_카테고리'+str(i)+'_매출금액(천원)'
        print(model_name)
        LSTMmodel = LSTMprediction(age_s[i][idx],'매출금액(천원)',model_name)
        loss_results[model_name] = LSTMmodel.training()
        result[model_name] = LSTMmodel.predict()
        
for idx, gens in enumerate(['남', '여']):
    for i in [1,2,3]:
        model_name = gens+'_카테고리'+str(i)+'_매출금액(천원)'
        print(model_name)
        LSTMmodel = LSTMprediction(gender_s[i][idx],'매출금액(천원)',model_name)
        loss_results[model_name] = LSTMmodel.training()
        result[model_name] = LSTMmodel.predict()

