# 訓練神經網路模型，並存起來
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from sklearn.model_selection import train_test_split
class OfflineNeuralNetworkModel(object):
    regularization = 0 # L2 regularization
    # --------------------- Tensorflow -------------------------- #
    def add_layer(self, inputs, in_size, out_size, keep_prob, activation_function = None, W=None, b=None):
        if W is None:
            W = tf.Variable(tf.random_normal([in_size, out_size]), name="w")
        if b is None:
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
        Weights = W
        biases = b
        self.regularization = tf.nn.l2_loss(Weights)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        # dropout
        # Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def normalize(self, data):
        normalized = (data-min(data)) / (max(data) - min(data))
        return normalized

    def shuffle(self, X, Y): #隨機順序
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        return (X[randomize], Y[randomize])

    def train_validation_split(self, X, Y, ratio):
        leng = math.floor(len(X)*ratio)
        x_train = X[:leng,:]
        y_train = Y[:leng,:]
        x_val = X[leng:,:]
        y_val =  Y[leng:,:]
        print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

        return x_train, y_train, x_val, y_val

    def loadData(self, file, train, predict):

        data = pd.read_csv(file, encoding="big5", header=0)
        data.dropna(inplace=True)
        data = pd.get_dummies(data)
        # data.iloc[0].to_csv("aaa.csv")
        ##----- 處理負值 -----##
        zeroList = []
        # for i in range(len(data)):
        #     # if (data.iloc[i, predict] <= 0): # & (data.iloc[i, predict] != -3):
        #     #     zeroList.append(i)
        #     for j in train:
        #         if (data.iloc[i, j] <= 0): #& (data.iloc[i, j] != -3):
        #             zeroList.append(i)
        #             break
        ##----- 處理負值 -----##

        rowLength = len(data)
        colLength = len(data.iloc[0])

        x_train = []
        y_train = []
        reg_no =[]

        for i in range(rowLength):
            x_train.append([])
            y_train.append([])

        for i in range(rowLength):
            for j in train:
                if (data.iloc[i, j] == -3) :
                    x_train[i].append(40)
                elif data.iloc[i, j] < 0:
                    x_train[i].append(0)
                else:
                    x_train[i].append(data.iloc[i,j])

            # for s in range(23, colLength):
            # for s in range(23, 25): #學歷
            # for s in range(25, 27): # 性別
            # for s in range(27, 34): # 入學方式
            # for s in range(34, 39): # 出生地
            # for s in range(39, 42): # 身分
            for s in range(42, 55): # 職業
                x_train[i].append(data.iloc[i, s])

            # reg_no.append(data.iloc[i, 0])

            if (data.iloc[i, predict] == -6):
                y_train[i].append(40)
            elif data.iloc[i, predict] < 0:
                y_train[i].append(0)
            else:
                y_train[i].append(data.iloc[i, predict])

        x_train = np.array(x_train).astype(np.float32)
        y_train = np.array(y_train).astype(np.float32)

        #----- 處理負值 -----##
        x_train = np.delete(x_train, zeroList, axis=0)
        y_train = np.delete(y_train, zeroList, axis=0)
        #----- 處理負值 -----##
        #
        # x_train, y_train = self.shuffle(x_train, y_train)
        #
        # for s in range(len(x_train[0])):
        #     x_train[:, s] = self.normalize(x_train[:, s])
        # y_train = self.normalize(y_train)
        # print(x_train, y_train)

        print(x_train.shape, y_train.shape)
        return x_train, y_train

    # 根據學年和學期來選擇要訓練的課程 online training
    def trainingYearAndSemester(self, semester, trainingYear, current=106, online=False): # current 代表目前學年度, training代表要訓練的學年度, semester 0: 上, 1: 下, 2: 全
        second_fall = [0, 1, 2, 3, 4]
        second_summer = [5, 6, 7, 8, 9]
        third_fall = [10]
        third_summer = [11]
        fourth = [12]

        trainingCourse = [second_fall, second_summer, third_fall, third_summer, fourth]
        course = []
        if online == False: # offline
            if current-3 > trainingYear:
                for i in range(len(trainingCourse)):
                    course.append(trainingCourse[i])
                return course

            else:
                for i in range(1, 4):  # offline total training
                    if trainingYear == current - i:
                        if semester == 0:
                            for j in range((i - 1) * 2):
                                course.append(trainingCourse[j])
                            return course
                        elif semester == 1:
                            for j in range((i * 2)-1):
                                course.append(trainingCourse[j])
                            return course

        if online == True & (trainingYear >= current-3):
            for i in range(1,4):  # online training
                if trainingYear == current - i:
                    if semester == 0:
                        return trainingCourse[(i-1)*2]
                    elif semester == 1:
                        return trainingCourse[(i*2)-1]




    def trainingParameter(self):

        # train_10 = [1,5,6]
        # # train_10 = [1,2,3,4,5,6,7,8,9,10,11,12]
        # train_11 = [1,5,6] #微積分及演習....2電路學
        # train_12 = [1,5,6]
        # train_13 = [4,7,9]
        # train_14 = [2,3,8]
        # train_15 = [1,5,6,10,11,12]
        # train_16 = [1,5,6,10,11,12]
        # train_17 = [1,5,6,10,11,12]
        # train_18 = [4,7,9,13]
        # train_19 = [1,5,6,10,11,12]
        # train_20 = [1,5,6,10,11,12,15,16,17,19]
        # train_21 = [4,7,9,13,18]
        # train_22 = [5,7,9,13,18,21]

        train_10 = [1,2,3,5,6,8]
        # train_10 = [1,2,3,4,5,6,7,8,9,10,11,12]
        train_11 = [1,2,3,5, 6,8]  # 微積分及演習....2電路學
        train_12 = [1, 2,3,5, 6, 8,10, 11]
        train_13 = [4, 7, 9]
        train_14 = [1,2, 3,5,6, 8,10,11,12]
        train_15 = [1,2, 3,5,6, 8,10,11,12,14]
        train_16 = [1,2, 3,5,6, 8,10,11,12,14,15]
        train_17 = [1,2, 3,5,6, 8,10,11,12,14,15,16]
        train_18 = [4, 7, 9, 13]
        train_19 = [1,2, 3,5,6, 8,10,11,12,14,15,16,17]
        train_20 = [1,2, 3,5,6, 8,10,11,12,14,15,16,17,19]
        train_21 = [4, 7, 9, 13, 18]
        train_22 = [4, 7, 9, 13, 18, 21]

        # train_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # train_11 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # train_12 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # train_13 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        # train_14 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # train_15 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # train_16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # train_17 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        # train_18 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        # train_19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        # train_20 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        # train_21 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # train_22 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

        predict_10 = 10 #1工程數學
        predict_11 = 11 #1機率
        predict_12 = 12 #1電子學
        predict_13 = 13 #1電子學實習
        predict_14 = 14 #1電路學
        predict_15 = 15 #2工程數學
        predict_16 = 16 #2微處理機
        predict_17 = 17 #2電子學
        predict_18 = 18 #2電子學實習
        predict_19 = 19 #2電路學
        predict_20 = 20 #1訊號與系統
        predict_21 = 21 #2實務專題(一)
        predict_22 = 22 #1實務專題(二)

        train = [train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18, train_19, train_20, train_21, train_22]
        predict = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        return train, predict


