import numpy as np
import pandas as pd
import math, os
import seaborn as sns
import matplotlib.pyplot as plt

class RegressionModel(object):

    def normalize(self, data):
        normalized = (data - min(data)) / (max(data) - min(data))
        return normalized

    def shuffle(self, X, Y): #隨機順序
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        return (X[randomize], Y[randomize])

    def loadData(self, file, train, predict): #讀取訓練資料
        data = pd.read_csv(file, encoding="big5", header=0)
        data.dropna(inplace=True)
        data = pd.get_dummies(data)

        zeroList = []
        # df = pd.DataFrame(data.iloc[0])
        # print(df)
        # df.to_csv("./experiment/parameter.csv")
        ##----- 處理負值 -----##
        for i in range(len(data)):
            if (data.iloc[i, predict] <= 0):  # & (data.iloc[i, predict] != -3):
                zeroList.append(i)
            for j in train:
                if data.iloc[i, j] <= 0:
                    zeroList.append(i)
                    break

        ##----- 處理負值 -----##

        rowLength = len(data)
        colLength = len(data.iloc[0])

        x_train = []
        y_train = []
        reg_no = []

        for i in range(rowLength):
            x_train.append([])

        for i in range(rowLength):
            for j in train:
                if data.iloc[i, j] < 0:
                    x_train[i].append(0)
                else:
                    x_train[i].append(data.iloc[i, j])
            # for s in range(23, colLength):
            # for s in range(23, 25): #學歷
            for s in range(25, 27): # 性別
            # for s in range(27, 34): # 入學方式
            # for s in range(34, 39):  # 出生地
            # for s in range(39, 42): # 身分
            # for s in range(42, 55): # 職業
                x_train[i].append(data.iloc[i, s])

            reg_no.append(data.iloc[i, 0])

            if data.iloc[i, predict] < 0:
                y_train.append(0)
            else:
                y_train.append(data.iloc[i, predict])
        x_train = np.array(x_train).astype(float)
        y_train = np.array(y_train).astype(float)

        #----- 處理負值 -----##
        x_train = np.delete(x_train, zeroList, axis=0)
        y_train = np.delete(y_train, zeroList, axis=0)
        #----- 處理負值 -----##

        # for s in range(len(x_train[0])):
        #     x_train[:, s] = self.normalize(x_train[:, s])

        # y_train = self.normalize(y_train)
        x_train, y_train = self.shuffle(x_train, y_train)
        # print("x_train size: ", x_train.shape, "y_train size: ", y_train.shape)
        return x_train, y_train

    def saveModel(self, W, save_dir, name): # 存權重模型
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.savetxt(os.path.join(save_dir, name), W)
        # np.savetxt(os.path.join(save_dir, 'b'), [b, ])

    def doRegression(self, x_train, y_train, lr=0.1, iteration=5000):
        W = np.zeros(len(x_train[0])).astype(float)
        x_t = x_train.transpose()
        sum_gradient = np.zeros(len(x_train[0]))
        print(W.shape)
        for epoch in range(iteration):
            hypothesis = np.dot(x_train, W)
            loss = (hypothesis - y_train)/2

            cost = np.sum(loss ** 2) / len(x_train)
            cost_a = math.sqrt(cost)
            gradient_W = np.dot(x_t, loss)
            sum_gradient += gradient_W ** 2

            W = W - lr * gradient_W / np.sqrt(sum_gradient)
            if (epoch+1) % 100 == 0 :
                print("Iteration %d | Cost: %f" % (epoch, cost_a))

        # print(hypothesis)
        return W

    def validation(self, W, x_test, y_test): # testing data validation
        y = np.dot(x_test, W)
        loss = (y - y_test)
        cost = np.sum(loss ** 2) / len(x_test)
        cost_average = math.sqrt(cost)
        print("Cost: %f" % cost_average)


regression = RegressionModel()
iteration = 5000
# train_10 = [1, 5, 6]
# train_11 = [1, 5, 6]  # 微積分及演習....2電路學
# train_12 = [1, 5, 6]
# train_13 = [4, 7, 9]
# train_14 = [2, 3, 8]
# train_15 = [1, 5, 6, 10, 11, 12]
# train_16 = [1, 5, 6, 10, 11, 12]
# train_17 = [1, 5, 6, 10, 11, 12]
# train_18 = [4, 7, 9, 13]
# train_19 = [1, 5, 6, 10, 11, 12]
# train_20 = [1, 5, 6, 10, 11, 12, 15, 16, 17, 19]
# train_21 = [4, 7, 9, 13, 18]
# train_22 = [5, 7, 9, 13, 18, 21]
train_10 = [1, 2, 3, 5, 6, 8]
# train_10 = [1,2,3,4,5,6,7,8,9,10,11,12]
train_11 = [1, 2, 3, 5, 6, 8]  # 微積分及演習....2電路學
train_12 = [1, 2, 3, 5, 6, 8, 10, 11]
train_13 = [4, 7, 9]
train_14 = [1, 2, 3, 5, 6, 8, 10, 11, 12]
train_15 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14]
train_16 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15]
train_17 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16]
train_18 = [4, 7, 9, 13]
train_19 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17]
train_20 = [1, 2, 3, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17, 19]
train_21 = [4, 7, 9, 13, 18]
train_22 = [4, 7, 9, 13, 18, 21]
predict_10 = 10  # 1工程數學
predict_11 = 11  # 1機率
predict_12 = 12  # 1電子學
predict_13 = 13  # 1電子學實習
predict_14 = 14  # 1電路學
predict_15 = 15  # 2工程數學
predict_16 = 16  # 2微處理機
predict_17 = 17  # 2電子學
predict_18 = 18  # 2電子學實習
predict_19 = 19  # 2電路學
predict_20 = 20  # 1訊號與系統
predict_21 = 21  # 2實務專題(一)
predict_22 = 22  # 1實務專題(二)

train = [train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18, train_19, train_20,
         train_21, train_22]
predict = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


from sklearn.linear_model import LinearRegression, LassoCV, Ridge, LassoLarsCV, ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

# lr = LinearRegression()

# year = ["allTo103"]
#
# for i in range(len(year)):
#     for j in [0]:
#         x_train, y_train = regression.loadData("studentData/studentInfo/"+str(year[i])+"_cleanData.csv", train[j], predict[j])
#         lr.fit(x_train, y_train)
#         mse = np.sqrt(np.mean((lr.predict(x_train) - y_train) ** 2))
#         print(mse)
#         coef  = lr.coef_
#
#         df = pd.DataFrame(coef)
#         df.to_csv("./experiment/W2.csv")
#         print(coef)
#         # joblib.dump(lr, 'experiment/save_'+str(year[i])+'/lr_'+str(j+11))
#         # joblib.load('model/save_100/lr_12')

# x_train, y_train = regression.loadData("studentData/studentInfo/100_cleanData.csv", train[3], predict[3])
# x_train1, y_train1 = regression.loadData("studentData/studentInfo/99_cleanData.csv", train[3], predict[3])
# x_test, y_test = regression.loadData("studentData/studentInfo/101_cleanData.csv", train[3], predict[3])
#
# lm = LinearRegression()
# lm1 = LinearRegression()
# lassocv = LassoCV(eps=1e-5)
# lassocv1 = LassoCV(eps=1e-5)
# gb = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=0.01, loss="ls")
# gb2 = GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=2, learning_rate=0.01, loss="ls")
#
# # fit 讓資料去training
# lm.fit(x_train, y_train)
# lassocv.fit(x_train, y_train)
# lm1.fit(x_train1, y_train1)
# lassocv1.fit(x_train1, y_train1)
# gb.fit(x_train, y_train)
# gb2.fit(x_train1, y_train1)
#
# mse = np.sqrt(np.mean((lm.predict(x_train) - y_train) ** 2))
# mse1 = np.sqrt(np.mean((lm1.predict(x_train1) - y_train1) ** 2))
# mse2 = np.sqrt(np.mean((lassocv.predict(x_train) - y_train) ** 2))
# mse3 = np.sqrt(np.mean((lassocv1.predict(x_train1) - y_train1) ** 2))
# mse4 = np.sqrt(np.mean((gb.predict(x_train) - y_train) ** 2))
# mse5 = np.sqrt(np.mean((gb.predict(x_train1) - y_train1) ** 2))
#
# print(mse, mse1, mse2, mse3, mse4, mse5)
#
# y_pre_lr_100 = lm.predict(x_test)
# y_pre_lr_99  = lm1.predict(x_test)
# y_pre_la_100= lassocv.predict(x_test)
# y_pre_la_99 = lassocv1.predict(x_test)
#
# p_mse = np.sqrt(np.mean((y_pre_lr_100 - y_test) ** 2))
# p_mse1 = np.sqrt(np.mean((y_pre_lr_99 - y_test) ** 2))
# p_mse2 = np.sqrt(np.mean((y_pre_la_100 - y_test) ** 2))
# p_mse3 = np.sqrt(np.mean((y_pre_la_99 - y_test) ** 2))
#
# print(p_mse, p_mse1, p_mse2, p_mse3)
#
# times = []
# result =  y_pre_lr_100*0.5+y_pre_lr_99*0.5+y_pre_la_100*0+y_pre_la_99*0
# result_mse = np.sqrt(np.mean((result - y_test) ** 2))
# print(result_mse)


## ------------------- ensemble version 1  ----------------- ##
# x_train, y_train = regression.loadData("studentData/studentInfo/100_cleanData.csv", train[0], predict[0])
# x_test, y_test = regression.loadData("studentData/studentInfo/99_cleanData.csv", train[0], predict[0])
#
# lm = LinearRegression()
# lassocv = LassoCV(eps=1e-5)
# ridge = Ridge(alpha=1e-5)
# elasticnetcv = ElasticNetCV(eps=1e-5)
#
# # print(lm.coef_, np.sqrt(mse))
# def RSME(estimator, x_train, y_train, cv=5):
#     loss = cross_val_score(estimator, x_train, y_train, cv=cv, scoring="neg_mean_squared_error")
#     print((np.sqrt(-loss)).mean())
# # loss = cross_val_score(lm, x_train, y_train, cv=5, scoring="neg_mean_squared_error",n_jobs= 4)
#
# # Gradient Boosting
# GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10,
#                                    loss='huber', random_state =5)
# # RSME(GBoost,x_train,y_train)
# RSME(lm, x_train, y_train)
# RSME(lassocv, x_train, y_train)
# RSME(ridge, x_train, y_train)
# RSME(elasticnetcv, x_train, y_train)
#
# # GBoost.fit(x_train, x_test)
# lm.fit(x_train, y_train)
# lassocv.fit(x_train, y_train)
# ridge.fit(x_train, y_train)
# elasticnetcv.fit(x_train, y_train)
#
# mse = np.mean((lm.predict(x_train) - y_train) ** 2)
# mse2 = np.mean((lassocv.predict(x_train) - y_train) ** 2)
# mse3 = np.mean((ridge.predict(x_train) - y_train) ** 2)
# mse4 = np.mean((elasticnetcv.predict(x_train) - y_train) ** 2)
# # mse5 = np.sqrt(np.mean((GBoost.predict(x_train) - y_train) ** 2))
#
# test_mse = np.sqrt(np.mean((lm.predict(x_test) - y_test) ** 2))
# test_mse2 = np.sqrt(np.mean((lassocv.predict(x_test) - y_test) ** 2))
# test_mse3 = np.sqrt(np.mean((ridge.predict(x_test) - y_test) ** 2))
# test_mse4 = np.sqrt(np.mean((elasticnetcv.predict(x_test) - y_test) ** 2))
#
# y_pre_linear = lm.predict(x_test)
# y_pre_lassocv = lassocv.predict(x_test)
# y_pre_ridge= ridge.predict(x_test)
# y_pre_elasticnetcv = elasticnetcv.predict(x_test)
#
# result = y_pre_linear*0.3+y_pre_lassocv*0.2+y_pre_ridge*0.3+y_pre_elasticnetcv*0.2
# print(test_mse, test_mse2, test_mse3, test_mse4)
# result_mse = np.sqrt(np.mean(( result- y_test) ** 2))
# print(result_mse)
# print(np.sqrt(mse), np.sqrt(mse2), np.sqrt(mse3), np.sqrt(mse4))
#
# print(result)
#
# W = regression.doRegression(x_train, y_train, lr, iteration)
# regression.validation(W, x_test, y_test)
#
# # print(W)
#
# # ensemble version 1

course = [9]
#-------------- 進行迴歸分析 training ---------------##
for i in course:
    x_train, y_train = regression.loadData("experiment/allTo103_cleanData.csv", train[i], predict[i])
    # print(x_train, y_train)
    W = regression.doRegression(x_train, y_train, 0.1)
    df = pd.DataFrame(W)
    # df.to_csv("./experiment/W.csv")
    regression.saveModel(W, "experiment/", "w_"+str(i+11))
    print(W, sum(W))
# ================================================ ##
#

# -------------- 進行迴歸分析 testing ---------------##
for i in course:
    x_test, y_test = regression.loadData("experiment/104_cleanData.csv", train[i], predict[i])
    # print(x_test, y_test)
    W = np.loadtxt(os.path.join("experiment/", "w_"+str(i+11)))
    regression.validation(W, x_test, y_test)
# ================================================ ##


##-------------- 進行在學學生預測 ---------------##

# data = pd.read_csv("studentData/studentInfo/101_cleanData.csv", encoding="big5", header=0)
# data.dropna(inplace=True)
# data = pd.get_dummies(data)
#
# currentData = []
# predictData = []
# reg_no = []
# for i in range(len(data)):
#     currentData.append([])
#
# for i in range(len(data)):
#     reg_no.append(data.iloc[i, 0])
#     for j in range(1,11):
#         if data.iloc[i, j] < 0:
#             currentData[i].append(0)
#         else:
#             currentData[i].append(data.iloc[i, j])
#     for k in range(11, 23):
#         currentData[i].append(0)
#
#     for s in range(23, len(data.iloc[0])):
#         currentData[i].append(data.iloc[i, s])
#
# currentData = np.array(currentData).astype(float)
# reg_no = np.array(reg_no).astype(float).reshape(len(data),1)
#
#
# # for s in range(10):
# #     currentData[:, s] = regression.normalize(currentData[:, s])
#
# def pred(train, currentData):
#     predictData = []
#     for i in range(len(currentData)):
#         if i in train:
#             predictData.append(currentData[:,i-1])
#         if i in range(22, 65):
#             predictData.append(currentData[:,i])
#     predictData = np.array(predictData)
#     return predictData.transpose()
#
#
# for i in range(len(train)):
#     predictData = pred(train[i], currentData)
#     W = np.loadtxt(os.path.join("model/", "w_"+str(i+11)))
#     score = np.dot(predictData, W)
#     score = [round(i, 1) for i in score] # 四捨五入
#     currentData[:, predict[i]-1] = score

# g = sns.jointplot(x=currentData[:,0], y=currentData[:,5], kind="reg")

# g = sns.heatmap(data.corr())
# plt.show()

# concatenateData = np.concatenate((reg_no, currentData[:,0:22]), axis=1)
# df = pd.DataFrame(concatenateData)
# df.to_csv("studentData/predict/101_predictData.csv")
# ================================================ ##
