from sklearn.svm import SVR
from OfflineNeuralNetworkModel import  OfflineNeuralNetworkModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import util


def trainingParameter():
    train_10 = [1,2,3,5,6,8]
    train_11 = [1,2,3,5, 6,8]  # 微積分及演習....2電路學
    train_12 = [1, 2,3,5, 6, 8,10, 11]
    train_13 = [4, 7, 9]
    train_14 = [1,2,5,6, 8,10,11,12]
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

    train = [train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18, train_19,
             train_20, train_21, train_22]
    predict = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    return train, predict


onnm = OfflineNeuralNetworkModel()
# train, predict = onnm.trainingParameter()
train, predict = trainingParameter()

l = onnm.trainingYearAndSemester(0, 104,online=True)
tc = util.flattenList(l)
print(tc)
#
for i in [9]:
    x_train, y_train = onnm.loadData("experiment/allTo103_cleanData.csv", train[i], predict[i])
    x_test, y_test = onnm.loadData("experiment/104_cleanData.csv", train[i], predict[i])

    y_train = np.ravel(y_train, order='C')
    y_test = np.ravel(y_test, order='C')


    # clf = RandomForestRegressor(n_estimators=10, random_state=0)
    # clf = SVR(kernel='linear', C=1e3, gamma=0.1)
    clf = LinearRegression()

    clf.fit(x_train,y_train)
    print(f_regression(x_train, y_train)[1]) # p-value 係數檢定
    print(clf.coef_)
    import matplotlib.pyplot as plt
    plt.plot(clf.coef_)
    # plt.show()
    print(x_test[10], y_test[10], clf.predict(x_test)[10])

    y_train = y_train.reshape(1,-1)
    y_test = y_test.reshape(1,-1)

    training_mse = np.sqrt(np.mean((clf.predict(x_train) - y_train) ** 2))
    testing_mse = np.sqrt(np.mean((clf.predict(x_test) - y_test) ** 2))
    print(training_mse, testing_mse,'\n')

    print(np.sum(np.multiply(clf.coef_, x_test[10]))+6.38)

    # util.plotPredictScore(clf.predict(x_test), y_test)
    # df = pd.DataFrame({"predict":clf.predict(x_test), "real": y_test.reshape(len(x_test))})
    #
    # x = clf.predict(x_test)
    # y =  y_test.reshape(len(x_test))
    #
    # util.measure(x, y)
    # df.to_csv("./experiment/svm/score.csv")

