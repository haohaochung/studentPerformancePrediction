# 讀取由training data 訓練的模型，使用testing data檢查loss
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import util
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def add_layer( inputs, in_size, out_size, keep_prob, activation_function=None, W=None, b=None):
    if W is None:
        W = tf.Variable(tf.random_normal([in_size, out_size]), name="w")
    if b is None:
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
    Weights = W
    biases = b
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # dropout
    # Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def predict(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={x_feeds: v_xs})
    loss = tf.sqrt(tf.reduce_mean(tf.square(v_ys - y_pre)))
    print(sess.run(loss))
    return y_pre, sess.run(loss)

def visualize(pre, y_real):
    df = pd.DataFrame(y_real.reshape(len(y_real)), pre.reshape(len(y_real)))
    x = pre.reshape(len(y_real))
    y = y_real.reshape(len(y_real))
    # util.measure(x, y)

    # df.to_csv("./experiment/svm/score2.csv")


def scoreWeights(count): # 106, 105
    index = count*(count+1)/2
    weight_list = []
    for y in range(count): # training year
        weight_list.append((1/index)*(y+1))
    print(weight_list)
    return weight_list

def flattenList(trainingCourse):
    if any(isinstance(i, list) for i in trainingCourse):  # is nested list
        flat_list = [item for sublist in trainingCourse for item in sublist]
    else:
        flat_list = trainingCourse
    print("Predicting course", flat_list)
    return flat_list

from OfflineNeuralNetworkModel import OfflineNeuralNetworkModel
onnm = OfflineNeuralNetworkModel()
train, predicts = onnm.trainingParameter()
l = onnm.trainingYearAndSemester(0, 102,online=False)
tc = flattenList(l)
# year = ["allTo104"]
# year = ["99"]
year = ["99", "100", "101", "102","103","104"]
layer_1_hidden = 256
layer_2_hidden = 256
# layer_3_hidden = 64

for i in [4]: #預測的科目
    x_test, y_test = onnm.loadData("studentData/studentInfo/105_cleanData.csv", train[i], predicts[i])

    year_prediction = []  # 科目每一年的預測list
    final_prediction = 0  # 科目將所有年集合起來的預測
    year_loss = []
    sw = scoreWeights(len(year))
    for j in range(len(year)): # 預測的年份
        print("=========== Training year {0} subject {1} ==============".format(str(year[j]), str(predicts[i])))

        x_feeds = tf.placeholder(tf.float32, [None, len(x_test[0])], name='x_input')  # testing data 中，只需要餵input就夠，output為使用training data 訓練得來的參數

        W = np.loadtxt(os.path.join("model/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
        b = np.loadtxt(os.path.join("model/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
        W2 = np.loadtxt(os.path.join("model2/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
        b2 = np.loadtxt(os.path.join("model2/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
        # W3 = np.loadtxt(os.path.join("model3/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
        # b3 = np.loadtxt(os.path.join("model3/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)

        l1 = add_layer(x_feeds, len(x_test[0]), layer_1_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W, b=b)  # pre_training
        l2 = add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W2, b=b2)  # pre_training
        # l3 = add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W3, b=b3)  # pre_training

        prediction = add_layer(l2, layer_2_hidden, 1, keep_prob=1, activation_function=None)

        # l1 = add_layer(x_feeds, len(x_test[0]), 128, keep_prob=1, activation_function=tf.nn.relu) # without pre_training
        # prediction = add_layer(l1, 128, 1, keep_prob=1, activation_function=None) # without pre_training

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Without Ensemble
        saver.restore(sess, './my_net_preTraining/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(
            1) + '.ckpt')  # pre_training
        # saver.restore(sess, './my_net/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(val) + '.ckpt') # without pre_training
        p, l = predict(x_test, y_test)
        mse = np.sqrt(np.mean((p - y_test) ** 2))
        year_prediction.append(p)
        year_loss.append(mse)
        print(l, mse)
        # Without Ensemble


        # val_prediction = 0
        # val_index=0
        # val_loss = []
        # for val in range(2): # 集合五個validation，乘上權重0.2
        #     saver.restore(sess, './my_net_preTraining/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(val) + '.ckpt') # pre_training
        #     # saver.restore(sess, './my_net/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(val) + '.ckpt') # without pre_training
        #     p, l = predict(x_test, y_test)
        #     if l < 40: # validation loss 超過20的不計算
        #         val_prediction += p #* 0.2 # 每個validation 各佔0.2
        #         val_index+=1
        #     val_loss.append(l)
        #
        # val_prediction = val_prediction / val_index
        # mse = np.sqrt(np.mean((val_prediction - y_test) ** 2))
        # year_prediction.append(val_prediction)
        # year_loss.append(mse)
        # print("Final loss: {0}, Minimal of five val_loss: {1}".format(mse, np.min(val_loss) > mse), '\n')

        tf.reset_default_graph() # 進行下一次的session前，要先清掉graph

    # 集合每一年的預測
    order = sorted(year_loss, reverse=True)
    for k in range(len(year)):
        for m in [m for m, x in enumerate(year_loss) if x == order[k]]:
            print(order, year_loss, m)
            final_prediction += year_prediction[m] * sw[k]

    result_mse = np.sqrt(np.mean((final_prediction- y_test) ** 2))
    print("=========== Subject {0} Final ==============".format(str(predicts[i])))
    print(result_mse, np.min(year_loss) > result_mse, '\n')
    # util.plotPredictScore(final_prediction, y_test)

    visualize(final_prediction, y_test)
    tf.reset_default_graph() # 進行下一次的session前，要先清掉graph

