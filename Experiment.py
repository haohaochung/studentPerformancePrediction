# 不同隱藏層層數與神經元個數實驗
import tensorflow as tf
import numpy as np
import pandas as pd
import timeit
import os
import util
from OfflineNeuralNetworkModel import OfflineNeuralNetworkModel
from autoEncoder import autoEncoder # from 檔名 import class
from sklearn.model_selection  import KFold, train_test_split
import time_affect_experiment as exp

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def early_stopping(sess, x_val, y_val):
    y_val_pre = (sess.run(prediction, feed_dict={x_feeds: x_val, keep_prob: 1}))
    valLoss = tf.sqrt(tf.reduce_mean(tf.square(y_val - y_val_pre)))
    return valLoss

def run_train(sess, x_train, y_train, x_val, y_val, iteration, kf_count, best_validation_loss):
    train_loss =[]
    val_loss=[]
    last_improvement = 0

    sess.run(tf.global_variables_initializer())

    for step in range(iteration):
        t, l, p = sess.run([train_step, loss, prediction], feed_dict={x_feeds: x_train, y_feeds: y_train, keep_prob: 0.5})

        if step % 100 == 0:
            trainLoss = (sess.run(loss, feed_dict={x_feeds: x_train, y_feeds: y_train, keep_prob: 0.5}))
            valLoss = early_stopping(sess, x_val, y_val)
            train_loss.append(trainLoss)
            val_loss.append(sess.run(valLoss))

            print("Training loss {0:.4f}, Validation loss {1:.4f}".format((trainLoss),(sess.run(valLoss))))

            if sess.run(valLoss) < best_validation_loss:
                best_validation_loss = sess.run(valLoss)
                last_improvement = step
                saver.save(sess, './experiment/year_' + str(year[i]) + '/net_' + str(predict[j])+"_"+ str(kf_count) + '.ckpt')

        if step - last_improvement > 500:
            print("Last step is {0}".format(last_improvement))
            break
    print("Finish all step")

    return train_loss, val_loss

def saveModel(W, save_dir, name): # 存權重模型
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.savetxt(os.path.join(save_dir, name), W)

def flattenList(trainingCourse):
    if any(isinstance(i, list) for i in trainingCourse):  # is nested list
        flat_list = [item for sublist in trainingCourse for item in sublist]
    else:
        flat_list = trainingCourse

    return flat_list

def scoreWeights(count): # 106, 105
    index = count*(count+1)/2
    weight_list = []
    for y in range(count): # training year
        weight_list.append((1/index)*(y+1))
    print(weight_list)
    return weight_list

def predicted(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={x_feeds: v_xs})
    loss = tf.sqrt(tf.reduce_mean(tf.square(v_ys - y_pre)))
    # print(sess.run(loss))
    return y_pre, sess.run(loss)

def scoreWeights(count): # 106, 105
    index = count*(count+1)/2
    weight_list = []
    for y in range(count): # training year
        weight_list.append((1/index)*(y+1))
    print(weight_list)
    return weight_list

def cross_validate(session, iteration, year, course, split_size=3):
    results = []
    kf_count = 0
    kf = KFold(n_splits=split_size)
    for train_idx, val_idx in kf.split(x_train, y_train):
        train_x = x_train[train_idx]
        train_y = y_train[train_idx]
        val_x = x_train[val_idx]
        val_y = y_train[val_idx]
        train_loss, val_loss = run_train(session, train_x, train_y, val_x, val_y, iteration, kf_count, best_validation_loss=1000000)
        kf_count += 1
        loss_output = pd.DataFrame()
        loss_output["train_loss"+ str(kf_count)] = train_loss
        loss_output["val_loss"+ str(kf_count)] = val_loss
        # loss_output.to_csv("./output/loss_" + year + "_" + course + "_" + str(kf_count) + ".csv")
        # loss_output.to_csv("./output/noPre/loss_" + year + "_" + course + "_" + str(kf_count) + ".csv")

        results.append(session.run(loss, feed_dict={x_feeds: val_x, y_feeds: val_y, keep_prob:1}))
    return results

start = timeit.default_timer()

onnm = OfflineNeuralNetworkModel()
train, predict = onnm.trainingParameter()
l = onnm.trainingYearAndSemester(0, 103,online=True)
tc = flattenList(l)
year = [99,100,101,102,103,104]
# year = ["allTo103"]

iteration = 5000
layer_1_hidden = 256
layer_2_hidden = 128
# layer_3_hidden = 256
# layer_4_hidden = 128
# print(tc)

# Training
course = [9]
# course = [0,1,2,3,4,5,6,7,8,9,10]
# for i in range(len(year)):
#     for j in course:
#         print("===========Training year {0} subject {1} ==============".format(str(year[i]), str(predict[j])))
#         x_train, y_train = onnm.loadData("studentData/studentInfo/" + str(year[i]) + "_cleanData.csv", train[j], predict[j])
#         keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#         x_feeds = tf.placeholder(tf.float32, [None, len(x_train[0])], name='x_input')
#         y_feeds = tf.placeholder(tf.float32, [None, 1], name='y_input')
#         #
#         print("Pre-training hidden layer 1...............................")
#         da = autoEncoder(layer_name="layer_1", input=x_train, n_visible=len(x_train[0]), n_hidden=layer_1_hidden)
#         tPreTrainingStart = timeit.default_timer()
#         l1w, l1b, l1o = da.train()
#         saveModel(l1w, "experiment/model/", "W"+str(year[i])+"_"+str(predict[j]))
#         saveModel(l1b, "experiment/model/", "b"+str(year[i])+"_"+str(predict[j]))
#         tPreTrainingEnd = timeit.default_timer()
#         print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))
#
#         print("Pre-training hidden layer 2...............................")
#         da2 = autoEncoder(layer_name="layer_2", input=l1o, n_visible=layer_1_hidden, n_hidden=layer_2_hidden)
#         tPreTrainingStart = timeit.default_timer()
#         l2w, l2b, l2o = da2.train()
#         saveModel(l2w, "experiment/model2/", "W" + str(year[i]) + "_" + str(predict[j]))
#         saveModel(l2b, "experiment/model2/", "b" + str(year[i]) + "_" + str(predict[j]))
#         tPreTrainingEnd = timeit.default_timer()
#         print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))
#         #
#         # print("Pre-training hidden layer 3...............................")
#         # da3 = autoEncoder(layer_name="layer_3", input=l2o, n_visible=layer_2_hidden, n_hidden=layer_3_hidden)
#         # tPreTrainingStart = timeit.default_timer()
#         # l3w, l3b, l3o = da3.train()
#         # saveModel(l3w, "experiment/model3/", "W" + str(year[i]) + "_" + str(predict[j]))
#         # saveModel(l3b, "experiment/model3/", "b" + str(year[i]) + "_" + str(predict[j]))
#         # tPreTrainingEnd = timeit.default_timer()
#         # print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))
#         # #
#         #
#         # print("Pre-training hidden layer 4...............................")
#         # da3 = autoEncoder(layer_name="layer_4", input=l3o, n_visible=layer_3_hidden, n_hidden=layer_4_hidden)
#         # tPreTrainingStart = timeit.default_timer()
#         # l4w, l4b, l4o = da3.train()
#         # saveModel(l4w, "experiment/model4/", "W" + str(year[i]) + "_" + str(predict[j]))
#         # saveModel(l4b, "experiment/model4/", "b" + str(year[i]) + "_" + str(predict[j]))
#         # tPreTrainingEnd = timeit.default_timer()
#         # print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))
#         # #
#
#
#         # # add hidden layer
#         l1 = onnm.add_layer(x_feeds, len(x_train[0]), layer_1_hidden, keep_prob=0.5, activation_function=tf.nn.relu, W=l1w, b=l1b) # with pre_training
#         l2 = onnm.add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob=0.5, activation_function=tf.nn.relu, W=l2w, b=l2b) # with pre_training
#         # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob=0.5, activation_function=tf.nn.relu, W=l3w, b=l3b) # with pre_training
#         # l4 = onnm.add_layer(l3, layer_3_hidden, layer_4_hidden, keep_prob=0.5, activation_function=tf.nn.relu, W=l4w, b=l4b) # with pre_training
#
#         # l1 = onnm.add_layer(x_feeds, len(x_train[0]), layer_1_hidden, keep_prob, activation_function=tf.nn.relu)  # without pre_training
#         # l2 = onnm.add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob, activation_function=tf.nn.relu)  # without pre_training
#         # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob, activation_function=tf.nn.relu)  # without pre_training
#         #
#         # prediction = onnm.add_layer(l3, layer_3_hidden, 1, keep_prob, activation_function=None)
#         #
#         prediction = onnm.add_layer(l2, layer_2_hidden, 1, keep_prob=1, activation_function=None)
#         # prediction = onnm.add_layer(l1, layer_1_hidden, 1, keep_prob, activation_function=None)
#
#
#         loss = tf.sqrt(tf.reduce_mean(tf.square(y_feeds - prediction))) + onnm.regularization * 0.01
#         # loss = (tf.reduce_mean(tf.square(y_feeds - prediction))) + onnm.regularization * 0.01
#
#         optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
#         train_step = optimizer.minimize(loss)
#         saver = tf.train.Saver()
#         # Launch the graph
#         with tf.Session() as sess:
#             # Without Ensemble
#             train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.2)
#             # print(train_x.shape)
#             train_loss, val_loss = run_train(sess, train_x, train_y, val_x, val_y, iteration, 1, best_validation_loss=10000000)
#             loss_output = pd.DataFrame()
#             loss_output["train_loss" + str(1)] = train_loss
#             loss_output["val_loss" + str(1)] = val_loss
#             loss_output.to_csv("./experiment/loss_" + str(year[i]) + "_" + str(predict[j]) + ".csv")
#
#             # # With Ensemble
#             # results = cross_validate(sess, iteration, str(year[i]), str(predict[j]))
#             # print("Cross-validation result: %s" % results)
#
#         tf.reset_default_graph()  # 進行下一次的session前，要先清掉graph
# stop = timeit.default_timer()
# Training

# Testing
# course = [5,6,7,8,9,10]
train, predicts = onnm.trainingParameter()
sw = scoreWeights(len(year))
for i in course: #預測的科目
    # for classes in [0,1,2]:
        x_test, y_test = onnm.loadData("studentData/studentInfo/105_cleanData.csv", train[i], predicts[i])
    #     x_test, y_test = exp.training(year=103, classes=classes, train_subject=i) # 時間相關因素實驗

        year_prediction = []  # 科目每一年的預測list
        final_prediction = 0  # 科目將所有年集合起來的預測
        year_loss = []

        for j in range(len(year)): # 預測的年份
            # print("=========== Training year {0} subject {1} ==============".format(str(year[j]), str(predicts[i])))

            x_feeds = tf.placeholder(tf.float32, [None, len(x_test[0])], name='x_input')  # testing data 中，只需要餵input就夠，output為使用training data 訓練得來的參數
            # #
            W = np.loadtxt(os.path.join("experiment/model/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
            b = np.loadtxt(os.path.join("experiment/model/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
            W2 = np.loadtxt(os.path.join("experiment/model2/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
            b2 = np.loadtxt(os.path.join("experiment/model2/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
            # W3 = np.loadtxt(os.path.join("experiment/model3/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
            # b3 = np.loadtxt(os.path.join("experiment/model3/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(np.float32)
            # W4 = np.loadtxt(os.path.join("experiment/model4/", "W" + str(year[j]) + "_" + str(predicts[i]))).astype(
            #     np.float32)
            # b4 = np.loadtxt(os.path.join("experiment/model4/", "b" + str(year[j]) + "_" + str(predicts[i]))).astype(
            #     np.float32)

            #
            l1 = onnm.add_layer(x_feeds, len(x_test[0]), layer_1_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W, b=b)  # pre_training
            l2 = onnm.add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W2, b=b2)  # pre_training
            # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W3, b=b3)  # pre_training
            # l4 = onnm.add_layer(l3, layer_3_hidden, layer_4_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W4, b=b4)  # with pre_training

            # prediction = onnm.add_layer(l3, layer_3_hidden, 1, keep_prob=1, activation_function=None)
            #
            # l1 = onnm.add_layer(x_feeds, len(x_test[0]), layer_1_hidden, keep_prob=1, activation_function=tf.nn.relu)  # without pre_training
            # l2 = onnm.add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob=1, activation_function=tf.nn.relu)  #without pre_training
            # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob, activation_function=tf.nn.relu)  # without pre_training

            # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob=1, activation_function=tf.nn.relu, W=W3,b=b3)  # withoutpre_training
            # prediction = onnm.add_layer(l1, layer_1_hidden, 1, keep_prob=1, activation_function=None)
            prediction = onnm.add_layer(l2, layer_2_hidden, 1, keep_prob=1, activation_function=None)

            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            # # Without Ensemble
            saver.restore(sess, './experiment/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(1) + '.ckpt')  # pre_training
            # saver.restore(sess, './my_net/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(val) + '.ckpt') # without pre_training
            # p, l = predicted(x_test, y_test)
            # mse = np.sqrt(np.mean((p - y_test) ** 2))
            # year_prediction.append(p)
            # year_loss.append(mse)

            # print(l, mse)
            # # Without Ensemble

            # val_prediction = 0
            # val_index=0
            # val_loss = []
            # for val in range(3): # 集合五個validation，乘上權重0.2
            #     saver.restore(sess, './experiment/year_' + str(year[j]) + '/net_' + str(predicts[i]) + "_" + str(val) + '.ckpt')  # pre_training
            #     p, l = predicted(x_test, y_test)
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

            y_pre = sess.run(prediction, feed_dict={x_feeds: x_test})
            print(y_pre)
            year_prediction.append(y_pre)

            tf.reset_default_graph() # 進行下一次的session前，要先清掉graph

            # 集合每一年的預測
        # order = sorted(year_loss, reverse=True)
        for k in range(len(year)):
        #     for m in [m for m, x in enumerate(year_loss) if x == order[k]]:
        #     #     print(order, year_loss, m)
        #         final_prediction += year_prediction[m] * sw[k]

            final_prediction += year_prediction[k] * sw[k]
        # result_mse = np.sqrt(np.mean((final_prediction - y_test) ** 2))
        print("=========== Subject {0} Final ==============".format(str(predicts[i])))
        print(final_prediction)
        # print(result_mse)
        # util.plotPredictScore(final_prediction, y_test)
        # util.timeAffectMeasure(final_prediction, y_test)
        # util.measure(final_prediction, y_test)

        df = pd.DataFrame(final_prediction)
        df.to_csv("./experiment/time_affect/score.csv")
        tf.reset_default_graph()  # 進行下一次的session前，要先清掉graph