import tensorflow as tf
import numpy as np
import pandas as pd
import timeit
import os
import util
from OfflineNeuralNetworkModel import OfflineNeuralNetworkModel
from autoEncoder import autoEncoder # from 檔名 import class
from sklearn.model_selection  import KFold, train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def early_stopping(sess, x_val, y_val):
    y_val_pre = (sess.run(prediction, feed_dict={x_feeds: x_val, keep_prob: 1}))
    valLoss = tf.sqrt(tf.reduce_mean(tf.square(y_val - y_val_pre)))
    return valLoss

def run_train(sess, x_train, y_train, x_val, y_val, iteration, kf_count, best_validation_loss):
    train_loss =[]
    val_loss=[]

    # writer = tf.summary.FileWriter("./tensorBoard/", graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    # 跑 for 迴圈 更新Weight200次 然後每訓練20次 印出一次Weight
    for step in range(iteration):
        t, l, p = sess.run([train_step, loss, prediction], feed_dict={x_feeds: x_train, y_feeds: y_train, keep_prob: 0.5})

        if step % 100 == 0:
            trainLoss = (sess.run(loss, feed_dict={x_feeds: x_train, y_feeds: y_train, keep_prob: 0.5}))
            valLoss = early_stopping(sess, x_val, y_val)
            train_loss.append(trainLoss)
            val_loss.append(sess.run(valLoss))

            print("Training loss {0:.4f}, Validation loss {1:.4f}".format(trainLoss, sess.run(valLoss)))

            if sess.run(valLoss) < best_validation_loss:
                best_validation_loss = sess.run(valLoss)
                last_improvement = step
                # saver.save(sess, './my_net/year_' + str(year[i]) + '/net_' + str(predict[j])+"_"+ str(kf_count) + '.ckpt')
                saver.save(sess, './my_net_preTraining/year_' + str(year[i]) + '/net_' + str(predict[j])+"_"+ str(kf_count) + '.ckpt')

        if step - last_improvement > 500:
            print("Last step is {0}".format(last_improvement))
            break
    print("Finish all step")

    return train_loss, val_loss

def cross_validate(session, iteration, year, course, split_size=2):
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

def saveModel(W, save_dir, name): # 存權重模型
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.savetxt(os.path.join(save_dir, name), W)


start = timeit.default_timer()

onnm = OfflineNeuralNetworkModel()
train, predict = onnm.trainingParameter()
l = onnm.trainingYearAndSemester(0, 102,online=False)
tc = util.flattenList(l)
# year = ["99"]
year = ["99","100","101","102","103","104"]
iteration = 5000
layer_1_hidden = 256
layer_2_hidden = 128
layer_3_hidden = 64
print(tc)
for i in range(len(year)):
    for j in [0]:
        print("===========Training year {0} subject {1} ==============".format(str(year[i]), str(predict[j])))
        x_train, y_train = onnm.loadData("studentData/studentInfo/" + str(year[i]) + "_cleanData.csv", train[j], predict[j])
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        x_feeds = tf.placeholder(tf.float32, [None, len(x_train[0])], name='x_input')
        y_feeds = tf.placeholder(tf.float32, [None, 1], name='y_input')

        print("Pre-training hidden layer 1...............................")
        da = autoEncoder(layer_name="layer_1", input=x_train, n_visible=len(x_train[0]), n_hidden=layer_1_hidden)
        tPreTrainingStart = timeit.default_timer()
        l1w, l1b, l1o = da.train()
        saveModel(l1w, "model/", "W"+str(year[i])+"_"+str(predict[j]))
        saveModel(l1b, "model/", "b"+str(year[i])+"_"+str(predict[j]))
        tPreTrainingEnd = timeit.default_timer()
        print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))

        print("Pre-training hidden layer 2...............................")
        da2 = autoEncoder(layer_name="layer_2", input=l1o, n_visible=layer_1_hidden, n_hidden=layer_2_hidden)
        tPreTrainingStart = timeit.default_timer()
        l2w, l2b, l2o = da2.train()
        saveModel(l2w, "model2/", "W" + str(year[i]) + "_" + str(predict[j]))
        saveModel(l2b, "model2/", "b" + str(year[i]) + "_" + str(predict[j]))
        tPreTrainingEnd = timeit.default_timer()
        print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))
        #
        # print("Pre-training hidden layer 3...............................")
        # da3 = autoEncoder(layer_name="layer_3", input=l2o, n_visible=layer_2_hidden, n_hidden=layer_3_hidden)
        # tPreTrainingStart = timeit.default_timer()
        # l3w, l3b, l3o = da3.train()
        # saveModel(l3w, "model3/", "W" + str(year[i]) + "_" + str(predict[j]))
        # saveModel(l3b, "model3/", "b" + str(year[i]) + "_" + str(predict[j]))
        # tPreTrainingEnd = timeit.default_timer()
        # print("Finish, it cost %f sec" % (tPreTrainingEnd - tPreTrainingStart))


        # # add hidden layer
        l1 = onnm.add_layer(x_feeds, len(x_train[0]), layer_1_hidden, keep_prob, activation_function=tf.nn.relu, W=l1w, b=l1b) # with pre_training
        l2 = onnm.add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob, activation_function=tf.nn.relu, W=l2w, b=l2b) # with pre_training
        # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob, activation_function=tf.nn.relu, W=l3w, b=l3b) # with pre_training

        prediction = onnm.add_layer(l2, layer_2_hidden, 1, keep_prob, activation_function=None)

        #
        # prediction = onnm.add_layer(l2, layer_2_hidden, 1, keep_prob, activation_function=None)  #without pre_training
        # l1 = onnm.add_layer(x_feeds, len(x_train[0]), layer_1_hidden, keep_prob, activation_function=tf.nn.relu) # without pre_training
        # l2 = onnm.add_layer(l1, layer_1_hidden, layer_2_hidden, keep_prob, activation_function=tf.nn.relu) # without pre_training
        # # l3 = onnm.add_layer(l2, layer_2_hidden, layer_3_hidden, keep_prob, activation_function=tf.nn.relu) # without pre_training


        # the error between prediction and real data
        loss = tf.sqrt(tf.reduce_mean(tf.square(y_feeds - prediction))) + onnm.regularization * 0.01

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        train_step = optimizer.minimize(loss)

        # 上面都是建立規則
        # 初始化所有變數
        # Can only save variables

        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:

            # Without Ensemble
            train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size=0.2)
            print(train_x.shape)
            train_loss, val_loss = run_train(sess, train_x, train_y, val_x, val_y, iteration, 1, best_validation_loss=1000000)
            loss_output = pd.DataFrame()
            loss_output["train_loss" + str(1)] = train_loss
            loss_output["val_loss" + str(1)] = val_loss
            loss_output.to_csv("./output/noEnsemble/loss_" + str(year[i]) + "_" + str(predict[j])  + ".csv")
            # loss_output.to_csv("./output/noKmeans/loss_" + str(year[i]) + "_" + str(predict[j]) + "_" + str(1) + ".csv") #kmeans

            # With Ensemble
            # results = cross_validate(sess, iteration, str(year[i]), str(predict[j]))
            # print("Cross-validation result: %s" % results)

        tf.reset_default_graph()  # 進行下一次的session前，要先清掉graph
stop = timeit.default_timer()
print("time:", stop - start)