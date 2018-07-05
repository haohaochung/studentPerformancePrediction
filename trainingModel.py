# 訓練神經網路模型，並存起來
import pandas as pd
import numpy as np
import tensorflow as tf
import math

# --------------------- Tensorflow -------------------------- #
def add_layer(inputs, in_size, out_size, activation_function = None):

    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="w")
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="b")
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def normalize(data):
    normalized = (data-min(data)) / (max(data) - min(data))
    return normalized


training_data = pd.read_csv('studentData/studentInfo/100_cleanData.csv', encoding="big5", header=0)
training_data.dropna(inplace=True)
training_data = pd.get_dummies(training_data)

rowLength = len(training_data)
colLength = len(training_data.iloc[0])
print(training_data.iloc[0])
term1 = []
reg_no =[]
term2_calculus = []
term2_logical = []

for i in range(rowLength):
    term1.append([])

for i in range(rowLength):
    for j in range(1, 6):
        if training_data.iloc[i, j] < 0 :
            term1[i].append(0)
        else:
            term1[i].append(training_data.iloc[i,j])
    for s in range(23, colLength):
        term1[i].append(training_data.iloc[i, s])

    reg_no.append(training_data.iloc[i, 0])

    if training_data.iloc[i, 6] < 0:
        term2_calculus.append(0)
    else:
        term2_calculus.append(training_data.iloc[i, 6])
    term2_logical.append(training_data.iloc[i, 7])


term1 = np.array(term1).astype(float)
term2_calculus = np.array(term2_calculus).astype(float)
term2_logical = np.array(term2_logical).astype(float)
#
# for s in range(5):
#     term1[:, s] = normalize(term1[:, s])
# #
# term2_logical = normalize(term2_logical)
# term2_calculus = normalize(term2_calculus)

# --------------------- Tensorflow -------------------------- #
#
# # 建立 Feeds
# x_feeds = tf.placeholder(tf.float32, [None, len(term1[0])], name='x_input')
# y_feeds = tf.placeholder(tf.float32, [len(term1)], name='y_input')
#
# # add hidden layer
# l1 = add_layer(x_feeds,len(term1[0]),256, activation_function=tf.nn.relu)
# # l2 = add_layer(l1,256,128, activation_function=tf.nn.relu)
# # l3 = add_layer(l2,512,256, activation_function=tf.nn.relu)
#
#
# prediction = add_layer(l1, 256, len(term1), activation_function=None)
# # the error between prediction and real data
# loss = tf.sqrt(tf.reduce_mean(tf.square(y_feeds - prediction)))
# optimizer =  tf.train.AdamOptimizer(learning_rate=0.001)
# train_step = optimizer.minimize(loss)
# # losss = tf.sqrt(loss)
#
# #上面都是建立規則
# # 初始化所有變數
# # Can only save variables
# saver = tf.train.Saver()
# # Launch the graph
# with tf.Session() as sess:
#     # writer = tf.summary.FileWriter("./tensorBoard/", graph=sess.graph)
#     sess.run(tf.global_variables_initializer())
#     # 跑 for 迴圈 更新Weight200次 然後每訓練20次 印出一次Weight
#     for step in range(1000):
#         t, l, p = sess.run([train_step, loss, prediction], feed_dict={x_feeds: term1, y_feeds: term2_calculus})
#         # t2, l2, p2 = sess.run([train_step, loss, prediction], feed_dict={x_feeds: term1, y_feeds: term2_logical})
#
#         if step % 100 ==0:
#             print(sess.run(loss, feed_dict={x_feeds: term1, y_feeds: term2_calculus}))
#
#             # print( sess.run(prediction, feed_dict={x_feeds: term1}))
#             # print(l)
#             # print(l2)
#
#     print(p[259])
#     # saver.save(sess, './my_net/save_net.ckpt')
#
#     print(sess.run(y_feeds, feed_dict={y_feeds: term2_calculus}))
#     # writer.close()
# # print(reg_no)

# # --------------------------------- Regression ------------------------------------

W_calculus = np.zeros(len(term1[0])).astype(float)
W_logical = np.zeros(len(term1[0])).astype(float)

lr = 0.05
iteration = 100
batch_size = 200
step_num = int(math.floor(len(term1) / batch_size))
x_t = term1.transpose()
sum_gradient = np.zeros(len(term1[0]))
print(W_calculus.shape)
#
# for epoch in range(iteration):
#     # Train with batch
#     sum_gradient = np.zeros(len(term1[0]))
#
#     for i in range(step_num):
#         X = term1[i * batch_size:(i + 1) * batch_size]
#         Y = term2_calculus[i * batch_size:(i + 1) * batch_size]
#         hypothesis = np.dot(X, W_calculus)
#         loss = (hypothesis - Y)
#         # gradient_W = 2 * np.dot(x_t, (hypothesis - train_y))
#         # gradient_W = np.sum(2 * (hypothesis - term2_calculus))
#         cost = np.sum(loss ** 2) / len(X)
#         cost_a = math.sqrt(cost)
#         gradient_W = np.dot(X.transpose(), loss)
#         print(0, gradient_W)
#         sum_gradient += gradient_W ** 2
#         print(1, sum_gradient)
#
#         # W_calculus = W_calculus - lr*gradient_W/len(term1)
#         W_calculus = W_calculus - lr * gradient_W / np.sqrt(sum_gradient)
#         print(2, W_calculus)
#     # b = b - lr*gradient_b/len(train_x)
#     print("Iteration %d | Cost: %f" % (epoch, cost_a))

for epoch in range(iteration):
    hypothesis = np.dot(term1, W_calculus)
    loss = (hypothesis - term2_calculus)

    # gradient_W = 2 * np.dot(x_t, (hypothesis - train_y))
    # gradient_W = np.sum(2 * (hypothesis - term2_calculus))
    cost = np.sum(loss**2) / len(term1)
    cost_a = math.sqrt(cost)
    gradient_W = np.dot(x_t, loss)
    sum_gradient += gradient_W**2

    # W_calculus = W_calculus - lr*gradient_W/len(term1)
    W_calculus = W_calculus - lr*gradient_W/np.sqrt(sum_gradient)

    # b = b - lr*gradient_b/len(train_x)
    print("Iteration %d | Cost: %f" % (epoch, cost_a))
#
# for i in range(iteration):
#     hypothesis = np.dot(term1, W_logical)
#     loss = (hypothesis - term2_logical) ** 2
#
#     # gradient_W = 2 * np.dot(x_t, (hypothesis - train_y))
#     gradient_W = np.sum(2 * (hypothesis - term2_logical))
#     cost = np.sum(loss) / len(term1)
#     cost_a = math.sqrt(cost)
#     W_logical = W_logical - lr*gradient_W/len(term1)
#     # b = b - lr*gradient_b/len(train_x)
#     print("2. Iteration %d | Cost: %f" % (i, cost_a))

print(W_calculus)
print(hypothesis)
