# coding: UTF-8
# 2層のニューラルネットワーク

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# ファイルからndarrayを取得
def open_with_numpy_loadtxt(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

x = tf.placeholder("float",[None, 4]) # 入力層
y_ = tf.placeholder("float",[None, 6]) # 出力層

# Tensorのsize mean: 平均 stdev: 標準偏差
w_h = tf.Variable(tf.random_normal([4, 10], mean=0.0, stddev=0.05))
w_o = tf.Variable(tf.random_normal([10, 6], mean=0.0, stddev=0.05))
b_h = tf.Variable(tf.zeros([10]))
b_o = tf.Variable(tf.zeros([6]))

# model
def model(X, w_h, b_h, w_o, b_o):
    # matmulは行列の掛け算をしてくれる
    # sigmoidは活性化関数
    h = tf.sigmoid(tf.matmul(X, w_h) + b_h)
    # Reluで活性化したい場合
    # h = tf.nn.relu(tf.matmul(x, w_h) + b_h)
    
    # softmax関数は与えたデータから確率っぽいものを求めてくれます
    # softmax関数も活性化関数です
    pyx = tf.nn.softmax(tf.matmul(h, w_o) + b_o)

    return pyx

y_hypo = model(x, w_h, b_h, w_o, b_o)

# 誤差関数(損失関数)
# 二乗誤差関数と交差エントロピー誤差関数があり前者は数値予測に後者はクラス分類に使う

# 交差エントロピー誤差関数
cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_hypo,1e-10,1.0)))
# 二乗誤差関数
# loss = tf.reduce_mean(tf.square(y - y_))

# 正則化(Regulaization)
L2_sqr = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
lambda_2 = 0.01

# loss
loss = cross_entropy + lambda_2 * L2_sqr
# 勾配降下法 誤差関数と学習率を指定
# 0.0001
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

data = open_with_numpy_loadtxt("RGBYB.csv")
labels = open_with_numpy_loadtxt("labelsRGBYB.csv")

testData = tf.convert_to_tensor(open_with_numpy_loadtxt("testRGBYB.csv"))
testLabel = tf.convert_to_tensor(open_with_numpy_loadtxt("testLabelsRGBYB.csv"))


saver = tf.train.Saver()

"""
ここから    dataとlabelをshuffe
"""
data_range = range(len(data))
range_list = list(data_range)
random.shuffle(range_list)

r_train_list = []
r_train_labels = []

for t in data_range:
    listNum = range_list[t]
    r_train_list.append(list(data[listNum]))
    r_train_labels.append(list(labels[listNum]))


data = r_train_list
labels = r_train_labels
data = tf.convert_to_tensor(data)
labels = tf.convert_to_tensor(labels)

# 重さやバイアスを保存してくれる
saver = tf.train.Saver()

with tf.Session() as sess:
    # 初期化
    sess.run(tf.global_variables_initializer())
    batch_xs = data.eval()
    batch_ys = labels.eval()

    # 分類結果の判定
    correct_prediction = tf.equal(tf.argmax(y_hypo,1), tf.argmax(y_,1))
    # 精度の計算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # セーブした変数を復元
    # saver.restore(sess, "/tmp/model.ckpt")
    # print("Model restored.")

    for i in range(400):
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i % 100 == 0:
            train_accuracy = accuracy.eval({x: batch_xs, y_: batch_ys})
            print('  step, accuracy = %6d: %6.3f' % (i, train_accuracy))

    print("学習結果")
    print("test dataでの精度")
    print(accuracy.eval({x: testData.eval(), y_: testLabel.eval()}) * 100)

    # 変数データを保存する
    # save_path = saver.save(sess, "./model.ckpt")
    # print("Model saved in file: %s" % save_path)

    testData = tf.cast(testData.eval(), tf.float32)
    y_hypo = model(testData.eval(), w_h, b_h, w_o, b_o)
    y_hypo = tf.cast(y_hypo.eval(), tf.float64)

    result = tf.nn.softmax(y_hypo.eval())
    print("result:")
    # 一番大きい値のindexがその判定した色の番号
    print(result[0:10].eval())

