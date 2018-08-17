import numpy as np
from PIL import Image
import pickle

# pip3 install pillow
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    # バッチ処理に対応するため、行列もサポートする必要がある。
    if a.ndim == 2:
        # 2次元の行列の場合は転置行列を使っていい感じに計算する。
        a_T = a.T
        c = np.max(a_T, axis=0)
        exp_a_T = np.exp(a_T - c) # for avoiding an overflow
        sum_exp_a_T = np.sum(exp_a_T, axis=0)
        y = exp_a_T / sum_exp_a_T
        return y.T
    c = np.max(a)
    exp_a = np.exp(a - c) # for avoiding an overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    # あらかじめ学習済みのパラメータデータをロードする。
    # 出力層は10で、それぞれ数字の0, 1, 2, 3, ..., 9の確率に対応する。
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    #y = a3  # argmaxで評価するだけの分類問題では活性化関数を使わなくても値の大小は変わらないため、これでも良い

    return y


#
# Main
#

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
