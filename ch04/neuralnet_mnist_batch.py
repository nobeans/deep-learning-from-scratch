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

def get_train_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=True)
    return x_train, t_train

def get_test_data():
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

def mean_squared_error(y, t):
    if y.ndim == 1:
        # 1次元配列から1xNの2次元の行列に変換する
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    return 0.5 * np.sum((y - t) ** 2) / batch_size

def cross_entropy_error(y, t):
    if y.ndim == 1:
        # 1次元配列から1xNの2次元の行列に変換する
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    delta = 1e-7 # for avoiding -inf when y nearly equals 0
    return -np.sum(t * np.log(y + delta)) / batch_size

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001 (誤差の発生しづらい十分小さい数)
    grad = np.zeros_like(x) # xと同じ形状の配列を生成する

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 元に戻す

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


#
# Main
#

x_train, t_train = get_train_data()
network = init_network()

# ミニバッチを選出する
train_size = t_train.shape[0]
batch_size = 100
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

y = predict(network, x_batch)

# 2乗和誤差法
e1 = mean_squared_error(y, t_batch)
print(e1)

# 交差エントロピー誤差
e2 = cross_entropy_error(y, t_batch)
print(e2)
