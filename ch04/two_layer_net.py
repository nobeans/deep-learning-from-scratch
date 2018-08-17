import numpy as np

# pip3 install pillow
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

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

def cross_entropy_error(y, t):
    if y.ndim == 1:
        # 1次元配列から1xNの2次元の行列に変換する
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    delta = 1e-7 # for avoiding -inf when y nearly equals 0
    return -np.sum(t * np.log(y + delta)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001 (誤差の発生しづらい十分小さい数)
    grad = np.zeros_like(x) # xと同じ形状の配列を生成する

    # 一般化されたNxM行列に対応するため、イテレータを使う。
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 元に戻す

        it.iternext()

    return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t_one_hot):
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t_one_hot, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, W1)
        grads['b1'] = numerical_gradient(loss_W, b1)
        grads['W2'] = numerical_gradient(loss_W, W2)
        grads['b2'] = numerical_gradient(loss_W, b2)
        return grads

    # 誤差逆伝播方
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        batch_num = x.shape[0]
        dy = (y - t) / batch_num
        grads = {}
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


#network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#print(network.params['W1'].shape)
#print(network.params['b1'].shape)
#print(network.params['W2'].shape)
#print(network.params['b2'].shape)
#
## 教師データ(given)
#x = np.random.rand(100, 784)  # 入力
#t = np.random.rand(100, 10)   # 正解データ
#
## 現状のパラメータ(W)における回答を求める。
#y = network.predict(x)
#print(y)
#
#print("----------------------------------------------------------------")
#
## 入力xに対するloss関数の値を求める。
## 入力xに対する正解データはtと仮定する。
#print(network.loss(x, t))
#
#print("----------------------------------------------------------------")

# 入力xに対するloss関数の勾配を求める。
# 入力xに対する正解データはtと仮定する。
#grads = network.numerical_gradient(x, t)
#print(grads['W1'].shape)
#print(grads['b1'].shape)
#print(grads['W2'].shape)
#print(grads['b2'].shape)

print("----------------------------------------------------------------")

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=True, one_hot_label=True)

# ハイパーパラメータ(色々調整する)
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    train_loss_list.append(network.loss(x_batch, t_batch))

    if i % iter_per_epoch == 0:
        train_acc_list.append(network.accuracy(x_train, t_train)) # 全訓練データに対する認識精度
        test_acc_list.append(network.accuracy(x_test, t_test))    # 全テストデータに対する認識精度

# loss関数の値のグラフの描画
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(0, iters_num)
plt.ylim(0, np.max(train_loss_list))
plt.legend(loc='upper right')
plt.show()

# 認識精度グラフの描画
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
