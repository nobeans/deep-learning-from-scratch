import numpy as np
from collections import OrderedDict

import sys, os
sys.path.append(os.pardir)

from numerical_gradient import numerical_gradient
from layers import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの初期化
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
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
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
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

## 入力xに対するloss関数の勾配を求める。
## 入力xに対する正解データはtと仮定する。
#grads = network.numerical_gradient(x, t)
#print(grads['W1'].shape)
#print(grads['b1'].shape)
#print(grads['W2'].shape)
#print(grads['b2'].shape)

