import numpy as np
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from softmax import softmax
from cross_entropy_error import cross_entropy_error

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

#print("Relu:")
#x = np.array([[1.0, -0.5], [-2.0, 3.0]])
#print("x:\n", x)
#relu = Relu()
#print("forward:\n", relu.forward(x))
#dout = np.array([[1, 1], [1, 1]])
#print("backward:\n", relu.backward(dout))
#print()


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx

#print("Sigmoid:")
#x = np.array([[1.0, -0.5], [-2.0, 3.0]])
#print("x:\n", x)
#sigmoid = Sigmoid()
#print("forward:\n", sigmoid.forward(x))
#dout = np.array([[1, 1], [1, 1]])
#print("backward:\n", sigmoid.backward(dout))
#print()


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

#print("Affine:")
#X = np.random.rand(2)
#W = np.random.rand(2, 3)
#B = np.random.rand(3)
#print("X:\n", X)
#print("W:\n", W)
#print("B:\n", B)
#affine = Affine(W, B)
#print("forward:\n", affine.forward(X))
#dout = np.random.rand(2, 3)
#print("backward:\n", affine.backward(dout))
#print()


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

#print("SoftmaxWithLoss:")
#X = np.random.rand(2)
#T = np.array([0, 1])
#print("X:\n", X)
#print("T:\n", T)
#swl = SoftmaxWithLoss()
#print("forward:", swl.forward(X, T))
#dout = 1
#print("backward:", swl.backward(dout))

