import numpy as np

# 行列計算で書き直してみる

def AND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([[0.5], [0.5]])
    B = 1 * -0.7
    Y = np.dot(X, W) + B
    return 0 if Y <= 0 else 1

def NAND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([[-0.5], [-0.5]])
    B = 1 * 0.7
    Y = np.dot(X, W) + B
    return 0 if Y <= 0 else 1

def OR(x1, x2):
    X = np.array([x1, x2])
    W = np.array([[0.5], [0.5]])
    B = 1 * -0.1
    Y = np.dot(X, W) + B
    return 0 if Y <= 0 else 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

assert XOR(0, 0) == 0
assert XOR(1, 0) == 1
assert XOR(0, 1) == 1
assert XOR(1, 1) == 0
