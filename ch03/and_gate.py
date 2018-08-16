import numpy as np

# 行列計算で書き直してみる

def AND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([[0.5], [0.5]])
    B = 1 * -0.7
    Y = np.dot(X, W) + B
    return 0 if Y <= 0 else 1

# Shared verification
assert AND(0, 0) == 0
assert AND(1, 0) == 0
assert AND(0, 1) == 0
assert AND(1, 1) == 1
