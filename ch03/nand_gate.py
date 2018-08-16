import numpy as np

# 行列計算で書き直してみる

def NAND(x1, x2):
    X = np.array([x1, x2])
    W = np.array([[-0.5], [-0.5]])
    B = 1 * 0.7
    Y = np.dot(X, W) + B
    return 0 if Y <= 0 else 1

# Shared verification
assert NAND(0, 0) == 1
assert NAND(1, 0) == 1
assert NAND(0, 1) == 1
assert NAND(1, 1) == 0
