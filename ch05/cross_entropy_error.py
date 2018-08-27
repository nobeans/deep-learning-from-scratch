import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        # 1次元配列から1xNの2次元の行列に変換する
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    delta = 1e-7 # for avoiding -inf when y nearly equals 0
    return -np.sum(t * np.log(y + delta)) / batch_size
