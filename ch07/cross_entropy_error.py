import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        # 1次元配列から1xNの2次元の行列に変換する
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
