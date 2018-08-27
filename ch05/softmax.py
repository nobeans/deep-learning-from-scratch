import numpy as np

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

#a = np.array([0.3, 2.9, 4.0])
#print(softmax(a))
#print(np.sum(softmax(a)))
#
#a = np.array([1010, 1000, 990])
#print(softmax(a))
#print(np.sum(softmax(a)))
