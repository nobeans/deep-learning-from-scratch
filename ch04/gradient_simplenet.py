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

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = np.copy(init_x) # for avoiding side-effect

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # ガウス分布で初期化

    def predict(self, x):
        # 簡単のためにバイアスはしれっと省略してる？？
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = simpleNet()
print(net.W)

# 教師データ(given)
x = np.array([0.6, 0.9])  # 入力
t = np.array([0, 0, 1])   # 正解データ

# 現状のパラメータ(W)における回答を求める。
p = net.predict(x)
print(p)

print("----------------------------------------------------------------")

# 入力xに対するloss関数の値を求める。
# 入力xに対する正解データはtと仮定する。
print(net.loss(x, t))

print("----------------------------------------------------------------")

# 入力xに対するloss関数の勾配を求める。
# 入力xに対する正解データはtと仮定する。
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)

print(numerical_gradient(f, np.random.randn(2, 3)))
print(numerical_gradient(f, net.W))
