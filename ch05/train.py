import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from two_layer_net import *


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

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
