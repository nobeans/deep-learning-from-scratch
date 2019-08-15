import matplotlib.pyplot as plt
from simple_conv_net import *
from optimizers import *

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# ハイパーパラメータ(色々調整する)
iters_num = 10000
#iters_num = 500
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
acc_sample_size = 1000

network = SimpleConvNet(
    input_dim=(1, 28, 28),
    conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
    hidden_size=100,
    output_size=10,
    weight_init_std=0.01)

#optimizer = SGD(lr = 0.95)
#optimizer = Momentum(lr = 0.1)
#optimizer = AdaGrad(lr = 1.5)
optimizer = Adam(lr = 0.01)

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
    grads = network.gradient(x_batch, t_batch)

    # パラメータの更新
    optimizer.update(network.params, grads)

    # 学習経過の記録
    train_loss_list.append(network.loss(x_batch, t_batch))

    if i % iter_per_epoch == 0:
        print("epoc:", int(i / iter_per_epoch) + 1, i)

    if i % batch_size == 0:
        print("batch:", int(i / batch_size) + 1, i)
        train_sample_mask = np.random.choice(train_size, acc_sample_size)
        acc_train = network.accuracy(x_train[train_sample_mask], t_train[train_sample_mask], batch_size) # 全訓練データに対する認識精度
        test_sample_mask = np.random.choice(test_size, acc_sample_size)
        acc_test = network.accuracy(x_test[test_sample_mask], t_test[test_sample_mask], batch_size)    # 全テストデータに対する認識精度
        print("acc:", acc_train, acc_test)
        train_acc_list.append(acc_train) # 全訓練データに対する認識精度
        test_acc_list.append(acc_test)    # 全テストデータに対する認識精度


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

