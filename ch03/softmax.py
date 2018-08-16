import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # for avoiding an overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))
print(np.sum(softmax(a)))

a = np.array([1010, 1000, 990])
print(softmax(a))
print(np.sum(softmax(a)))
