# 2.3.1
#def AND(x1, x2):
#    w1, w2, theta = 0.5, 0.5, 0.7
#    tmp = x1*w1 + x2*w2
#    if tmp <= theta:
#        return 0
#    if tmp > theta:
#        return 1

# 2.3.2
import numpy as np
#x = np.array([0, 1])
#w = np.array([0.5, 0.5])
#b = -0.7
#print(w*x)
#print(np.sum(w*x))
#print(np.sum(w*x) + b)
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    if tmp > 0:
        return 1

# Shared verification
assert AND(0, 0) == 0
assert AND(1, 0) == 0
assert AND(0, 1) == 0
assert AND(1, 1) == 1
