import numpy as np
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.1
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    if tmp > 0:
        return 1

# Shared verification
assert OR(0, 0) == 0
assert OR(1, 0) == 1
assert OR(0, 1) == 1
assert OR(1, 1) == 1
