import numpy as np
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    if tmp > 0:
        return 1

# Shared verification
assert NAND(0, 0) == 1
assert NAND(1, 0) == 1
assert NAND(0, 1) == 1
assert NAND(1, 1) == 0
