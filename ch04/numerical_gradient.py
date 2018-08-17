import numpy as np

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001 (誤差の発生しづらい十分小さい数)
    grad = np.zeros_like(x) # xと同じ形状の配列を生成する

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 元に戻す

    return grad

def function_2(x):
    return np.sum(x**2)

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([1.0, 1.0])))
print(numerical_gradient(function_2, np.array([0.5, 0.5])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
