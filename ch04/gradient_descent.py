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

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = np.copy(init_x) # for avoiding side-effect

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function_2(x):
    return np.sum(x**2)

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, 0.1, 100))
print(gradient_descent(function_2, init_x, 10.0, 100))
print(gradient_descent(function_2, init_x, 1e-10, 100))
