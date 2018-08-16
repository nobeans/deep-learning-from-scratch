print("----------------------------------------------------------------")
print("# 1.3.1")
print("")
print(1 - 2)
print(4 * 5)
print(7 / 5)
print(3 ** 2)


print("----------------------------------------------------------------")
print("# 1.3.2")
print("")
print(type(10))
print(type(2.718))
print(type("hello"))

print("----------------------------------------------------------------")
print("# 1.3.3")
print("")
x = 10
print(x)
x = 100
print(x)
y = 3.14
print(x * y)
print(type(x * y))

print("----------------------------------------------------------------")
print("# 1.3.4")
print("")
a = [1, 2, 3, 4, 5]
print(a)
print(len(a))
print(a[0])
print(a[4])
a[4] = 99
print(a)
print("")
print(a[0:2])
print(a[1:])
print(a[:3])
print(a[:-1])
print(a[:-2])

print("----------------------------------------------------------------")
print("# 1.3.5")
print("")
me = {'height':180}
print(me['height'])
me['weight'] = 70
print(me)

print("----------------------------------------------------------------")
print("# 1.3.6")
print("")
hungry = True
sleepy = False
print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

print("----------------------------------------------------------------")
print("# 1.3.7")
print("")
if hungry:
    print("I'm hungry.")
hungry = False
if hungry:
    print("I'm hungry.")
else:
    print("I'm not hungry.")
    if sleepy:
        print("I'm sleepy.")
    else:
        print("I'm not sleepy.")

print("----------------------------------------------------------------")
print("# 1.3.8")
print("")
for i in [1, 2, 3]:
    print(i)

print("----------------------------------------------------------------")
print("# 1.3.9")
print("")
def hello():
    print("Hello World!")
hello()

def hello(object):
    print("Hello " + object + "!")
hello("cat")

print("----------------------------------------------------------------")
print("# 1.4.2")
print("")
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")
    def hello(self):
        print("Hello " + self.name + "!")
    def goodbye(self):
        print("Good-bye " + self.name + "!")
m = Man("David")
m.hello()
m.goodbye()

print("----------------------------------------------------------------")
print("# 1.5.1")
print("")
import numpy as np
# You must install numpy like this:
# > pip3 install number

print("----------------------------------------------------------------")
print("# 1.5.2")
print("")
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

print("----------------------------------------------------------------")
print("# 1.5.3")
print("")
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x / 2.0)

print("----------------------------------------------------------------")
print("# 1.5.4")
print("")
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)
B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
print(A * 10)

print("----------------------------------------------------------------")
print("# 1.5.5")
print("")
B = np.array([10, 20])
print(A * B)

print("----------------------------------------------------------------")
print("# 1.5.6")
print("")
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])
for now in X:
    print(now)
X = X.flatten()
print(X)
print(np.array([0, 2, 4]))
print(X[np.array([0, 2, 4])])
print(X > 15)
print(X[X > 15])

print("----------------------------------------------------------------")
print("# 1.6.1")
print("")
import matplotlib.pyplot as plt
x = np.arange(0, 6, 0.1)
y = np.sin(x)
print(x)
print(y)
plt.plot(x, y)
plt.show()

print("----------------------------------------------------------------")
print("# 1.6.2")
print("")
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

print("----------------------------------------------------------------")
print("# 1.6.3")
print("")
from matplotlib.image import imread
img = imread('lena.png')
plt.imshow(img)
plt.show()
