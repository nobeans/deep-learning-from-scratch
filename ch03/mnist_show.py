import sys, os
sys.path.append(os.pardir)

# pip3 install pillow
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

index = 2

img = x_train[index]
label = t_train[index]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
