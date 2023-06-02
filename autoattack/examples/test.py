import random
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
import cv2
import matplotlib.pyplot as plt
'''
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
])

train_dataset = ImageFolder("C:\\Users\\ForYou\\Desktop\\auto-attack-master\\data\\test\\airplane\\airbus_s_000009.png",
                                transform=transform_train)

print(train_dataset)
'''
image = cv2.imread("C:\\Users\\ForYou\\Desktop\\auto-attack-master\\data\\test\\airplane\\airbus_s_000009.png")
plt.imshow(image)
# plt.show()
'''
print(image)
print("")
print(image[0], image[0].shape)
print("")
print(image[0][0], image[0][0].shape)
print("")
print(image[:, :, 0], image[:, :, 0].shape)
print(image[:, :, 1], image[:, :, 1].shape)
print(image[:, :, 2], image[:, :, 2].shape)
'''
print(image[0, :, 2], image.shape, "1")
print(image[:, :, 2])
# image[0, :, 2] += random.uniform(-8/255, 8/255)
a = random.uniform(-8/255, 8/255)
a = round(a, 4)
print(a)

image[0, :, 2] += 50
print(image[0, :, 2], image[:, :, 2].shape, "2")
print(image[:, :, 2])
# cv2.imshow("newimage", image)
plt.imshow(image)
# plt.show()
print(time.time_ns())

import torch, gc
gc.collect()
torch.cuda.empty_cache()