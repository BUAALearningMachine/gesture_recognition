from __future__ import print_function
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

from GestureDataset import GestureDataset

train_dataset = GestureDataset(root=osp.abspath('./gesture_file'), train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

img = train_dataset.get_img(1000)
img.show()
print(train_dataset.get_anno(1000))
