import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import os.path as osp


class GestureDataset(data.Dataset):
    training_folder = 'train'
    test_folder = 'test'

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        # 6种手势
        self.classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five']

        if self.train:
            data_folder = osp.join(root, self.training_folder)
        else:
            data_folder = osp.join(root, self.test_folder)
        self.data, self.targets = prepare_data(data_folder)

    def get_img(self, index):
        return self.data[index]

    def get_anno(self, index):
        return self.classes[int(self.targets[index])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def prepare_data(root):
    data = []
    targets = []
    for root_temp, dirs, files in os.walk(root, topdown=True):
        for name in files:
            temp = osp.join(root_temp, name)
            img = Image.open(temp)
            # 转成灰度图片
            img = img.convert('L')
            data.append(img)
            targets.append(root_temp[-1:])
    return data, targets
