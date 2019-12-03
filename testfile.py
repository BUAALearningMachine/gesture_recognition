import os
import os.path as osp
from PIL import Image


def prepare_data(dir):
    data = []
    targets = []
    for root_temp, dirs, files in os.walk(dir, topdown=True):
        for name in files:
            temp = osp.join(root_temp, name)
            img = Image.open(temp)
            img = img.convert('L')
            data.append(img)
            targets.append(root_temp[-1:])


prepare_data(osp.join(osp.abspath('./gesture_file'), 'train'))
