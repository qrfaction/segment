from nipy import load_image
from setting import LABEL_PATH,IMAGE_PATH
import torch
from torch.utils.data import Dataset
import numpy as np


def get_files(path,prefix = True):
    import os
    files = os.listdir(path)   #一对label和image文件同名
    if prefix:
        files = [path + f for f in files]
    else:
        files = [f for f in files]
    return np.array(files)

def get_image(f):
    image = np.load(f).tolist()
    return torch.FloatTensor(image)

def get_batch_images(files):
    batch_images = []
    for f in files:
        batch_images.append(np.load(f).tolist())
    return torch.FloatTensor(batch_images)

class Generator(Dataset):
    def __init__(self,files):
        self.images = np.array([IMAGE_PATH + f for f in files])
        self.labels = np.array([LABEL_PATH + f for f in files])
        self.length = len(files)

    def __getitem__(self, index):  # 返回的是tensor
        images = get_image(self.images[index])
        labels = get_image(self.labels[index])
        return images,labels

    def __len__(self):
        return self.length