from nipy import load_image
from setting import LABEL_PATH,IMAGE_PATH
import torch
from torch.utils.data import Dataset
import numpy as np

def get_files():
    import os
    files = os.listdir(LABEL_PATH)   #一对label和image文件同名
    return files

def get_images(files,volatile):
    result = []
    for f in files:
        image = load_image(f).transpose((3,0,1,2))  #轴兑换
        result.append(image)
    return torch.autograd.Variable(torch.FloatTensor(np.array(result).tolist()),volatile=volatile)

class Generator(Dataset):
    def __init__(self,files):
        self.images = np.array([IMAGE_PATH + f for f in files])
        self.labels = np.array([LABEL_PATH + f for f in files])
        self.length = len(files)
            # torch.FloatTensor(labels.tolist())

    def __getitem__(self, index):  # 返回的是tensor
        images = get_images(self.images[index],volatile=False)
        labels = -get_images(self.labels[index],volatile=False)   # 0  -1
        return images,labels

    def __len__(self):
        return self.length