from nipy import load_image
from setting import LABEL_PATH,IMAGE_PATH
import torch
from torch.utils.data import Dataset
import numpy as np
import json

def get_files():
    import os
    files = os.listdir(LABEL_PATH)   #一对label和image文件同名
    return files

def get_images(files,volatile,islabel):
    result = []
    if islabel:
        for f in files:
            image = load_image(f).get_data().transpose((3, 0, 1, 2))  # 轴对换
            image[1,:,:,:] = np.float64(image[0,:,:,:]==1)
            image[2,:,:,:] = np.float64(image[0,:,:,:]==2)
            image[0,:,:,:] = np.float64(image[0,:,:,:]==0)
            result.append(image)
    else:
        for f in files:
            image = load_image(f).get_data().transpose((3,0,1,2))  #轴对换
            result.append(image)
    return torch.autograd.Variable(torch.FloatTensor(np.array(result).tolist()),volatile=volatile)

class Generator(Dataset):
    def __init__(self,files):
        self.images = np.array([IMAGE_PATH + f for f in files])
        self.labels = np.array([LABEL_PATH + f for f in files])
        # with open('effec_range.json','r') as f:
        #     self.effect_range = json.loads(f.read())
        self.length = len(files)
            # torch.FloatTensor(labels.tolist())

    def __getitem__(self, index):  # 返回的是tensor
        images = get_images(self.images[index],volatile=False,islabel=False)
        labels = get_images(self.labels[index],volatile=False,islabel=True)
        return images,labels

    def __len__(self):
        return self.length