from nipy.io.api import save_image
from nipy.core.api import Image, AffineTransform
from setting import LABEL_PATH,IMAGE_PATH,INFO,OUTPUT
import torch
from torch.utils.data import Dataset
import numpy as np
import json

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

def get_total_segment(y_pred,filename):
    y_pred = y_pred.argmax(axis=0)     # (x,y,z)
    y_pred = np.expand_dims(y_pred,axis=3)  #(x,y,z,1)

    with open(INFO+'shape.json','w') as f:
        shape_info = json.loads(f.read())
    if shape_info[filename] == '160':
        segment = np.zeros((192,192,160,1))
        segment[60:140, 60:152, 30:130] = y_pred
    elif shape_info[filename] == '166':
        segment = np.zeros((256,256, 166, 1))
        segment[90:170, 95:187, 30:130] = y_pred
    elif shape_info[filename] == '180':
        segment = np.zeros((256,256,180, 1))
        segment[100:180, 100:192, 40:140] = y_pred
    else:
        raise ValueError('shape error')
    cmap = AffineTransform('kji', 'zxy', np.eye(4))
    img = Image(y_pred, cmap)
    save_image(img,OUTPUT+filename)





















