from nipy.io.api import save_image,load_image
from nipy.core.api import Image, AffineTransform
from setting import LABEL_PATH,IMAGE_PATH,INFO,OUTPUT,PRE_LABEL_PATH
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from model import dice_metric
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
    # print(y_pred)
    # print(y_pred.shape)
    label = load_image(PRE_LABEL_PATH + filename)
    # L = np.transpose(L,(3,0,1,2))

    # val_labels = LABEL_PATH + filename[:-7] + '.npy'
    # ll = np.load(val_labels)
    y_pred = y_pred.argmax(axis=0)

    # x1 = np.sum(label[:, :, :] == 1)
    # x2 = np.sum(label[:, :, :] == 2)
    # print('label',x1+x2,x1,x2)

    # y_pred = y_pred.argmax(axis=0)     # (x,y,z)
    #
    # y_pred = np.expand_dims(y_pred,axis=3)  #(x,y,z,1)
    # y1 = np.sum(y_pred[:, :, :] == 1)
    # y2 = np.sum(y_pred[:, :, :] == 2)
    # print('y_pred', y1 + y2,y1,y2)

    # x1 = label>0
    # x2 = y_pred>0


    with open(INFO+'shape.json','r') as f:
        shape_info = json.loads(f.read())
    if shape_info[filename] == '160':
        # print(np.sum(x1[60:140, 60:152, 30:130] * x2)/(x1.sum()+x2.sum()))
        # print(np.sum(label[60:140, 60:152, 30:130,0]==ll))
        segment = np.zeros((192,192,160,1))
        segment[60:140, 60:152, 30:130,0] = y_pred
    elif shape_info[filename] == '166':
        # print(np.sum(x1[90:170, 95:187, 30:130] * x2) / (x1.sum() + x2.sum()))
        # print(np.sum(label[90:170, 95:187, 30:130,0] == ll))
        segment = np.zeros((256,256, 166, 1))
        segment[90:170, 95:187, 30:130,0] = y_pred
    elif shape_info[filename] == '180':
        # print(np.sum(x1[100:180, 100:192, 40:140] * x2) / (x1.sum() + x2.sum()))
        # print(np.sum(label[100:180, 100:192, 40:140,0] == ll))
        segment = np.zeros((256,256,180, 1))
        segment[100:180, 100:192, 40:140,0] = y_pred
    else:
        raise ValueError('shape error')


    # print(segment.sum())
    # print(y_pred.sum())
    # print(label.get_data().sum(),'\n')
    print(segment.sum(),filename)
    img = Image(segment,label.coordmap)
    save_image(img,OUTPUT+filename)





















