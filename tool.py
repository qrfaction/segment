from nipy.io.api import save_image,load_image
from nipy.core.api import Image, AffineTransform
from setting import LABEL_PATH,IMAGE_PATH,INFO,OUTPUT,PRE_LABEL_PATH,INFO,crop_setting
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from random import randint,choice

def get_files(path,prefix = True):
    import os
    files = os.listdir(path)   #一对label和image文件同名
    if prefix:
        files = [path + f for f in files]
    else:
        files = [f for f in files]
    return np.array(files)

def get_image(f):
    return np.load(f)

def get_batch_images(files):
    return [get_image(f).tolist() for f in files]

def crop_3d(image,id_h,area=None):
    """
                 x1        y1       z1            x2       y2      z2         x      y         z
    180        115:161   118:172   55:81   90   115:162  112:174  99:121    37:214  36:239   28:149
    192        75:120    81:135    46:70   78   74:122   80:134   86:114    15:160  20:182   16:143
    256_166    100:163   105:173   48:75   82   98:162   107:173  90:117    25:215  28:237   19:144

    80 * 80 * 40
    :param image: 图像
    :param id_h: 海马体ID
    :param area:  crop区域
    :return:
    """
    if area is None:
        area = crop_setting['3d'+str(image.shape[2])]
    z = 'z'+str(id_h)
    return image[
                area['x'][0]:area['x'][1],
                area['y'][0]:area['y'][1],
                area[z][0]:area[z][1]
            ]

def crop_2d_slice(image,id_h,axis,shift,window=5,area=None):
    window = window//2
    if area is None:
        area = crop_setting['2d'+str(image.shape[2])]
    z = 'z'+str(id_h)
    if axis == 'x':
        return image[
            area['x'][0]+shift-window : area['x'][0]+shift+window+1,
            area['y'][0]:area['y'][1],
            area[z][0]:area[z][1]
        ]
    elif axis == 'y':
        return image[
            area['x'][0]:area['x'][1],
            area['y'][0] + shift - window : area['y'][0] + shift + window + 1,
            area[z][0]:area[z][1]
        ]
    elif axis == 'z':
        return image[
            area['x'][0]:area['x'][1],
            area['y'][0]:area['y'][1],
            area[z][0] + shift - window : area[z][0] + shift + window + 1,
                ]
    else:
        raise ValueError('crop_2d axis error')

class Generator_3d(Dataset):
    def __init__(self,files):
        self.images = []
        self.labels = []
        for i in [1,2]:
            for f in files:
                self.images.append((IMAGE_PATH+f,i))
                self.labels.append((LABEL_PATH+f,i))

        self.length = len(self.images)

    def __getitem__(self, index):  # 返回的是tensor
        image_file,h_id = self.images[index]
        label_file,h_id = self.labels[index]
        image = get_image(image_file)
        label = get_image(label_file)
        data = np.concatenate([image,label],axis=-1)
        assert data.shape == (80,80,40,2)
        data = self.process(data,h_id)
        image = data[:1]
        label = data[1:]
        return torch.FloatTensor(image),torch.FloatTensor(label)

    def process(self,image,h_id):
        image = crop_3d(image,h_id)
        image = self.flip(image)
        image = image.transpose((3,0,1,2))
        return image

    def __len__(self):
        return self.length

    def flip(self,image):
        axis = randint(0, 2)
        if axis == 0:
            image = image[::-1, :, :, :]
        elif axis == 1:
            image = image[:, ::-1, :, :]
        elif axis == 2:
            image = image[:, :, ::-1, :]
        else:
            raise ValueError('3d generator flip error')
        return image

    def Gs_noise(self,image,noise_num=5):
        image = image + torch.randn(image.shape) * noise_num

class Generator_2d_slice(Dataset):
    def __init__(self, files,axis):
        self.axis = axis
        if axis == 'x':
            size = crop_setting['2d166']['x'][1] - crop_setting['2d166']['x'][0]
        elif axis == 'y':
            size = crop_setting['2d166']['y'][1] - crop_setting['2d166']['y'][0]
        elif axis == 'z':
            size = crop_setting['2d166']['z1'][1] - crop_setting['2d166']['z1'][0]
        else:
            raise ValueError('generator 2d slice axis error')
        self.flip_axis = ['x','y','z']
        self.flip_axis.remove(axis)

        self.images = []
        self.labels = []
        for i in [1, 2]:
            for shift in range(size):
                for f in files:
                    self.images.append((IMAGE_PATH+f,i,shift))
                    self.labels.append((LABEL_PATH+f,i,shift))

        self.length = len(self.images)

    def __getitem__(self, index):  # 返回的是tensor
        image_file,h_id,shift = self.images[index]
        label_file,h_id,shift = self.labels[index]
        image = get_image(image_file)
        label = get_image(label_file)
        data = np.concatenate([image, label], axis=-1)
        data = self.process(data, h_id,shift)
        image = data[0]
        label = data[1]
        return torch.FloatTensor(image),torch.FloatTensor(label)

    def process(self, image, h_id,shift):
        image = crop_2d_slice(image,h_id,self.axis,shift)
        image = self.flip(image)
        image = image.transpose((3, 0, 1, 2))
        return image

    def __len__(self):
        return self.length

    def flip(self,image):
        axis = choice(self.flip_axis)
        if axis == 'x':
            image = image[::-1, :, :, :]
        elif axis == 'y':
            image = image[:, ::-1, :, :]
        elif axis == 'z':
            image = image[:, :, ::-1, :]
        else:
            raise ValueError('3d generator flip error')
        return image



def seg_recovery_3d(y_pred,filename):
    label = load_image(PRE_LABEL_PATH + filename)
    shape = label.get_data().shape
    segment = np.zeros(shape)

    h1 = y_pred[0].around()
    h2 = y_pred[1].around()

    area = crop_setting['3d' + str(shape[2])]

    segment[
        area['x'][0]:area['x'][1],
        area['y'][0]:area['y'][1],
        area['z1'][0]:area['z1'][1],
        0
    ] = h1[0]
    segment[
        area['x'][0]:area['x'][1],
        area['y'][0]:area['y'][1],
        area['z2'][0]:area['z2'][1],
        1
    ] = h2[0]

    img = Image(segment,label.coordmap)
    save_image(img,OUTPUT+filename)




def size_exchange(array):
    array1=[]
    array2=[]
    if array.shape == (1,192,192,160):
        for i in range(array.shape[3]):
            slices = PIL.Image.fromarray(array[0,:,:,i])
            slicesx = slices.resize((256,256),PIL.Image.AFFINE)
            slicesxx = np.asarray(slicesx) 
            array1.append(slicesxx)

        array1 = np.stack(array1)
        array1 = np.transpose(array1,(1,2,0))
        array1 = array1[np.newaxis,:,:,:]
        
        for i in range(array1.shape[1]):
            slices = PIL.Image.fromarray(array1[0,i,:,:])
            slicesx = slices.resize((212,256),PIL.Image.AFFINE)
            slicesxx = np.asarray(slicesx) 
            array2.append(slicesxx)
            
        array2 = np.stack(array2)
        array2 = array2[np.newaxis,:,:,:]
        
    else:
        print ('shibai')
#    return data
    return array2[:,:,:,26:186]




    # print(segment.sum())
    # print(y_pred.sum())
    # print(label.get_data().sum(),'\n')
    print(segment.sum(),filename)
    img = Image(segment,label.coordmap)
    save_image(img,OUTPUT+filename)


# =============================================================================
# 把一个三维array的size(1,192,192,160)转为(1,256,256,180)
# =============================================================================
def size_exchange(array):
    array1=[]
    array2=[]
    if array.shape == (1,192,192,160):
        for i in range(array.shape[3]):
            slices = PIL.Image.fromarray(array[0,:,:,i])
            slicesx = slices.resize((256,256),PIL.Image.AFFINE)
            slicesxx = np.asarray(slicesx) 
            array1.append(slicesxx)

        array1 = np.stack(array1)
        array1 = np.transpose(array1,(1,2,0))
        array1 = array1[np.newaxis,:,:,:]
        
        for i in range(array1.shape[1]):
            slices = PIL.Image.fromarray(array1[0,i,:,:])
            slicesx = slices.resize((212,256),PIL.Image.AFFINE)
            slicesxx = np.asarray(slicesx) 
            array2.append(slicesxx)
            
        array2 = np.stack(array2)
        array2 = array2[np.newaxis,:,:,:]
        
    else:
        print ('shibai')
#    return data
    return array2[:,:,:,26:186]



















