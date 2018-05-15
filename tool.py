from nipy.io.api import save_image,load_image
from nipy.core.api import Image
from config import LABEL_PATH,IMAGE_PATH,OUTPUT,PRE_LABEL_PATH,INFO,crop_config
import numpy as np
import json
from random import randint

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
    return [get_image(f) for f in files]

def swap_axis(image,task,axis):
    if task == 'slice':
        if axis == 'x':
            return image.transpose((2,1,0))
        elif axis == 'y':
            return image.transpose((0,2,1))
        elif axis == 'z':
            return image
        else:
            raise ValueError("swapaxis error  task slice")
    elif task == 'convlstm':
        if axis == 'x':
            return image
        elif axis == 'y':
            return image.transpose((1, 0, 2, 3))
        elif axis == 'z':
            return image.transpose((2, 1, 0, 3))
        else:
            raise ValueError("swapaxis error  task convlstm")
    elif task == 'Unet':
        return image
    else:
        raise ValueError("don't have this task")

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
        area = crop_config['3d'+str(image.shape[2])]
    z = 'z'+str(id_h)
    return image[
                area['x'][0]:area['x'][1],
                area['y'][0]:area['y'][1],
                area[z][0]:area[z][1]
            ]

def select_area(img_info,h_id):
    h = 'h' + str(h_id) + '_'
    area = {}
    for axis,i_size in zip(['x','y','z'],[80,80,40]):
        len_h = img_info[h+axis+'_range']+1
        start = img_info[h+axis]

        assert len_h<i_size
        start = randint(start[0]-(i_size-len_h),start[0])
        area[axis] = [start,start+i_size]

    area['z'+str(h_id)] = area['z']
    return area

class Generator_3d:
    def __init__(self,files,batchsize):
        self.images = []
        self.labels = []
        self.files = []
        for i in [1,2]:
            for f in files:
                self.images.append((IMAGE_PATH+f,i))
                self.labels.append((LABEL_PATH+f,i))
                self.files.append((f[:-3]+'nii.gz',i))

        with open(INFO+'image_info.json','r') as f:
            self.img_info = json.loads(f.read())

        self.batchsize = batchsize
        self.begin = 0
        self.end = self.batchsize
        self.index = list(range(0, len(self.images)))
        np.random.shuffle(self.index)


    def getitem(self, index):  # 返回的是tensor
        image_file,h_id = self.images[index]
        label_file,h_id = self.labels[index]
        file,h_id = self.files[index]

        image = get_image(image_file)
        label = get_image(label_file)

        data = np.concatenate([image,label],axis=-1)
        data = self.process(data,h_id,file)
        assert data.shape == (80,80,40,2)

        image = data[:, :, :, :1]
        label = data[:, :, :, 1:]
        return image,label

    def process(self,image,h_id,file):
        # area = select_area(self.img_info[file], h_id)
        image = crop_3d(image,h_id)
        image = self.flip(image)
        return image

    def get_batch_data(self):
        batch_image = []
        batch_label = []
        for i in self.index[self.begin:self.end]:
            image,label = self.getitem(i)
            batch_image.append(image)
            batch_label.append(label)

        self.begin = self.end
        self.end += self.batchsize
        if self.end > len(self.labels):
            np.random.shuffle(self.index)
            self.begin = 0
            self.end = self.batchsize

        return np.array(batch_image),np.array(batch_label)

    def flip(self,image):
        axis = randint(0, 1)
        if axis == 0:
            image = image[::-1, :, :, :]
        axis = randint(0, 1)
        if axis == 0:
            image = image[:, ::-1, :, :]
        axis = randint(0, 1)
        if axis == 0:
            image = image[:, :, ::-1, :]
        return image

class Generator_convlstm:
    def __init__(self,files,axis,batchsize):
        self.images = []
        self.labels = []
        for i in [1,2]:
            for f in files:
                self.images.append((IMAGE_PATH+f,i))
                self.labels.append((LABEL_PATH+f,i))


        self.axis = axis

        self.batchsize = batchsize
        self.begin = 0
        self.end = self.batchsize
        self.index = list(range(0, len(self.images)))
        np.random.shuffle(self.index)


    def getitem(self, index):  # 返回的是tensor
        image_file,h_id = self.images[index]
        label_file,h_id = self.labels[index]
        image = get_image(image_file)
        label = get_image(label_file)
        data = np.concatenate([image,label],axis=-1)
        data = self.process(data,h_id)
        # print(data.shape)
        # assert data.shape == (80, 80, 40, 2)
        image = data[:,:,:,:1]
        label = data[:,:,:,1:]
        return image,label

    def process(self,image,h_id):
        image = crop_3d(image,h_id)
        image = self.flip(image)
        image = swap_axis(image,'convlstm',self.axis)
        return image

    def get_batch_data(self):
        batch_image = []
        batch_label = []
        for i in self.index[self.begin:self.end]:
            image,label = self.getitem(i)
            batch_image.append(image)
            batch_label.append(label)

        self.begin = self.end
        self.end += self.batchsize
        if self.end > len(self.labels):
            np.random.shuffle(self.index)
            self.begin = 0
            self.end = self.batchsize

        return np.array(batch_image),np.array(batch_label)



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

def seg_recovery(y_pred,filename):
    label = load_image(PRE_LABEL_PATH + filename)
    shape = label.get_data().shape
    segment = np.zeros(shape)

    h1 = y_pred[0].around()
    h2 = y_pred[1].around()

    area = crop_config['3d' + str(shape[2])]

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

def deal_label(modelname,label,axis=None):
    if modelname == 'convlstm':
        assert axis is not None
        h1 = crop_3d(label,1)
        h2 = crop_3d(label,2)
        h1 = swap_axis(h1,'convlstm',axis)
        h2 = swap_axis(h2,'convlstm', axis)
    elif modelname == 'Unet':
        h1 = crop_3d(label, 1)
        h2 = crop_3d(label, 2)
    else:
        raise ValueError("don't have this model")
    return h1,h2


