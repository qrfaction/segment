from nipy.io.api import save_image,load_image
from nipy.core.api import Image, AffineTransform
from setting import LABEL_PATH,IMAGE_PATH,WINDOW,OUTPUT,PRE_LABEL_PATH,INFO,crop_setting
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
        area = crop_setting['3d'+str(image.shape[2])]
    z = 'z'+str(id_h)
    return image[
                area['x'][0]:area['x'][1],
                area['y'][0]:area['y'][1],
                area[z][0]:area[z][1]
            ]

def crop_2d_slice(image,id_h,axis,shift,window=WINDOW,area=None):
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

class Generator_3d:
    def __init__(self,files,batchsize):
        self.images = []
        self.labels = []
        for i in [1,2]:
            for f in files:
                self.images.append((IMAGE_PATH+f,i))
                self.labels.append((LABEL_PATH+f,i))

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
        data = self.process(data, h_id)
        assert data.shape == (80,80,40,2)
        image = data[:, :, :, :1]
        label = data[:, :, :, 1:]
        return image,label

    def process(self,image,h_id):
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
        axis = randint(0, 2)
        if axis == 0:
            image = image[::-1, :, :, :]
        elif axis == 1:
            image = image[:, ::-1, :, :]
        else:
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
        assert data.shape == (80,80,40,2)
        data = self.process(data,h_id)
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

        return batch_image,batch_label



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

class Generator_2d_slice:
    def __init__(self, files,axis,batchsize):
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



        self.batchsize = batchsize
        self.begin = 0
        self.end = self.batchsize
        self.index = list(range(0, len(self.images)))
        np.random.shuffle(self.index)

    def getitem(self, index):  # 返回的是tensor
        image_file,h_id,shift = self.images[index]
        label_file,h_id,shift = self.labels[index]
        image = get_image(image_file)
        label = get_image(label_file)
        data = np.concatenate([image, label], axis=-1)
        data = self.process(data, h_id,shift)
        image = data[:,:,:,0]
        label = data[:,:,:,1]

        postion = 1 + WINDOW // 2
        label = label[:, :, postion:postion + 1]

        return image,label


    def process(self, image, h_id,shift):
        image = crop_2d_slice(image,h_id,self.axis,shift)
        image = self.flip(image)
        image = swap_axis(image,'slice',self.axis)
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

        return batch_image,batch_label

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

def seg_recovery(y_pred,filename):
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

def deal_label(modelname,label,axis=None):
    if modelname == 'slice':
        assert axis is not None
        if axis == 'x':
            size = label.shape[0]
        elif axis == 'y':
            size = label.shape[1]
        elif axis == 'z':
            size = label.shape[2]
        else:
            raise ValueError('3d generator flip error')
        h1 = []
        h2 = []
        for i in range(size):
            slice_h1 = crop_2d_slice(label,1,axis,i)
            slice_h2 = crop_2d_slice(label,2,axis,i)
            slice_h1 = swap_axis(slice_h1,'slice',axis)
            slice_h2 = swap_axis(slice_h2,'slice',axis)
            h1.append(slice_h1)
            h2.append(slice_h2)
    elif modelname == 'convlstm':
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

def inference(model,modelname,image,axis=None):
    if modelname == 'slice':
        assert axis is not None
        if axis == 'x':
            size = image.shape[0]
        elif axis == 'y':
            size = image.shape[1]
        elif axis == 'z':
            size = image.shape[2]
        else:
            raise ValueError('3d generator flip error')
        seq_slice_h1 = []
        seq_slice_h2 = []
        for i in range(size):
            slice_h1 = crop_2d_slice(image,1,axis,i)
            slice_h2 = crop_2d_slice(image,2,axis,i)
            slice_h1 = swap_axis(slice_h1,'slice',axis)
            slice_h2 = swap_axis(slice_h2,'slice',axis)
            seq_slice_h1.append(slice_h1)
            seq_slice_h2.append(slice_h2)

        y_pred_h1 = model.predict(seq_slice_h1)
        y_pred_h2 = model.predict(seq_slice_h2)
        h1 = []
        h2 = []
        for i in range(size):
            h1.append(swap_axis(y_pred_h1[i],'slice',axis))
            h2.append(swap_axis(y_pred_h2[i],'slice',axis))
    elif modelname == 'convlstm':
        assert axis is not None
        h1 = crop_3d(image,1)
        h2 = crop_3d(image,2)
        h1 = swap_axis(h1,'convlstm',axis)
        h2 = swap_axis(h2,'convlstm', axis)
        h1 = model.predict([h1])[0]
        h2 = model.predict([h2])[0]
        print(h1.shape)
    elif modelname == 'Unet':
        h1 = crop_3d(image, 1)
        h2 = crop_3d(image, 2)
        h1 = model.predict_on_batch(np.array([h1]))[0]
        h2 = model.predict_on_batch(np.array([h2]))[0]
    else:
        raise ValueError("don't have this model")
    return np.array(h1),np.array(h2)


