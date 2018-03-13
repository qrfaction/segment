from tool import get_files
from nipy import load_image,save_image
from setting import IMAGE_PATH,LABEL_PATH,PRE_IMAGE_PATH,PRE_LABEL_PATH
import numpy as np
from tqdm import tqdm
from nipy.core.api import Image, AffineTransform

def get_label():
    images = get_files(PRE_IMAGE_PATH)
    labels = get_files(PRE_LABEL_PATH)

    for f_i,f_l in zip(images,labels):
        im = load_image(f_i)
        label = load_image(f_l)

        print(im.coordmap,label.coordmap)
        # print(im.shape,label.shape)
        break

def crop_image():
    images = get_files(PRE_IMAGE_PATH)
    labels = get_files(PRE_LABEL_PATH)

    for f_i, f_l  in tqdm(zip(images, labels)):
        im = load_image(f_i).get_data()
        label = load_image(f_l).get_data()
        sum1 = np.sum(label[:,:,:] == 1)
        sum2 = np.sum(label[:,:,:] == 2)
        if im.shape == (256,256,180,1):
            im = im[100:180,100:192,40:140]
            label = label[100:180,100:192,40:140]
        elif im.shape == (256,256,166,1):
            im = im[90:170,95:187,30:130]
            label = label[90:170,95:187,30:130]
        elif im.shape == (192,192,160,1):
            im = im[60:140,60:152,30:130]
            label = label[60:140,60:152,30:130]

        crop_sum1 = np.sum(label[:,:,:] == 1)
        crop_sum2 = np.sum(label[:,:,:] == 2)

        assert sum1 == crop_sum1
        assert sum2 == crop_sum2
        assert im.shape==(80,92,100,1)

        L = np.zeros((label.shape[0], label.shape[1], label.shape[2],3))
        L[:,:,:,0] = np.float64(label[:,:,:,0] == 0)
        L[:,:,:,1] = np.float64(label[:,:,:,0] == 1)
        L[:,:,:,2] = np.float64(label[:,:,:,0] == 2)

        label = L
        im = im.transpose((3, 0, 1, 2))
        label = label.transpose((3, 0, 1, 2))

        np.save(IMAGE_PATH + f_l[len(PRE_LABEL_PATH):-7]+'.npy',im)
        np.save(LABEL_PATH + f_l[len(PRE_LABEL_PATH):-7]+'.npy',label)

if __name__=='__main__':
    crop_image()
    # get_label()














