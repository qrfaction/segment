from tool import get_files
from nipy import load_image,save_image
from setting import IMAGE_PATH,LABEL_PATH,PRE_IMAGE_PATH,PRE_LABEL_PATH,INFO
import numpy as np
from tqdm import tqdm
import json

def get_shape():
    images = get_files(PRE_IMAGE_PATH,prefix=False)
    im_shapes = {}
    for f in images:
        im = load_image(PRE_IMAGE_PATH+f)
        print(im.__array__)

        im = im.get_data()
        im_shapes[f] = str(im.shape[2])

    with open(INFO+'shape.json','w') as f:
        f.write(json.dumps(im_shapes,indent=4, separators=(',', ': ')))


def normlize_data(image):
    vaild_area = (image >= 0)
    invaild_area = (image < 0)

    image[vaild_area] = np.log(image[vaild_area] + 1)

    i_mean = image[vaild_area].mean()
    i_std = image[vaild_area].std()

    low = i_mean - 3.5 * i_std
    high = i_mean + 3.5 * i_std
    image = (image - low) / (high - low)

    image[invaild_area] = -1

    return image

def prepocess():
    images = get_files(PRE_IMAGE_PATH)
    labels = get_files(PRE_LABEL_PATH)
    files = get_files(PRE_IMAGE_PATH,prefix=False)

    for f_i, f_l ,f in tqdm(zip(images, labels,files)):
        im = load_image(f_i).get_data()
        label = load_image(f_l).get_data()

        label[label == 2] = 1

        # im = cv2.resize(im,(192,192,160,1),interpolation=cv2.INTER_LINEAR)
        # print(im.shape)
        # im = normlize_data(im)

        np.save(IMAGE_PATH + f[:-7]+'.npy',im)
        np.save(LABEL_PATH + f[:-7]+'.npy',label)

if __name__=='__main__':
    # get_shape()
    prepocess()














