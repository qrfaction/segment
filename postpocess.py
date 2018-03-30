import numpy as np


def average(results,weights=None):

    if weights is None:
        weights = np.ones(len(results))

    y_pred = np.zeros(results[0].shape)
    for w,result in zip(weights,results):
        y_pred +=w*result

    y_pred /=np.sum(weights)

    return y_pred


def boost():

    pass

def ostu(image):
    img_seq= np.sort(image.flatten())
    print(img_seq)
    best_thres = 0
    max_g = -1
    index = -2800
    stride = 10
    while(index <= -300):
        thres = img_seq[index]
        foreground = img_seq[img_seq>thres]
        w0 = len(foreground)
        u0 = np.mean(foreground)

        background = img_seq[img_seq <= thres]
        w1 = len(background)
        u1 = np.mean(background)

        g = w0 * w1 * (u0-u1)**2

        if g > max_g:
            max_g = g
            best_thres = thres
        index+=stride

    image[image>best_thres] = 1
    image[image<=best_thres] = 0
    return image

def threshold_filter(image,y_pred):
    region = image[image>0.75]
    y_pred[region] = 0
    return y_pred