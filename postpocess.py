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
    # img_seq=img_seq[-3600:]

    best_thres = 0
    max_g = -1
    index = -2800
    stride = 10
    best_i = index
    var_rate = 0
    while(index <= -300):
        thres = img_seq[index]

        foreground = img_seq[img_seq>thres]*1000
        w0 = len(foreground)
        vari0 = foreground.var()
        m0 = np.mean(foreground)

        background = img_seq[img_seq <= thres]*1000
        w1 = len(background)
        vari1 = background.var()
        m1 = np.mean(background)

        g = ((m0-m1)**2)/(vari0+vari1)
        # g = (m0-m1)**2

        if g > max_g:
            max_g = g
            best_thres = thres
            best_i = index
            var_rate = vari1/vari0

        index+=stride
    print(best_i,var_rate)
    image[image>best_thres] = 1
    image[image<=best_thres] = 0
    return image

def threshold_filter(image,y_pred):
    region = image[image>0.75]
    y_pred[region] = 0
    return y_pred