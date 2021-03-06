import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy import ndimage

def average(results,weights=None):

    if weights is None:
        weights = np.ones(len(results))

    y_pred = np.zeros(results[0].shape)
    for w,result in zip(weights,results):
        y_pred +=w*result

    y_pred /=np.sum(weights)

    return y_pred

def get_bound(img_seq,up=-2800,low=-16000,window=10):

    g = np.zeros(up-low)
    g[0] = img_seq[low:low+window].sum() - img_seq[low-window:low].sum()
    for i in range(low+1,up):
        pos = i - low
        g[pos] = g[pos-1] + img_seq[i+window-1] + img_seq[i-window-1] - 2*img_seq[i-1]


    max_g = g.max()
    thres_index = np.where(g>0.9*max_g)[0] + low
    return thres_index

def score_grad(image):
    thres_seq = np.sort(image.flatten())
    bound = get_bound(thres_seq,-400,-2800,window=2)

    # max_g = -1
    # best_i = -2000
    # for index in range(-2800, -350, 10):
    #     thres = thres_seq[index]
    #
    #     foreground = thres_seq[thres_seq > thres]
    #     m0 = np.mean(foreground)
    #
    #     background = thres_seq[thres_seq <= thres]
    #     m1 = np.mean(background)
    #
    #     weight = len(foreground) * len(background)
    #
    #     g = weight * ((m0 - m1) ** 2)
    #
    #     if g > max_g:
    #         max_g = g
    #         # best_thres = thres_seq[best_i]
    #         best_i = index

    # thres_index = (int(np.median(bound))+best_i)//2
    thres_index = int(np.median(bound))
    thres = thres_seq[thres_index]

    image[image>=thres] = 1
    image[image<thres]  = 0

    return image

def image_smooth(image,size=3):
    conv_kernel = np.ones((size,size,size,1))
    return ndimage.convolve(image,conv_kernel,mode='constant')

def ostu(image):
    thres_seq= np.sort(image.flatten())
    # img_seq=img_seq[-25600:]

    best_thres = 0
    max_g = -1
    index = -2800
    best_i = index
    # var_rate = 0

    scores = []
    for index in range(-2800,-350,10):
        thres = thres_seq[index]

        foreground = thres_seq[thres_seq>thres]
        # var0 = foreground.var()
        m0 = np.mean(foreground)

        background = thres_seq[thres_seq<=thres]
        # var1 = background.var()
        m1 = np.mean(background)

        weight = len(foreground)*len(background)

        g = weight*((m0-m1)**2)

        scores.append((g,index))

        if g > max_g:
            max_g = g
            best_thres = thres_seq[best_i]
            best_i = index

    image[image>best_thres] = 1
    image[image<=best_thres] = 0
    return image


def threshold_filter(image,t=0.5):
    thres_seq = np.sort(image.flatten())
    # thres_index = -int(np.sum(thres_seq[-5000:]))
    thres_index = -len(thres_seq[thres_seq>t])
    if thres_index < -2800:
        thres_index=-2800
    elif thres_index > -400:
        thres_index=-400
    thres = thres_seq[thres_index]
    image[image >= thres] = 1
    image[image < thres] = 0
    return image

def thres_predict(image):
    thres_seq = np.sort(image.flatten())
    img_seq = 10 * (thres_seq - thres_seq[0]) / (thres_seq[-1] - thres_seq[0])

    def get_samples(img_seq,num_neg):
        low = -num_neg-2800

        X = np.zeros((num_neg+2800,4))
        X[:,0] = img_seq[low:]

        # cal grad
        X[0,1] = img_seq[low:low+10].sum()-img_seq[low-10:low].sum()
        for i in range(low+1,-10):
            pos = i-low
            X[pos,1] = X[pos-1,1] + img_seq[i+9] + img_seq[i-11] - 2*img_seq[i-1]
        X[-10:,1] = X[-11,1]

        # cal mean
        for i in range(low,0):
            pos = i-low
            X[pos,2] = img_seq[low-200:i].mean()
            X[pos,3] = img_seq[i:].mean()

        # norm
        for i in range(4):
            min_col = X[:,i].min()
            max_col = X[:,i].max()
            X[:,i] = (X[:,i]-min_col)/(max_col-min_col)

        train_data = np.concatenate([X[:-2800],X[-300:]])
        test_data = X[-2800:-300]
        Y = np.zeros(num_neg+300)
        Y[-300:] = 1
        return train_data,Y,test_data

    train_data,Y,test_data = get_samples(img_seq,10000)
    lr = LogisticRegression(n_jobs=-1,max_iter=200)
    lr.fit(train_data,Y)
    y_pred = lr.predict(test_data)

    thres_index = int(-300 - y_pred.sum())
    thres = thres_seq[thres_index]

    image[image>=thres] = 1
    image[image<thres] = 0

    return image
