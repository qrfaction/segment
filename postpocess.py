import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

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
    thres_index = np.where(g>0.7*max_g)[0] + low
    return thres_index

def score_grad(image):
    thres_seq = np.sort(image.flatten())
    img_seq = 10 * (thres_seq - thres_seq[0]) / (thres_seq[-1] - thres_seq[0])
    bound = get_bound(img_seq,-400,-2800,window=50)
    thres_index = int(np.median(bound))
    thres = thres_seq[thres_index]
    # for thres in bound:
    #     neg_mean = img_seq[-3000:thres].mean()
    #     pos_mean =

    # thres = thres_seq[bound]
    # print(bound,thres)
    image[image>=thres]=1
    image[image<thres] =0
    return image

def ostu(image):
    thres_seq= np.sort(image.flatten())
    # img_seq=img_seq[-25600:]

    img_seq = (thres_seq-thres_seq[0])/(thres_seq[-1]-thres_seq[0])
    best_thres = 0
    max_g = -1
    index = -2800
    best_i = index
    # var_rate = 0

    bound = get_bound(img_seq,-300,-2800)
    img_seq = img_seq[-3000:]
    for index in bound:
        thres = img_seq[index]

        foreground = img_seq[img_seq>thres]
        var0 = foreground.var()
        m0 = np.mean(foreground)

        background = img_seq[img_seq<=thres]
        var1 = background.var()
        m1 = np.mean(background)

        weight = len(foreground)*(3000-len(foreground))

        g = weight*((m0-m1)**2)/(var0+var1)

        if g > max_g:
            max_g = g
            best_thres = thres_seq[best_i]
            best_i = index
            # var_rate = var1/var0


    # print(best_i,thres_seq[best_i])
    image[image>best_thres] = 1
    image[image<=best_thres] = 0
    return image

def threshold_filter(image,y_pred):
    region = image[image>0.75]
    y_pred[region] = 0
    return y_pred