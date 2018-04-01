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


def boost():

    pass

def auto_thres(image):
    img_seq = np.sort(image.flatten())
    thres_index = int(img_seq[-3000:].sum())
    if thres_index>2800:
        thres_index=2800
    elif thres_index<300:
        thres_index=300
    print(thres_index)
    thres = img_seq[-thres_index]
    image[image>=thres]=1
    image[image<thres]=0
    return image

def get_bound(img_seq,neg=-2800):
    max_g = 0
    best_i = 0
    size = len(img_seq)+neg
    for i in range(240000,size):
        g = img_seq[i+10] - img_seq[i]
        if g > max_g:
            max_g = g
            best_i = i
    return best_i

def score_grad(image):
    thres_seq = np.sort(image.flatten())
    img_seq = 10 * (thres_seq - thres_seq[0]) / (thres_seq[-1] - thres_seq[0])
    bound = get_bound(img_seq,-300)
    bound = bound-80*80*40
    if bound<-2800:
        bound=-2800
    elif bound>-400:
        bound=-400
    thres = thres_seq[bound]
    print(bound,thres)
    image[image>=thres]=1
    image[image<thres] = 1
    return image

def thres_predict(image,neg=-2800,pos=-300,step=10):
    thres_seq = np.sort(image.flatten())
    img_seq = 10*(thres_seq - thres_seq[0])/(thres_seq[-1] - thres_seq[0])

    thres = thres_seq[-1200]

    bound = get_bound(img_seq)
    mean_neg = img_seq[bound:neg].mean()
    mean_pos = img_seq[pos:].mean()

    for i in range(-2800,-400,step):
        dist_neg = img_seq[i]-mean_neg
        dist_pos = mean_pos-img_seq[i]
        if dist_pos<dist_neg:
            thres = thres_seq[i]
            print(i,80*80*40-bound,thres)
            break
    image[image >= thres] = 1
    image[image < thres] = 0
    return image

def ostu(image):
    thres_seq= np.sort(image.flatten())
    # img_seq=img_seq[-25600:]

    img_seq = (thres_seq-thres_seq[0])/(thres_seq[-1]-thres_seq[0])
    best_thres = 0
    max_g = -1
    index = -2800
    stride = 10
    best_i = index
    var_rate = 0
    while(index <= -400):
        thres = img_seq[index]

        foreground = img_seq[img_seq>thres]
        var0 = foreground.var()
        m0 = np.mean(foreground)
        num_pos = len(foreground)+300

        background = img_seq[img_seq<=thres]
        var1 = background.var()
        m1 = np.mean(background)
        num_neg = 2800-num_pos

        weight = (num_pos*num_neg)

        g = weight*((m0-m1)**2)/(var0+var1)
        # g = (m0-m1)**2

        if g > max_g:
            max_g = g
            best_thres = thres_seq[best_i]
            best_i = index
            var_rate = var1/var0

        index+=stride
    print(best_i,var_rate,thres_seq[best_i])
    image[image>best_thres] = 1
    image[image<=best_thres] = 0
    return image

def threshold_filter(image,y_pred):
    region = image[image>0.75]
    y_pred[region] = 0
    return y_pred