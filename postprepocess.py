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


def threshold_filter(image,y_pred):
    region = image[image>0.75]
    y_pred[region] = 0
    return y_pred