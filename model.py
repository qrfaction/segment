import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)


import numpy as np
from keras import backend as K
from keras.layers import Conv3D,BatchNormalization,Conv3DTranspose,Input,MaxPool3D,Activation
from keras.layers import concatenate,ConvLSTM2D,TimeDistributed,Conv2D,Conv2DTranspose,MaxPool2D
from keras.models import Model
from keras.optimizers import Nadam
from postpocess import ostu,auto_thres
from sklearn.metrics import roc_auc_score

def block_warp(block_name,input_layer,filters,kernal_size=3, dilation_rate=(1,1)):
    if block_name == 'conv':
        y = Conv3D(filters=filters,kernel_size=3,padding='same')(input_layer)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv3D(filters=filters,kernel_size=3,padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    elif block_name == 'deconv':
        y = Conv3DTranspose(filters=filters,kernel_size=3,strides=2,padding='same')(input_layer)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    elif block_name == 'time_conv':
        y = TimeDistributed(Conv2D(filters=filters,kernel_size=3,padding='same'))(input_layer)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('relu'))(y)
    elif block_name == 'time_deconv':
        y = TimeDistributed(Conv2DTranspose(filters=filters,kernel_size=3,padding='same'))(input_layer)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('relu'))(y)
    elif block_name == 'inception':
        # y = TimeDistributed(Conv2DTranspose(filters=filters,kernel_size=3,padding='same'))(input_layer)
        # y = TimeDistributed(BatchNormalization())(y)
        # y = TimeDistributed(PReLU())(y)s
        pass

    elif block_name == 'conv_2d':
        y = Conv2D(filters=filters, kernel_size=kernal_size,strides=1, padding='same')(input_layer)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    elif block_name == 'deconv_2d':
        y = Conv2DTranspose(filters=filters, kernel_size=kernal_size, strides=2, padding='same')(input_layer)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    elif block_name == 'deeplab_block':
        y = Conv2D(filters=filters, kernel_size=3, padding='same', dilation_rate=dilation_rate)(input_layer)
        y = MaxPool2D((2, 2), strides=2,  padding='same')(y)
    else:
        raise ValueError("layer error")
    return y


def get_model(modelname,axis=None):
    if modelname == 'Unet':
        x = Input((80,80,40,1))
        conv0 = block_warp('conv',x,32)
        conv1 = block_warp('conv', MaxPool3D(padding='same',strides=2)(x),64)
        conv2 = block_warp('conv', MaxPool3D(padding='same',strides=2)(conv1),128)
        deconv1 = block_warp('deconv', conv2,64)
        deconv1 = block_warp('conv', concatenate([deconv1,conv1]),64)
        deconv2 = block_warp('deconv', deconv1,32)
        deconv2 = block_warp('conv', concatenate([deconv2, conv0]),32)
        output = Conv3D(filters=1,kernel_size=3,activation='sigmoid',padding='same')(deconv2)

    elif modelname == 'convlstm':
        assert axis is not None
        if axis == 'x':                       # x y z
            x = Input((80, 80, 40, 1))
        elif axis == 'y':                     # y x z
            x = Input((80, 80, 40, 1))
        elif axis == 'z':                    # z y x
            x = Input((40, 80, 80, 1))
        else:
            raise ValueError("convlstm axis error")

        pass

    elif modelname == 'slice':
        assert axis is not None
        if axis == 'x':  #  y z  x
            x = Input((80,40,5))
        elif axis == 'y':  # x z y
            x = Input((80,40,5))
        elif axis == 'z':  # x y z
            x = Input((80,80,5))
        else:
            raise ValueError("deeplabv3+ axis error")
        conv0 = block_warp('conv_2d', x, 32)
        conv1 = block_warp('conv_2d', MaxPool2D(padding='same', strides=2)(x), 64)
        conv2 = block_warp('conv_2d', MaxPool2D(padding='same', strides=2)(conv1), 128)
        deconv1 = block_warp('deconv_2d', conv2, 64)
        deconv1 = block_warp('conv_2d', concatenate([deconv1, conv1]), 64)
        deconv2 = block_warp('deconv_2d', deconv1, 32)
        deconv2 = block_warp('conv_2d', concatenate([deconv2, conv0]), 32)
        output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(deconv2)
    else:
        raise ValueError("don't write this model")

    model = Model(inputs=[x],outputs=[output])
    model.compile(optimizer=Nadam(lr=0.001),loss=focalLoss)
    print(model.summary())
    return model

def focalLoss(y_true,y_pred,alpha=3):
    weight1 = K.pow(1 - y_pred, alpha)
    weight2 = K.pow(y_pred,alpha)
    loss = y_true * K.log(y_pred) * weight1 +\
        (1 - y_true) * K.log(1 - y_pred) * weight2
    loss = -K.mean(loss)
    return loss

def sumLoss(y_true,y_pred,alpha=2,w=0):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)

    # weight1 = K.pow(1 - y_pred, alpha)
    # weight2 = K.pow(y_pred, alpha)
    # loss = y_true * K.log(y_pred) * weight1 + \
    #        (1 - y_true) * K.log(1 - y_pred) * weight2
    loss = y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred)
    loss = -K.mean(loss)

    num_pos = K.sum(y_true,axis=1)
    pos_loss = (K.sum(y_pred*y_true,axis=1)-num_pos)**2
    pos_loss /=num_pos

    num_neg = K.sum(1-y_true,axis=1)
    neg_loss = (K.sum(y_pred*(1-y_true), axis=1)-num_neg)**2
    neg_loss /= num_neg
    thres_loss = K.mean(pos_loss+neg_loss)
    loss = (1-w)*loss + w*thres_loss
    return loss

def stdLoss(y_true,y_pred,alpha=2,w=0.05):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)

    weight1 = K.pow(1 - y_pred, alpha)
    weight2 = K.pow(y_pred, alpha)
    loss = y_true * K.log(y_pred) * weight1 + \
           (1 - y_true) * K.log(1 - y_pred) * weight2
    loss = -K.mean(loss)

    y_pred = y_pred*1000

    num_pos = K.sum(y_true,axis=1)
    num_neg = K.sum(1-y_true, axis=1)

    mean_pos = K.sum(y_true*y_pred,axis=1)/num_pos
    mean_neg = K.sum((1-y_true)*y_pred,axis=1)/num_neg
    std_class = (mean_neg-mean_pos)**2                     #  类间方差

    mean_pos = K.reshape(mean_pos,(-1,1))
    std_pos = (y_true * (y_pred - mean_pos)) ** 2
    std_pos = K.sum(std_pos,axis=1) / num_pos      #正例方差

    mean_neg = K.reshape(mean_neg,(-1,1))
    std_neg = ((1-y_true) * (y_pred - mean_neg)) ** 2
    std_neg = K.sum(std_neg,axis=1) /num_neg        #负例方差

    std_loss = (std_pos + std_neg)/std_class
    std_loss = K.mean(std_loss)
    # print(std_loss)

    loss = (1-w) * loss + w * std_loss
    return loss

def norm_celoss(y_true,y_pred):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    num_pos = K.sum(y_true,axis=1)
    num_neg = K.sum(1-y_true,axis=1)
    loss_pos = K.sum(y_true * K.log(y_pred),axis=1)/num_pos
    loss_neg = K.sum((1-y_true) * K.log(1-y_pred),axis=1)/num_neg
    loss = -0.5*(loss_pos+loss_neg)
    return loss

def diceLoss(y_true, y_pred,smooth = 0):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    y = y_true * y_pred
    intersection = K.sum(y,axis=1)
    loss =  2*( intersection + smooth) / (K.sum(y_true,axis=1) + K.sum(y_pred,axis=1) + smooth)
    # loss = K.log(loss)
    return -K.mean(loss)

def auc(y_true,y_pred):
    score = 0
    for i, j in zip(y_true, y_pred):
        i = i.flatten()
        j = j.flatten()
        score += roc_auc_score(i,j)
    score = score / len(y_true)
    return score

def dice_metric(y,y_pred):
    score = 0
    for i,j in zip(y_pred,y):
        # i = np.around(i)
        i = ostu(i)
        # i = auto_thres(i)
        score += 2*np.sum(i*j)/(np.sum(i)+np.sum(j))
    score = score/len(y)
    return score

def ce_loss(y_pred,y):
    score = 0
    for i, j in zip(y_pred, y):
        loss= j * np.log(i)
        loss = -np.mean(loss)
        score +=loss
    score = score / len(y)
    return score














