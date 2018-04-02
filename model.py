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
from postpocess import ostu,get_bound,score_grad
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


def get_model(modelname,axis=None,loss=None):
    if modelname == 'Unet':
        x = Input((80,80,40,1))
        conv0 = block_warp('conv',x,16)
        conv1 = block_warp('conv', MaxPool3D(padding='same',strides=2)(x),32)
        conv2 = block_warp('conv', MaxPool3D(padding='same',strides=2)(conv1),64)
        deconv1 = block_warp('deconv', conv2,32)
        deconv1 = block_warp('conv', concatenate([deconv1,conv1]),32)
        deconv2 = block_warp('deconv', deconv1,16)
        deconv2 = block_warp('conv', concatenate([deconv2, conv0]),16)
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
        # conv0 = block_warp('conv_2d', x, 32)
        # conv1 = block_warp('conv_2d', MaxPool2D(padding='same', strides=2)(x), 64)
        # conv2 = block_warp('conv_2d', MaxPool2D(padding='same', strides=2)(conv1), 128)
        # deconv1 = block_warp('deconv_2d', conv2, 64)
        # deconv1 = block_warp('conv_2d', concatenate([deconv1, conv1]), 64)
        # deconv2 = block_warp('deconv_2d', deconv1, 32)
        # deconv2 = block_warp('conv_2d', concatenate([deconv2, conv0]), 32)
        # output = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(deconv2)
    else:
        raise ValueError("don't write this model")
    assert loss is not None
    model = Model(inputs=[x],outputs=[output])
    model.compile(optimizer=Nadam(lr=0.001,clipvalue=1),loss=loss)
    print(model.summary())
    return model

def focalLoss(y_true,y_pred,alpha=2):
    weight1 = K.pow(1 - y_pred, alpha)
    weight2 = K.pow(y_pred,alpha)
    loss = y_true * K.log(y_pred) * weight1 +\
        (1 - y_true) * K.log(1 - y_pred) * weight2
    loss = -K.mean(loss)
    return loss

def sumLoss(y_true,y_pred,w=1e-7):
    loss = norm_celoss(y_true, y_pred)

    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)

    num_pos = K.sum(y_true,axis=1)
    pos_loss = (K.sum(y_pred*y_true,axis=1)-num_pos)**2
    pos_loss /=num_pos

    num_neg = K.sum(1-y_true,axis=1)
    neg_loss = (K.sum(y_pred*(1-y_true), axis=1)-num_neg)**2
    neg_loss /= num_neg
    thres_loss = K.mean(pos_loss+neg_loss)
    loss = (1-w)*loss + w*thres_loss
    return loss

def norm_celoss(y_true,y_pred,w=0.05):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    num_pos = K.sum(y_true,axis=1)
    num_neg = K.sum(1-y_true,axis=1)
    loss_pos = K.sum(y_true * K.log(y_pred),axis=1)/num_pos
    loss_neg = K.sum((1-y_true) * K.log(1-y_pred),axis=1)/num_neg
    loss = (w*loss_pos+(1-w)*loss_neg)
    loss = -K.mean(loss)
    return loss

def diceLoss(y_true, y_pred,smooth = 0):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    y = y_true * y_pred
    intersection = K.sum(y,axis=1)
    loss =  2*( intersection + smooth) / (K.sum(y_true,axis=1) + K.sum(y_pred,axis=1) + smooth)
    # loss = K.log(loss)
    return -K.mean(loss)

def regression_metric(y_true,y_pred):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    true_thres = K.sum(y_true,axis=1)
    pred_thres = K.sum(y_pred,axis=1)
    loss = K.mean((true_thres-pred_thres)**2)
    return loss**0.5

def pos_metric(y_true,y_pred):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    true_thres = K.sum(y_true, axis=1)
    pred_thres = K.sum(y_pred*y_true, axis=1)
    loss = K.mean((true_thres - pred_thres) ** 2)
    return loss ** 0.5

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
        score += 2*np.sum(i*j)/(np.sum(i)+np.sum(j))
    score = score/len(y)
    return score

def pos_reg_score(y_true,y_pred):
    score = 0
    for i, j in zip(y_pred, y_true):
        pos_num = j.sum()
        thres = np.sum(i*j)
        score += (pos_num-thres)**2
    score = score / len(y_true)
    return score ** 0.5














