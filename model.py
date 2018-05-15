import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)

from keras.utils import multi_gpu_model

import numpy as np
from keras import backend as K
from keras.layers import Conv3D,BatchNormalization,Conv3DTranspose,Input,MaxPool3D,Activation,Lambda,Permute
from keras.layers import concatenate,ConvLSTM2D,TimeDistributed,Conv2D,Conv2DTranspose,MaxPool2D,Bidirectional,add
from keras.models import Model
from keras.optimizers import Nadam,RMSprop
from sklearn.metrics import roc_auc_score

def block_warp(block_name,input_layer,filters,kernal_size=3, dilation_rate=1):
    def conv_block(input_layer,filters,k=3):
        y = Conv3D(filters=filters, kernel_size=k, padding='same')(input_layer)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        return y

    if block_name == 'conv':
        y = conv_block(input_layer,filters)
        y = conv_block(y,filters)

    elif block_name == 'dialtion':
        y = Conv3D(filters=filters, kernel_size=kernal_size, padding='same', dilation_rate=dilation_rate)(input_layer)
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
        y = TimeDistributed(Conv2DTranspose(filters=filters,kernel_size=3,padding='same',strides=2))(input_layer)
        y = TimeDistributed(BatchNormalization())(y)
        y = TimeDistributed(Activation('relu'))(y)

    elif block_name == 'inception':
        filters = filters//4

        # c1 = Conv3D(filters=filters, kernel_size=1, padding='same')(input_layer)
        c1 = conv_block(input_layer,filters,1)

        # c3 = Conv3D(filters=filters, kernel_size=1, padding='same')(input_layer)
        # c3 = Conv3D(filters=filters, kernel_size=3, padding='same')(c3)
        c3 = conv_block(input_layer,filters,1)
        c3 = conv_block(c3,filters,3)

        c5 = MaxPool3D(pool_size=3,padding='same',strides=1)(input_layer)
        # c5 = Conv3D(filters=filters, kernel_size=3, padding='same')(c5)
        c5 = conv_block(c5,filters,3)


        # c7 = Conv3D(filters=filters, kernel_size=1, padding='same')(input_layer)
        # c7 = Conv3D(filters=filters, kernel_size=3, padding='same')(c7)
        # c7 = Conv3D(filters=filters, kernel_size=3, padding='same')(c7)
        c7 = conv_block(input_layer,filters,1)
        c7 = conv_block(c7, filters, 3)
        c7 = conv_block(c7, filters, 3)

        y = concatenate([c1,c3,c5,c7])
        # c_all = BatchNormalization()(c_all)
        # y = Activation('relu')(c_all)

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
        conv0 = block_warp('conv',x,24)
        conv1 = block_warp('conv', MaxPool3D(padding='same',strides=2)(conv0),48)

        conv2 = block_warp('conv', MaxPool3D(padding='same',strides=2)(conv1),96)

        deconv1 = block_warp('deconv', conv2,48)
        # conv1 = block_warp('conv',conv1,64)
        # conv1 = block_warp('conv',conv1,32)
        deconv1 = block_warp('conv', concatenate([deconv1,conv1]),48)

        deconv2 = block_warp('deconv', deconv1,24)
        # conv0 = block_warp('conv', conv0, 32)
        # conv0 = block_warp('conv', conv0, 16).....
        deconv2 = block_warp('conv', concatenate([deconv2, conv0]),48)

        output = Conv3D(filters=1,kernel_size=3,padding='same')(deconv2)
        # output = BatchNormalization()(output)
        # output =

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


        conv0 = block_warp('conv',x,8)
        encoder = Conv3D(16,kernel_size=3,padding='same',activation='relu',strides=2)(conv0)

        x_z = encoder
        x_x = Permute((3, 2, 1, 4))(encoder)
        x_y = Permute((2, 1, 3, 4))(encoder)

        lstm_z = Bidirectional(ConvLSTM2D(filters=32,padding='same',return_sequences=True,kernel_size=3)
                                ,merge_mode='sum')(x_z)


        lstm_y = Bidirectional(ConvLSTM2D(filters=32,padding='same',return_sequences=True,kernel_size=3)
                               , merge_mode='sum')(x_y)
        lstm_y = Permute((2,1,3,4))(lstm_y)


        lstm_x = Bidirectional((ConvLSTM2D(filters=32,padding='same',return_sequences=True,kernel_size=3))
                               , merge_mode='sum')(x_x)
        lstm_x = Permute((3,2,1,4))(lstm_x)


        add_layer = add([lstm_x,lstm_y,lstm_z])


        decoder = Conv3DTranspose(filters=32,kernel_size=3,strides=2,padding='same')(add_layer)

        # conv1 = block_warp('conv',conv0,32)
        #
        # decoder = multiply([conv1,decoder])

        output = Conv3D(filters=1,kernel_size=3,activation='tanh',padding='same')(decoder)
    else:
        raise ValueError("don't have this model")

    output = Lambda(norm_layer)(output)

    assert loss is not None

    # with tf.device('/cpu:0'):
    model = Model(inputs=[x],outputs=[output])
    # model = multi_gpu_model(model,gpus=3)
    model.compile(optimizer=Nadam(lr=0.001,clipnorm=1),loss=loss,metrics=[diceMetric])
    print(model.summary())
    return model

def norm_layer(x):
    x = (x - K.min(x,axis=[1,2,3,4],keepdims=True)) / \
           (K.max(x,axis=[1,2,3,4],keepdims=True)-K.min(x,axis=[1,2,3,4],keepdims=True))
    return x

def celoss(y_true,y_pred):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    loss = y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred)
    return -K.mean(loss)

def diceMetric(y_true,y_pred,smooth = 0):
    y_pred = K.batch_flatten(K.round(y_pred))
    y_true = K.batch_flatten(y_true)
    y = y_true * y_pred
    intersection = K.sum(y, axis=1)
    loss = 2 * (intersection + smooth) / (K.sum(y_true, axis=1) + K.sum(y_pred, axis=1) + smooth)
    return K.mean(loss)

def diceLoss(y_true, y_pred):
    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    y = y_true * y_pred
    intersection = K.sum(y, axis=1)
    loss = 2 * intersection / (K.sum(y_true, axis=1) + K.sum(y_pred, axis=1))
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
        score += 2*np.sum(i*j)/(np.sum(i)+np.sum(j))
    score = score/len(y)
    return score













