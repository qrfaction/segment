import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)

from model import get_model,dice_metric,auc
from tool import get_batch_images,get_files,Generator_3d,deal_label,crop_3d
from tool import swap_axis,Generator_convlstm
import numpy as np
from config import MODEL_PATH,BATCHSIZE,OUTPUT,K,IMAGE_PATH,LABEL_PATH



class segment_model:

    def __init__(self,valfiles,modelname='Unet',axis = None,metric=dice_metric,loss=None,postPocess=None):

        self.basenet = get_model(modelname=modelname,axis=axis,loss=loss)

        if valfiles is not None:
            self.valfiles = valfiles
            self.get_valset()

        self.modelname = modelname
        self.axis = axis
        self.metric = metric

        self.postPocess = postPocess

    def get_valset(self):
        val_images = [IMAGE_PATH + f for f in self.valfiles]
        val_labels = [LABEL_PATH + f for f in self.valfiles]
        self.val_images = get_batch_images(val_images)

        labels = []
        for f in val_labels:
            labels.append(np.load(f))
        self.val_labels = np.array(labels)

    def fit(self,X,Y):
        score = self.basenet.train_on_batch(X,Y)
        return score

    def predict(self,X):
        return self.basenet.predict_on_batch(X)

    def inference(self,image):
        if self.modelname == 'convlstm':
            assert self.axis is not None
            h1 = crop_3d(image, 1)
            h2 = crop_3d(image, 2)
            h1 = swap_axis(h1, 'convlstm', self.axis)
            h2 = swap_axis(h2, 'convlstm', self.axis)
            h1,h2 = self.predict(np.array([h1,h2]))
        elif self.modelname == 'Unet':
            h1 = crop_3d(image, 1)
            h2 = crop_3d(image, 2)
            h1,h2 = self.predict(np.array([h1, h2]))
        else:
            raise ValueError("don't have this model")
        return np.array(h1), np.array(h2)

    def evaluate(self):
        y_pred = []
        y = []
        for label,image in zip(self.val_labels,self.val_images):
            h1,h2 = self.inference(image)
            h1_label,h2_label = deal_label(self.modelname,label,self.axis)

            h1,h2=self.postPocess(h1),self.postPocess(h2)

            y_pred.append(h1)
            y.append(h1_label)
            y_pred.append(h2)
            y.append(h2_label)

        score = self.metric(y,y_pred)

        print('val:',score)
        return score

    def save(self,path):
        self.basenet.save_weights(path)

    def load(self,path):
        self.basenet.load_weights(path)

def train_model(model,train_files,batchsize = BATCHSIZE,model_name = 'Unet',axis=None):

    if 'convlstm' in model_name:
        assert axis is not None
        generator = Generator_convlstm(files=train_files,axis=axis,batchsize=batchsize)
    elif 'Unet' in model_name:
        generator = Generator_3d(train_files,batchsize)
    else:
        raise NameError("don't have this model")

    iter = 1
    best_score = -1
    best_epoch = 1
    while True:
        samples_x,samples_y = generator.get_batch_data()

        score = model.fit(samples_x,samples_y)
        if iter % 100 == 1 :
            print('train',score)

        # if iter>7500:
            cur_score = model.evaluate()
            if  best_score < cur_score:
                best_score = cur_score
                best_epoch = iter
                model.save(MODEL_PATH+model_name+'.h5')
                print(best_score, best_epoch, '\n')
        # elif iter - best_epoch > 500:
        #     return best_score
        iter += 1


def main(loss,modelname='Unet',axis=None,metric=dice_metric,postPocess=None,postPocess_str=None):
    from sklearn.cross_validation import KFold
    files = get_files(LABEL_PATH,prefix=False)


    K = 10
    kf = KFold(len(files), n_folds=K, shuffle=True)



    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = files[train_index]
        validset = files[valid_index]


        model = segment_model(valfiles=validset,modelname=modelname,
                              axis=axis,metric=metric,loss=loss,postPocess=postPocess)
        if axis == None:
            model_id = modelname+'_'+str(i)+'_None_'+postPocess_str+'_'
        else:
            model_id = modelname+'_'+str(i)*2+'_'+axis+'_'+postPocess_str+'_'

        best_score = train_model(model,trainset,batchsize=8,model_name=model_id,axis=axis)
        print(str(i),': ',best_score)


if __name__=='__main__':
    from model import diceLoss
    from postpocess import score_grad,ostu,thres_predict,threshold_filter

    main(
        loss=diceLoss,
        modelname='convlstm',
        axis='x',
        metric=dice_metric,
        postPocess=threshold_filter,
        postPocess_str='around',
    )


