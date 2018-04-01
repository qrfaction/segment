import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
session = tf.Session(config=config)
KTF.set_session(session)

from model import get_model,dice_metric,auc,pos_reg_score
from tool import get_batch_images,get_files,Generator_3d,deal_label,inference,Generator_2d_slice,Generator_convlstm
import numpy as np
from setting import MODEL_PATH,BATCHSIZE,OUTPUT,K,IMAGE_PATH,LABEL_PATH,SUMMARY_PATH



class segment_model:

    def __init__(self,valfiles,modelname='Unet',axis = None):

        self.basenet = get_model(modelname=modelname,axis=axis)


        self.valfiles = valfiles
        self.get_valset()

        self.modelname = modelname
        self.axis = axis


    def get_valset(self):
        val_images = [IMAGE_PATH + f for f in self.valfiles]
        val_labels = [LABEL_PATH + f for f in self.valfiles]
        self.val_images = get_batch_images(val_images)

        labels = []
        for f in val_labels:
            labels.append(np.load(f))
        self.val_labels = np.array(labels)

    def fit(self,X,Y):
        # self.basenet.fit(X,Y,batch_size=2)
        score = self.basenet.train_on_batch(X,Y)
        print('train:',score)

    def predict(self,X):
        # return self.basenet.predict(X,batch_size=1)
        return self.basenet.predict_on_batch(X)

    def evaluate(self):
        y_pred = []
        y = []
        for label,image in zip(self.val_labels,self.val_images):
            h1,h2 = inference(self.basenet,self.modelname,image,self.axis)
            h1_label,h2_label = deal_label(self.modelname,label,self.axis)
            y_pred.append(h1)
            y.append(h1_label)
            y_pred.append(h2)
            y.append(h2_label)
        score = dice_metric(y,y_pred)
        # score = auc(y,y_pred)
        reg_loss = pos_reg_score(y,y_pred)
        print('val:',score,'reg_loss',reg_loss)
        return score

    def save(self,path):
        self.basenet.save_weights(path)

    def load(self,path):
        self.basenet.load_weights(path)

def train_model(model,train_files,batchsize = BATCHSIZE,model_name = 'Unet',axis=None):

    if model_name=='slice':
        assert axis is not None
        generator = Generator_2d_slice(train_files,axis,batchsize=batchsize)
    elif model_name=='convlstm':
        assert axis is not None
        generator = Generator_convlstm(files=train_files,axis=axis,batchsize=batchsize)
    elif model_name=='Unet':
        generator = Generator_3d(train_files,batchsize)
    else:
        raise NameError("don't have this model")

    iter = 1
    best_score = -1
    best_epoch = 1
    while True:
        samples_x,samples_y = generator.get_batch_data()

        model.fit(samples_x,samples_y)
        if iter>1000:
            cur_score = model.evaluate()
            if  best_score < cur_score:
                best_score = cur_score
                best_epoch = iter
                model.save(MODEL_PATH+model_name+'.h5')
                print(best_score, best_epoch, '\n')
            elif iter - best_epoch > 300:  # patience 为5
                model.load(MODEL_PATH+model_name+'.h5')
                return best_score
        iter += 1


def main(modelname='Unet',axis=None):
    from sklearn.cross_validation import KFold
    files = get_files(LABEL_PATH,prefix=False)

    kf = KFold(len(files), n_folds=K, shuffle=True)


    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = files[train_index]
        validset = files[valid_index]

        model = segment_model(valfiles=validset,modelname=modelname,axis=axis)


        train_model(model,trainset,batchsize=3,model_name=modelname,axis=axis)

        break
if __name__=='__main__':
    main('slice','z')
























