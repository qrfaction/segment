from model import get_model,dice_metric
import torch
from tool import get_batch_images,get_files,Generator_3d
import numpy as np
from setting import MODEL_PATH,BATCHSIZE,OUTPUT,K,IMAGE_PATH,LABEL_PATH,SUMMARY_PATH




class segment_model:

    def __init__(self,valfiles,modelname='Unet'):

        self.basenet = get_model(modelname=modelname)


        self.valfiles = valfiles
        self.get_valset()

    def get_valset(self):
        val_images = [IMAGE_PATH + f for f in self.valfiles]
        val_labels = [LABEL_PATH + f for f in self.valfiles]
        self.val_images = get_batch_images(val_images)

        labels = []
        for f in val_labels:
            labels.append(np.load(f))
        self.val_labels = np.array(labels)

    def fit(self,X,Y):
        self.basenet.fit(X,Y,batch_size=2)

    def predict(self,X):
        return self.basenet.predict(X,batch_size=1)

    def evaluate(self):
        y_pred = self.predict(self.val_images)
        score = dice_metric(y_pred,self.val_labels)
        print(score)
        return score

    def save(self,path):
        self.basenet.save_weights(path)

    def load(self,path):
        self.basenet.load_weights(path)

def train_model(model,train_files,batchsize = BATCHSIZE,model_name = 'baseline'):

    generator = Generator_3d(train_files,batchsize)

    iter = 1
    best_score = -1
    best_epoch = 1
    while True:
        samples_x,samples_y = generator.get_batch_data()

        model.fit(samples_x,samples_y)
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


def main(use_gpu=True):
    from sklearn.cross_validation import KFold
    files = get_files(LABEL_PATH,prefix=False)

    kf = KFold(len(files), n_folds=K, shuffle=True)


    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = files[train_index]
        validset = files[valid_index]

        model = get_model(valfiles=validset,use_gpu=use_gpu)


        train_model(model,trainset,batchsize=2)
        model.load(MODEL_PATH+'baseline.pkl')
        model.get_segment()
        break
if __name__=='__main__':
    main(True)
























