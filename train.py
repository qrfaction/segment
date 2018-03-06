from nipy import save_image
from model import Unet,focalloss
from Nadam import Nadam
from torch import nn
import torch
from tool import get_images,Generator,get_files
import numpy as np
from setting import MODEL_PATH,BATCHSIZE,OUTPUT,K

class get_model:
    def __init__(self,files,alpha = 3,loss = 'focalLoss'):
        super(get_model,self).__init__()

        # self.basenet = nn.DataParallel(baseNet(dim, embedding_matrix,trainable)).cuda()
        self.basenet = Unet()

        self.files = files

        self.optimizer = Nadam(self.basenet.parameters(),lr=0.001)
        self.basenet.train()

        if loss == 'focalLoss':
            self.loss_f = focalloss(alpha=alpha)
        elif loss =='ceLoss':
            self.loss_f = nn.BCELoss()
        elif loss=='dice_metric':
            pass

    def fit(self,X,Y):
        X = torch.autograd.Variable(X)
        Y = torch.autograd.Variable(Y)
        y_pred = self.basenet(X)
        loss = self.loss_f(y_pred, Y)
        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self,x_files):
        self.basenet.eval()
        X = get_images(x_files,volatile=True)
        y_pred = self.basenet(X)
        self.basenet.train()
        return -y_pred.numpy()

def train_model(model,modelname,train_files,val_files,batchsize = BATCHSIZE):
    def dice_metric(y_pred,y):
        score = 0
        for i,j in zip(y_pred,y):
            score +=2*np.sum(i*j)/(np.sum(i)+np.sum(j))
        return score

    dataset = Generator(train_files)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=2,
    )
    iter = 1
    best_score = -1

    while True:
        for samples_x,samples_y in loader:
            model.fit(samples_x,samples_y)

            if iter>= 1:
                # evaulate
                y_pred = model.predict(val_files, batch_size=2048, verbose=1)
                cur_score = dice_metric(y_pred,samples_y)
                print(cur_score)

                if iter == 1 or best_score < cur_score:
                    best_score = cur_score
                    best_epoch = iter
                    best_result = y_pred
                    print(best_score, best_epoch, '\n')
                    model.save(model.state_dict(),MODEL_PATH+modelname+ '.pkl')
                elif iter - best_epoch > 10:  # patience 为5
                    for output,file in zip(best_result,val_files):
                        output = output.around()
                        save_image(output,OUTPUT+file)
                    return best_score
            iter += 1


def main():
    from sklearn.cross_validation import KFold
    files = get_files()

    kf = KFold(len(files), n_folds=K, shuffle=True)


    for i, (train_index, valid_index) in enumerate(kf):
        print('第{}次训练...'.format(i))
        trainset = files[train_index]
        validset = files[valid_index]

        model = get_model(trainset)
        train_model(model,"baseline" ,trainset,validset,batchsize=5 )