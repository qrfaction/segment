from nipy import save_image
from model import Unet,focalloss,celoss
from Nadam import Nadam
from setting import LABEL_PATH
from torch import nn
import torch
from tool import get_batch_images,Generator,get_files
import numpy as np
from setting import MODEL_PATH,BATCHSIZE,OUTPUT,K,IMAGE_PATH,LABEL_PATH

torch.backends.cudnn.enabled = False


class get_model:
    def __init__(self,alpha = 3,loss = 'ceLoss',use_gpu=True):
        super(get_model,self).__init__()

        # self.basenet = nn.DataParallel(baseNet(dim, embedding_matrix,trainable)).cuda()
        self.basenet = Unet()
        if use_gpu:
            self.basenet.cuda()

        self.use_gpu = use_gpu


        self.optimizer = Nadam(self.basenet.parameters(),lr=0.001)
        self.basenet.train()

        if loss == 'focalLoss':
            self.loss_f = focalloss(alpha=alpha)
        elif loss =='ceLoss':
            self.loss_f = celoss()
        elif loss=='dice_metric':
            pass

    def fit(self,X,Y):
        if self.use_gpu:
            X = X.cuda()
            Y = Y.cuda()
        X = torch.autograd.Variable(X)
        Y = torch.autograd.Variable(Y)
        y_pred = self.basenet(X)
        loss = self.loss_f(y_pred, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self,X):
        self.basenet.eval()
        # X = get_batch_images(x_files)
        if self.use_gpu:
            X = X.cuda()
        X = torch.autograd.Variable(X,volatile=True)
        y_pred = self.basenet(X).data
        self.basenet.train()
        if self.use_gpu:
            return y_pred.cpu().numpy()
        return y_pred.numpy()


    def save(self,path):
        torch.save(self.basenet.state_dict(),path)

    def load(self,path):
        self.basenet.load_state_dict(torch.load(path))

def train_model(model,modelname,train_files,val_files,batchsize = BATCHSIZE):
    def dice_metric(y_pred,y):
        score = 0
        for i,j in zip(y_pred,y):
            score +=2*np.sum(i*j)/(np.sum(i)+np.sum(j))
        score = score/len(y)
        return score

    def get_val_images(val_files):
        val_images = [IMAGE_PATH + f for f in val_files]
        val_labels = [LABEL_PATH + f for f in val_files]
        val_images = get_batch_images(val_images)
        labels = []
        for f in val_labels:
            labels.append(np.load(f))
        labels = np.array(labels)
        return val_images,labels

    dataset = Generator(train_files)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=2,
    )
    iter = 1
    best_score = -1

    val_images,val_labels = get_val_images(val_files)

    while True:
        for samples_x,samples_y in loader:
            model.fit(samples_x,samples_y)

            if iter >= 1:
                # evaulate
                y_pred = model.predict(val_images)
                cur_score = dice_metric(y_pred,val_labels)

                if iter == 1 or best_score < cur_score:
                    best_score = cur_score
                    best_epoch = iter
                    best_result = y_pred
                    print(best_score, best_epoch, '\n')
                    # model.save(MODEL_PATH+modelname+ '.pkl')
                elif iter - best_epoch > 50:  # patience 为5
                    # for output,file in zip(best_result,val_files):
                    #     output = np.around(output)
                    #     np.save(OUTPUT+file,output)
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

        model = get_model(use_gpu=use_gpu)

        train_model(model,"baseline" ,trainset,validset,batchsize=1)

if __name__=='__main__':
    main(True)