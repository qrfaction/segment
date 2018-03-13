from model import Unet,focalloss,celoss,dice_metric
from Nadam import Nadam
from tensorboardX import SummaryWriter
import torch
from tool import get_batch_images,Generator,get_files
import numpy as np
from setting import MODEL_PATH,BATCHSIZE,OUTPUT,K,IMAGE_PATH,LABEL_PATH,SUMMARY_PATH


torch.backends.cudnn.enabled = False


class get_model:
    def __init__(self,valfiles,alpha = 3,loss = 'ceLoss',use_gpu=True):

        self.basenet = Unet()
        if use_gpu:
            self.basenet.cuda()

        print(self.basenet)

        self.use_gpu = use_gpu

        self.writer = SummaryWriter(SUMMARY_PATH)

        self.get_valset(val_files=valfiles)

        self.optimizer = Nadam(self.basenet.parameters(),lr=0.001)
        self.basenet.train()

        if loss == 'focalLoss':
            self.loss_f = focalloss(alpha=alpha)
        elif loss =='ceLoss':
            self.loss_f = celoss()
        elif loss=='dice_metric':
            pass

        self.iter = 0

    def get_valset(self,val_files):
        val_images = [IMAGE_PATH + f for f in val_files]
        val_labels = [LABEL_PATH + f for f in val_files]
        self.val_images = get_batch_images(val_images)

        labels = []
        for f in val_labels:
            labels.append(np.load(f))
        self.val_labels = np.array(labels)


    def fit(self,X,Y):
        if self.use_gpu:
            X = X.cuda()
            Y = Y.cuda()
        X = torch.autograd.Variable(X)
        Y = torch.autograd.Variable(Y)
        y_pred = self.basenet(X)
        loss = self.loss_f(y_pred, Y)

        self.iter+=1
        self.writer.add_scalar('trainset/Loss',loss[0],self.iter)

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

    def evaluate(self):
        y_pred = self.predict(self.val_images)
        score = dice_metric(y_pred,self.val_labels)
        self.writer.add_scalar('validset/scores',score,self.iter)
        return score

    def save(self,path):
        torch.save(self.basenet.state_dict(),path)

    def load(self,path):
        self.basenet.load_state_dict(torch.load(path))



def train_model(model,train_files,batchsize = BATCHSIZE):

    dataset = Generator(train_files)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=2,
    )
    iter = 1
    best_score = -1
    best_epoch = 1
    while True:
        for samples_x,samples_y in loader:
            model.fit(samples_x,samples_y)
            cur_score = model.evaluate()
            if  best_score < cur_score:
                best_score = cur_score
                best_epoch = iter
                print(best_score, best_epoch, '\n')
            elif iter - best_epoch > 50:  # patience 为5
                model.writer.close()
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

        train_model(model,trainset,batchsize=1)

if __name__=='__main__':
    main(True)