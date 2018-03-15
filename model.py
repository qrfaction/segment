import torch
import torch.nn as nn
import numpy as np


# 图像       (256, 256, 166, 1)
# 标签为 0  -1


def calcu_padding(in_size,out_size,deconv=False,stride =np.array([2,2,2]) ,
                  kernel_size = np.array([3,3,3]),dialation = np.array([1,1,1])):
    "计算补0数量 "
    if deconv:
        #这里默认output_padding = 1
        return (kernel_size + stride*(in_size - 1) - out_size + 1 )//2
    else:
        #  这里默认步长为1
        return (dialation*(kernel_size-1) + out_size - in_size)//2

def block_warp(block_name,in_dim,out_dim,in_size,out_size,kernel_size=np.array([3,3,3])):
    in_size = np.array(in_size)
    out_size = np.array(out_size)
    if block_name == 'conv':
        padding1 = calcu_padding(
            in_size = in_size,
            out_size = out_size,
            kernel_size = kernel_size,
        )
        padding2 = calcu_padding(
            in_size=out_size,
            out_size=out_size,
            kernel_size=kernel_size,
        )
        padding1 = tuple(padding1.tolist())
        padding2 = tuple(padding2.tolist())
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size.tolist(), stride=1, padding=padding1),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(0.1),
            nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size.tolist(), stride=1, padding=padding2),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(0.1),
        )
    elif block_name == 'deconv':
        padding = calcu_padding(
            in_size=in_size,
            out_size=out_size,
            kernel_size=kernel_size,
            deconv=True,
            stride=np.array([2,2,2]),

        )
        padding = tuple(padding.tolist())
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size.tolist(), stride=2, padding=padding,output_padding=1),
            nn.LeakyReLU(0.1),
        )



class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        if True:   # encoder block
            self.conv0 = block_warp('conv',1,32,(80,92,40),(80,92,40))

            self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 128 128 83
            self.conv1 = block_warp('conv',32,64,(40,46,20),(40,46,20))

            self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 64 64 42
            self.conv2 = block_warp('conv',64,128,(20,23,10),(20,23,10))


        if True:  # decoder block
            self.deconv6 = block_warp('deconv',64,32,(40,46,20),(80,92,40))
            self.conv6 = block_warp('conv',32+32,32, (80,92,40), (80,92,40))

            self.deconv5 = block_warp('deconv',128,64, (20, 23,20), (40,46,40))
            self.conv5 = block_warp('conv',64+64,64, (40, 46, 40), (40,46,40))


        self.out_layer = \
            nn.Sequential(
                nn.Conv3d(32,1, kernel_size=1, stride=1),
                nn.Sigmoid()    # batch  channel  w d h
            )

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(self.maxpool1(conv0))
        conv2 = self.conv2(self.maxpool2(conv1))

        deconv5 = self.deconv5(conv2)
        deconv5 = torch.cat([deconv5, conv1], dim=1)
        deconv5 = self.conv5(deconv5)

        deconv6 = self.deconv6(deconv5)
        deconv6 = torch.cat([deconv6, conv0], dim=1)  # 192,192,160 128
        deconv6 = self.conv6(deconv6)                 # 192,192,160 64

        output = self.out_layer(deconv6)              # 192,192,160  3
        return output



class focalloss(nn.Module):
    def __init__(self,alpha):
        super(focalloss, self).__init__()
        self.alpha = alpha
    def forward(self,y_pred,y_true):
        weight1 = torch.pow(1-y_pred,self.alpha)
        weight2 = torch.pow(y_pred,self.alpha)
        loss = (
                    y_true * torch.log(y_pred) * weight1 +
                    (1-y_true) * torch.log(1-y_pred) * weight2
                )
        loss = -torch.sum(loss)/(y_true.size()[0]*6)
        return  loss

class celoss(nn.Module):
    def __init__(self):
        super(celoss, self).__init__()
        self.alpha = 2
    def forward(self,y_pred,y_true):
        pos = y_pred
        neg = 1-y_pred
        weight2 = torch.pow(pos, self.alpha)
        weight1 = torch.pow(neg, self.alpha)
        loss = y_true * torch.log(pos) *weight1  + (1-y_true)*torch.log(neg) *weight2
        loss = -torch.mean(loss)
        return  loss

class diceloss(nn.Module):
    def __init__(self,smooth=0):
        super(diceloss, self).__init__()
        self.smooth = smooth
    def forward(self,y_pred,y_true):
        batch_size = y_true.size(0)
        loss = 0
        for i in range(batch_size):
            loss += -(2.0*torch.sum(y_pred[i]*y_true[i]+self.smooth))/(
                                  torch.sum(y_pred[i]) + torch.sum(y_true[i])+self.smooth)

        return  loss/batch_size

def dice_metric(y_pred,y):
    score = 0
    for i,j in zip(y_pred,y):
        # a = i.argmax(axis=0)
        # b = j.argmax(axis=0)
        # print(a.shape,b.shape)
        # a = np.float64(a>0)
        # b = np.float64(b>0)
        # x1 = np.sum(a*b)
        # x2 = np.sum(a)
        # x3 = np.sum(b)
        # score += 2*x1/(x2+x3)
        i = np.around(i)
        # i[i>0.1] = 1
        # i[i<=0.1] = 0
        score += 2*np.sum(i*j)/(np.sum(i)+np.sum(j))
        # print(i.shape)
    score = score/len(y)
    return score

def ce_loss(y_pred,y):
    score = 0
    for i, j in zip(y_pred, y):
        loss= y*np.log(y_pred)
        loss = -np.mean(loss)
        score +=loss
    score = score / len(y)
    return score














