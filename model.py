from torch.utils.data import Dataset
from tool import Generator,get_files,get_images
import torch
import torch.nn as nn
import numpy as np


# 图像       (256, 256, 166, 1)
# 标签为 0  -1


def calcu_padding(in_size,out_size,deconv=False,stride =np.array([2,2,2]) ,
                  kernel_size = np.array([3,3,3]),dialation = np.array([1,1,1])):
    "计算补0数量 "
    if deconv:
        #这里默认output_padding = 0
        return (kernel_size + stride*(in_size - 1) - out_size)//2
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
        return nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding1),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(0.1),
            nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding2),
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
        return nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.1),
        )

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        if True:   # encoder block
            self.conv0 = block_warp('conv',1,64,(256, 256, 166),(256,256,166))

            self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 128 128 83
            self.conv1 = block_warp('conv',64,128,(128,128,83),(128,128,84))

            self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 64 64 42
            self.conv2 = block_warp('conv',128,256,(64,64,42),(64,64,42))

            self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 32 32 21
            self.conv3 = block_warp('conv', 256,512, (32, 32, 21), (32, 32, 21))

        if True:  # decoder block
            self.deconv6 = block_warp('deconv',128 ,64,(128,128,84),(256,256,166))
            self.conv6 = block_warp('conv', 64 + 64, 64, (256, 256, 166), (256, 256, 166))

            self.deconv5 = block_warp('deconv',256,128, (64, 64, 42), (128, 128, 84))
            self.conv5 = block_warp('conv', 128+128,128, (128, 128, 84), (128, 128, 84))

            self.deconv4 = block_warp('deconv',512,256, (32, 32, 21), (64, 64, 42))
            self.conv4 = block_warp('conv',256+256,256, (64, 64, 42), (64, 64, 42))

        self.out_layer = \
            nn.Sequential(
                nn.Conv3d(64,1, kernel_size=1, stride=1),
                nn.Sigmoid()    # batch  channel  w d h
            )


    def forward(self, x):
        conv0 = self.conv1(x)                       # 256 256 166 64

        conv1 = self.conv1(self.maxpool1(conv0))    # 128 128 84 128
        conv2 = self.conv2(self.maxpool2(conv1))   # 64 64 42 256
        conv3 = self.conv3(self.maxpool3(conv2))   # 32 32 21 512

        deconv4 = self.deconv4(conv3)              # 64 64 42 256
        deconv4 = torch.cat([deconv4,conv2],dim=1) # 64 64 42 512
        deconv4 = self.conv4(deconv4)              # 64 64 42 256

        deconv5 = self.deconv5(deconv4)               # 128 128 84 128
        deconv5 = torch.cat([deconv5, conv1], dim=1)  # 128 128 84 256
        deconv5 = self.conv5(deconv5)                 # 128 128 84 128

        deconv6 = self.deconv6(deconv5)                 # 256 256 166 64
        deconv6 = torch.cat([deconv6, conv0], dim=1)    # 256 256 166 128
        deconv6 = self.conv6(deconv6)                  # 256 256 166 64

        output = self.out_layer(deconv6)              # 256 256 166 2
        output = output.squeeze()
        return output


class focalloss(nn.Module):
    def __init__(self,alpha):
        super(focalloss, self).__init__()
        self.alpha = alpha
    def forward(self,y_pred,y_true):
        weight1 = torch.pow(1-y_pred,self.alpha)
        weight2 = torch.pow(y_pred,self.alpha)
        loss = -(
                    y_true * torch.log(y_pred) * weight1 +
                    (1-y_true) * torch.log(1-y_pred) * weight2
                )
        loss = torch.sum(loss)/(y_true.size()[0]*6)
        return  loss

class celoss(nn.Module):
    def __init__(self):
        super(celoss, self).__init__()
    def forward(self,y_pred,y_true):
        loss = y_true * y_pred.log()
        loss = -loss.mean()
        return  loss


class diceloss(nn.Module):
    def __init__(self,smooth):
        super(diceloss, self).__init__()
        self.smooth = smooth
    def forward(self,y_pred,y_true):
        loss = -(torch.sum(y_pred*y_true)
                   +self.smooth)/(torch.sum(y_true)+torch.sum(y_pred)+self.smooth)
        return  loss



















