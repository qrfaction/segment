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
            self.conv0 = block_warp('conv',1,16,(80,92,100),(80,92,100))

            self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 128 128 83
            self.conv1 = block_warp('conv',16,32,(40,46,50),(40,46,50))

            self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   # 64 64 42
            self.conv2 = block_warp('conv',32,64,(20,23,25),(20,23,25))


        if True:  # decoder block
            self.deconv6 = block_warp('deconv',32,16,(40,46,50),(80,92,100))
            self.conv6 = block_warp('conv',16+16,16, (80,92,100), (80,92,100))

            self.deconv5 = block_warp('deconv',64,32, (20, 23,25), (40,46,50))
            self.conv5 = block_warp('conv',32+32,32, (40, 46, 50), (40,46,50))

            # self.deconv4 = block_warp('deconv',512,256, (24,24, 20), (48, 48, 40))
            # self.conv4 = block_warp('conv',256+256,256, (48, 48,40), (32,32, 40))

        self.out_layer = \
            nn.Sequential(
                nn.Conv3d(16,3, kernel_size=1, stride=1),
                nn.Softmax()    # batch  channel  w d h
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
        output = output.squeeze()
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



















