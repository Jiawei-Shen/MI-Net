
import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.init import kaiming_normal_,constant_ 

from math import sqrt

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()

        assert kernel_size % 2 == 1, 'kernel size should be odd'

        self.padding = (kernel_size - 1) // 2

        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)

        weight_tensor[0, 0, (kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1

        self.weight = nn.Parameter(weight_tensor)

        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)

        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()

        return F.conv2d(x, expand_weight,

                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()

        self.pre_conv1 = ShareSepConv(dilation * 2 - 1)

        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)

        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)

        self.pre_conv2 = ShareSepConv(dilation * 2 - 1)

        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)

        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))

        y = self.norm2(self.conv2(self.pre_conv2(y)))

        return F.relu(x + y)


class SmoothDilatedResBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResBlock, self).__init__()

        self.pre_conv1 = ShareSepConv(dilation * 2 - 1)

        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)

        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)

        self.pre_conv2 = ShareSepConv(dilation * 2 - 1)

        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=1, bias=False)

        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))

        y = self.norm2(self.conv2(self.pre_conv2(y)))

        return F.relu(y)



class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)

        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)

        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)

        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))

        y = self.norm2(self.conv2(y))

        return F.relu(x + y)


class ResBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                               bias=False)

        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)

        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=1, groups=group, bias=False)

        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)                               

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))

        y = self.norm2(self.conv2(y))

        return F.relu(y)



class MI_Net(nn.Module):
    def __init__(self, in_c=3, out_c=3, only_residual=True):

        super(MI_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_c, 64, 3, 1, 1, bias=False)

        self.norm1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)

        self.norm2 = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)

        self.norm3 = nn.InstanceNorm2d(64, affine=True)

        self.res2 = SmoothDilatedResBlock(64, dilation=2)

        self.res4 = SmoothDilatedResBlock(64, dilation=4)

        self.res1 = ResBlock(64, dilation=1)

        self.gate = nn.Conv2d(64 * 3, 3, 3, 1, 1, bias=True)

        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)

        self.norm4 = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.norm5 = nn.InstanceNorm2d(64, affine=True)

        self.deconv1 = nn.Conv2d(64, out_c, 1)

        self.deconv2 = nn.Conv2d(out_c, out_c, 1)

        self.pre_processing = nn.Sequential(
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
            ResidualBlock(64, dilation=1),
        )

        self.only_residual = only_residual
        self.layer1=nn.Sequential(nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0),
        nn.AdaptiveAvgPool2d(1),
        nn.Linear(1,1),
        nn.Linear(1,1),
        nn.Sigmoid())
        self.conv_su1=nn.Conv2d(3,3,1,1,0)
        self.conv_su2=nn.Conv2d(3,3,1,1,0)


        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                kaiming_normal_(m.weight,0.1)     
                

            
    def forward(self, Ih):

        x = F.relu(self.norm1(self.conv1(Ih)))

        x = F.relu(self.norm2(self.conv2(x)))

        x = F.relu(self.norm3(self.conv3(x)))

        x = self.pre_processing(x)

        I_f = x 
              
        for _ in range(8):
            x = self.res1(x)                     
            x = torch.add(x, I_f)          
        x1 = x         
     
        for _ in range(8):
            x = self.res2(x)            
            x = torch.add(x, x1)
        x2 = x    

        for _ in range(8):
            x = self.res4(x)            
            x = torch.add(x, x2)
        x3 = x 


        gates = self.gate(torch.cat((x1, x2, x3), dim=1))
        
        gated_y = x1 * gates[:, [0], :, :] + x2 * gates[:, [1], :, :] + x3 * gates[:, [2], :, :]

        x = F.relu(self.norm4(self.deconv3(gated_x)))

        x = F.relu(self.norm5(self.deconv2(x)))


        x = self.deconv1(x)          


        x=self.conv_su1(x)+self.conv_su2(Ih)                                   
        x=nn.LeakyReLU(0.1,inplace=False)(x)                                
        x=x+torch.mul(x,self.layer1(x))

        x = self.deconv2(x)

        return x