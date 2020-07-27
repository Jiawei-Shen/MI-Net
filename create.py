import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as transforms
import os
from torch.utils.data import DataLoader
import time
import numpy as np
import math
import sys
import torch.nn.functional as F
for index in sys.path:
	if'/home/spl208_rtxtitan/anaconda3/envs/py36/lib/python3.6/site-packages' in index:
		sys.path.remove(index)
		sys.path.insert(0, index)
		break
import cv2 

# 以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    # stpe1:初始化
    def __init__(self, txt, target_transform=None):
        fh = open(txt, 'r')  # 打开标签文件
        #hazy_path = os.listdir(path + 'hazy/')
        #gt_path = os.listdir(path + 'gt/')
        imgs = []   #创建列表，装东西
        for line in fh:  # 遍历标签文件每行
            line = line.rstrip()  # 删除字符串末尾的空格
            words = line.split()  # 通过空格分割字符串，变成列表
            imgs.append((words[0], words[1]))  # 把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transforms.Compose([
                                                #transforms.Resize(34),  # 缩放到 96 * 96 大小
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
                                            ])
        self.target_transform = target_transform

    def __getitem__(self, index):  # 检索函数
        fn, label = self.imgs[index]  # 读取文件名、标签
        img = cv2.imread(fn)  # 通过PIL.Image读取图片

        #cv2.imshow('a',img)
        #cv2.waitKey(0)
        
        ipt = self.transform(img)
        #ipt=torch.from_numpy(img).permute(2,0,1).float()
        target = cv2.imread(label)
        target = self.transform(target)
        #target=torch.from_numpy(target).permute(2,0,1).float()

        return ipt, target

    def __len__(self):
        return len(self.imgs)

'''
class ITS_Dataset(Dataset):
    # stpe1:初始化
    def __init__(self, path, target_transform=None):
        
        hazy_path = os.listdir(path + 'hazy/')
        gt_path = os.listdir(path + 'gt/')
        hazy_path.sort()
        gt_path.sort()
        
        imgs = []   #创建列表，装东西
        for i in range(len(gt_path)):  # 遍历标签文件每行
            for j in range(10):
            imgs.append((hazy_path[i*10+j], gt_path[i]))  # 把图片名words[0]，标签int(words[1])放到imgs里
        self.imgs = imgs
        self.transform = transforms.Compose([
                                                #transforms.Resize(34),  # 缩放到 96 * 96 大小
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
                                            ])
        self.target_transform = target_transform

    def __getitem__(self, index):  # 检索函数
        fn, label = self.imgs[index]  # 读取文件名、标签
        img = cv2.imread(fn)  # 通过PIL.Image读取图片

        #cv2.imshow('a',img)
        #cv2.waitKey(0)
        
        ipt = self.transform(img)
        target = cv2.imread(label)
        target = self.transform(target)

        return ipt, target

    def __len__(self):
        return len(self.imgs)
'''

class Hazydataset(Dataset):
    def __init__(self,hazy_path,ground_truth_path):
        self.hazy_path=hazy_path
        self.ground_truth=ground_truth_path
        self.hazy_ids=os.listdir(self.hazy_path)
        self.hazy_ids.sort()
        self.groundtruth=os.listdir(self.ground_truth)
        self.groundtruth.sort()
        self.files=[]
        for id1,id2 in zip(self.hazy_ids,self.groundtruth):         
            hazy=os.path.join(self.hazy_path,id1)
            gr=os.path.join(self.ground_truth,id2)
            self.files.append({"hazy":hazy,"gr":gr})
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        path=self.files[index]

        input_img = cv2.imread(path["hazy"])
        height, weight, channels = input_img.shape
        #input_img = cv2.resize(input_img, None, fx=1/4, fy=1/4)
        input_img = input_img[0:height//2*2, 0:weight//2*2]
                
        input=ArrayToTensor(input_img).permute(2,0,1).float()

        gr_img = cv2.imread(path["gr"])
        #gr_img = cv2.resize(gr_img, None, fx=1/4, fy=1/4)
        gr_img =  gr_img[0:height//2*2, 0:weight//2*2]
        
        gr=ArrayToTensor(gr_img).permute(2,0,1).float()

        return input ,gr  

class Raindataset(Dataset):
    def __init__(self, rain_path,ground_truth_path):
        self.rain_path=rain_path
        self.ground_truth=ground_truth_path
        self.rain_ids=os.listdir(self.rain_path)
        self.rain_ids.sort()
        self.groundtruth=os.listdir(self.ground_truth)
        self.groundtruth.sort()
        self.files=[]
        for id1,id2 in zip(self.rain_ids,self.groundtruth):         
            rain=os.path.join(self.rain_path,id1)
            gr=os.path.join(self.ground_truth,id2)
            self.files.append({"rain":rain,"gr":gr})
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        path=self.files[index]

        input_img = cv2.imread(path["rain"])
        height, weight, channels = input_img.shape
        #input_img = cv2.resize(input_img, None, fx=1/4, fy=1/4)
        input_img = input_img[0:height//2*2, 0:weight//2*2]
                
        input=ArrayToTensor(input_img).permute(2,0,1).float()

        gr_img = cv2.imread(path["gr"])
        #gr_img = cv2.resize(gr_img, None, fx=1/4, fy=1/4)
        gr_img =  gr_img[0:height//2*2, 0:weight//2*2]
        
        gr=ArrayToTensor(gr_img).permute(2,0,1).float()

        return input ,gr  


class Hazydataset_OUT(Dataset):
    def __init__(self,hazy_path,ground_truth_path):
        self.hazy_path=hazy_path
        self.ground_truth=ground_truth_path
        self.hazy_ids=os.listdir(self.hazy_path)
        self.hazy_ids.sort(key = lambda x: float(x.split('_')[0]))
        self.groundtruth=os.listdir(self.ground_truth)
        self.groundtruth.sort(key = lambda x: int(x.split('.')[0]))
        self.files=[]
        for id1,id2 in zip(self.hazy_ids,self.groundtruth):         
            hazy=os.path.join(self.hazy_path,id1)
            gr=os.path.join(self.ground_truth,id2)
            self.files.append({"hazy":hazy,"gr":gr})
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        path=self.files[index]

        input_img = cv2.imread(path["hazy"])
        height, weight, channels = input_img.shape
        input_img =  input_img[0:height//2*2, 0:weight//2*2]        
        input=ArrayToTensor(input_img).permute(2,0,1).float()

        gr_img = cv2.imread(path["gr"])
        gr_img =  gr_img[0:height//2*2, 0:weight//2*2]
        gr=ArrayToTensor(gr_img).permute(2,0,1).float()

        return input ,gr  


class Hazydataset_RESIDE(Dataset):
    def __init__(self,hazy_path,ground_truth_path):
        self.hazy_path=hazy_path
        self.ground_truth=ground_truth_path
        self.hazy_ids=os.listdir(self.hazy_path)
        self.hazy_ids.sort(key = lambda x: float(x.split('_')[0])) #+ float(x.split('_')[1]) * 0.1)
        self.gt_ids=os.listdir(self.ground_truth)
        self.gt_ids.sort(key = lambda x: int(x.split('.')[0]))
        self.files=[]
        for len_gt in range(len(self.gt_ids)):
            for i in range(10):         
                hazy=os.path.join(self.hazy_path,self.hazy_ids[len_gt*10 + i])
                gr=os.path.join(self.ground_truth,self.gt_ids[len_gt])
                self.files.append({"hazy":hazy,"gr":gr})

    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        path=self.files[index]
        input = cv2.imread(path["hazy"])        
        input=ArrayToTensor(input).permute(2,0,1).float()
        gt_img = cv2.imread(path["gr"])           
        gr=ArrayToTensor(gt_img).permute(2,0,1).float()
        return input ,gr  

import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os, sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from utils import *
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
#import cv2


class denoisingset(data.Dataset):
    def __init__(self, path, size=(300, 300)):
        super(denoisingset, self).__init__()
        self.size = size
        self.path = path
        self.train = ['BSD', 'DIV2K']
        self.dirs = ['15', '25', '30', '50', '75']
        self.files = []
        for o in self.dirs:
            dst = os.path.join(path, self.train[0], "noise" + o)
            diro = os.listdir(dst)

            for i in diro:
                noise = os.path.join(dst, i)
                clean = os.path.join(path, self.train[0], 'train', i)
                self.files.append({"noise": noise, "gr": clean})

        for o in self.dirs:
            dst = os.path.join(path, self.train[1], "noise" + o)
            diro = os.listdir(dst)

            for i in diro:
                noise = os.path.join(dst, i)
                clean = os.path.join(path, self.train[1], 'train', i)
                self.files.append({"noise": noise, "gr": clean})
           

    def __len__(self):
        return len(self.files)

    def augdata(self, data, target):
        rand_rot = random.randint(0, 3)
        data = FF.rotate(data, 90 * rand_rot)
        target = FF.rotate(target, 90 * rand_rot)
        #hr = FF.rotate(hr, 90 * rand_rot)

        data = np.array(data)
        target = np.array(target)
        #hr = np.array(hr)

        # cv2.imshow('a',data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(data.shape)

        return ArrayToTensor(data).permute((2, 0, 1)), ArrayToTensor(target).permute((2, 0, 1))

    def __getitem__(self, index):
        id = self.files[index]
        
        # noise = Image.open(id["noise"])
        # clean = Image.open(id["gr"])
        noise = cv2.imread(id["noise"])
        clean = cv2.imread(id["gr"])
        """
        sobelx = cv2.Sobel(noise, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(noise, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        High_frequency = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        """
        noise = Image.fromarray(noise)
        clean = Image.fromarray(clean)
        #hr = Image.fromarray(High_frequency)
        i, j, h, w = tfs.RandomCrop.get_params(noise, self.size)

        noise_crop = FF.crop(noise, i, j, h, w)
        clean_crop = FF.crop(clean, i, j, h, w)
        #hr_crop = FF.crop(hr, i, j, h, w)
        n, c = self.augdata(noise_crop.convert("RGB"), clean_crop.convert("RGB"))

        return n.float().cuda(), c.float().cuda()
class denoisingset1(data.Dataset):
    def __init__(self, path, size=(300, 300)):
        super(denoisingset1, self).__init__()
        self.size = size
        self.path = path
        self.train = ['BSD', 'DIV2K']
        self.dirs = ['15', '25', '30', '50', '75']
        self.files = []
        '''
        for o in self.dirs:
            dst = os.path.join(path, self.train[0], "noise" + o)
            diro = os.listdir(dst)

            for i in diro:
                noise = os.path.join(dst, i)
                clean = os.path.join(path, self.train[0], 'train', i)
                self.files.append({"noise": noise, "gr": clean})

        for o in self.dirs:
            dst = os.path.join(path, self.train[1], "noise" + o)
            diro = os.listdir(dst)

            for i in diro:
                noise = os.path.join(dst, i)
                clean = os.path.join(path, self.train[1], 'train', i)
                self.files.append({"noise": noise, "gr": clean})
        '''        
        sigma =['15', '25', '30', '50', '75']
        name = ['urban100']
        for i in name:
            for j in sigma:
                   
                path =os.path.join(r'/home/spl208_rtxtitan/桌面/shenjw/denoise_dataset/DATASET/test',i)

                dst = os.path.join(path,"noise"+j)
                
                diro =os.listdir(dst)
                for k in diro:
                    noise = os.path.join(dst,k)
                    clean =os.path.join(path,'gr',k)
                    self.files.append({"noise": noise, "gr": clean}) 

      

    def __len__(self):
        return len(self.files)

    def augdata(self, data, target):
        rand_rot = random.randint(0, 3)
        data = FF.rotate(data, 90 * rand_rot)
        target = FF.rotate(target, 90 * rand_rot)
        #hr = FF.rotate(hr, 90 * rand_rot)

        data = np.array(data)
        target = np.array(target)
        #hr = np.array(hr)

        # cv2.imshow('a',data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(data.shape)

        return ArrayToTensor(data).permute((2, 0, 1)), ArrayToTensor(target).permute((2, 0, 1))

    def __getitem__(self, index):
        id = self.files[index]
        
        # noise = Image.open(id["noise"])
        # clean = Image.open(id["gr"])
        noise = cv2.imread(id["noise"])
        clean = cv2.imread(id["gr"])
        """
        sobelx = cv2.Sobel(noise, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(noise, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        High_frequency = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        """
        noise = Image.fromarray(noise)
        clean = Image.fromarray(clean)
        #hr = Image.fromarray(High_frequency)
         
        i, j, h, w = tfs.RandomCrop.get_params(noise, self.size)

        noise_crop = FF.crop(noise, i, j, h, w)
        clean_crop = FF.crop(clean, i, j, h, w)
        #hr_crop = FF.crop(hr, i, j, h, w)
        n, c = self.augdata(noise_crop.convert("RGB"), clean_crop.convert("RGB"))

        return n.float().cuda(), c.float().cuda()
class denoise_testset(data.Dataset):
    def __init__(self,path,sigma):
        super(denoise_testset,self).__init__()
        self.path = path
        self.files =[]
        dst = os.path.join(path,"noise"+sigma)

        diro =os.listdir(dst)          
        for i in diro:
            noise = os.path.join(dst,i)
            clean =os.path.join(path,'gr',i)

            self.files.append({"noise": noise, "gr": clean}) 
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):
        id = self.files[index]
        #print(id["noise"])
        # noise = Image.open(id["noise"])
        # clean = Image.open(id["gr"])
        noise = cv2.imread(id["noise"])
        clean = cv2.imread(id["gr"])
        
        if clean.shape[0]%2 ==1:
            noise = noise[:clean.shape[0]-1,:,:]
            clean = clean[:clean.shape[0]-1,:,:]
        if clean.shape[1]%2 ==1:
            noise =noise[:,:clean.shape[1]-1,:]
            clean =clean[:,:clean.shape[1]-1,:]

            
        """
        sobelx = cv2.Sobel(noise, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(noise, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        High_frequency = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        """
        n =ArrayToTensor(noise).permute((2,0,1))
        c = ArrayToTensor(clean).permute((2,0,1))

        return n.float().cuda(), c.float()     


class small_set(data.Dataset):
    def __init__(self, path, size=(256, 256)):
        super(small_set, self).__init__()
        self.size = size
        self.path = path
        #self.train = ['BSD', 'DIV2K']
        self.dirs = ['15', '25', '30', '50', '75']
        self.files = []
        for o in self.dirs:
            dst = os.path.join(path, "noise" + o)
            diro = os.listdir(dst)

            for i in diro:
                noise = os.path.join(dst, i)
                clean = os.path.join(path,'train', i)
                
                self.files.append({"noise": noise, "gr": clean})

       
           

    def __len__(self):
        return len(self.files)

    def augdata(self, data, target):
        rand_rot = random.randint(0, 3)
        data = FF.rotate(data, 90 * rand_rot)
        target = FF.rotate(target, 90 * rand_rot)
        #hr = FF.rotate(hr, 90 * rand_rot)

        data = np.array(data)
        target = np.array(target)
        #hr = np.array(hr)

        # cv2.imshow('a',data)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(data.shape)

        return ArrayToTensor(data).permute((2, 0, 1)), ArrayToTensor(target).permute((2, 0, 1))

    def __getitem__(self, index):
        id = self.files[index]
        #print(id["noise"])
        # noise = Image.open(id["noise"])
        # clean = Image.open(id["gr"])
        noise = cv2.imread(id["noise"])
        clean = cv2.imread(id["gr"])
        """
        sobelx = cv2.Sobel(noise, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(noise, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)
        High_frequency = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        """
        noise = Image.fromarray(noise)
        clean = Image.fromarray(clean)
        #hr = Image.fromarray(High_frequency)
        i, j, h, w = tfs.RandomCrop.get_params(noise, self.size)

        noise_crop = FF.crop(noise, i, j, h, w)
        clean_crop = FF.crop(clean, i, j, h, w)
        #hr_crop = FF.crop(hr, i, j, h, w)
        n, c = self.augdata(noise_crop.convert("RGB"), clean_crop.convert("RGB"))

        return n.float().cuda(), c.float().cuda()        
   
def  denoiseloss(src,target,ssim = False):
    if ssim == False:
        return torch.mean(abs(src-target))
    else:
        l1_loss = torch.mean(abs(src-target))/255.
        (_, channel, _, _) = target.size()
        window = create_window(11, channel).to(src.device).type(src.dtype)
        ssim_loss = 1 - Ssim(src, target )
        return (l1_loss + ssim_loss,l1_loss,ssim_loss) 
           



def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 

def Ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
 
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
        
    return ret
     

   
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f}({:.3f})'.format(self.val, self.avg)


def validate(model, path,eval_loader):
       
    batch_time = AverageMeter()  
    model.eval()
    end = time.time()
    MSE=0
    average_MSE=0
    sum=0
    for i,(img,ground_truth) in enumerate(eval_loader):
        output=np.squeeze(model(img).float().cpu().detach().numpy(),0).transpose(1,2,0)
        ground_truth=np.squeeze(ground_truth.float().numpy(),0).transpose(1,2,0)
   
        path=r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/CRG{}.jpg'.format(i)
        #print(output)
        #cv2.imwrite(path,output)          
        #r, g, b = cv2.split(output)
        #output = cv2.merge([b, g, r])
        #cv2.imshow('a',np.squeeze(output,0).transpose(1,2,0))
        
        
        mse = np.mean( (output - ground_truth) ** 2 )
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        psnr=20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        MSE+=mse
        sum+=1
        average_MSE = MSE/sum   
        print(average_MSE)
        print("PSNR: {}".format(psnr)) 
    return MSE/sum

def ArrayToTensor(array):
    assert type(array) is np.ndarray
    return torch.from_numpy(array)

#a = MyDataset(txt='label.txt')
