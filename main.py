import argparse, os, time
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import MINet as NET

from torch.autograd import Variable
from torch.utils.data import DataLoader

import create
from create import MyDataset, validate,AverageMeter,Hazydataset_OUT, Hazydataset_RESIDE, Raindataset
import math

'''
#fix cv2_python environment problem
import sys
for index in sys.path:
	if'/home/spl208_rtxtitan/anaconda3/envs/py36/lib/python3.6/site-packages' in index:
		sys.path.remove(index)
		sys.path.insert(0, index)
		break
'''

import cv2 


# Training settings
parser = argparse.ArgumentParser(description="Pytorch MI_Net")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=220, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate, Default=0.01")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default=5")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.001, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model, Default=None')
parser.add_argument("--output", default="", type=str, help='path to test result, Default=None')


def main():
	global opt, model
	opt = parser.parse_args()
	print(opt)
	cuda = opt.cuda
	os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
	if cuda  and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without --cuda")

	opt.seed = random.randint(1, 10000)
	print("Random Seed: ", opt.seed)

	cudnn.benchmark = True

	print("===> Loading datasets")


#import your training images or training set	
	trainset=Hazydataset_RESIDE(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/ITS_v2/hazy',r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/ITS_v2/clear')
	training_data_loader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize, shuffle=True)

	testset=Hazydataset_RESIDE(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/SOTS/nyuhaze500/hazy',r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/SOTS/nyuhaze500/gt')
	test_loader=torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)

	trainset_outside=Hazydataset_OUT(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/SOTS/outdoor/hazy',r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/SOTS/outdoor/gt')
	train_loader_outside=torch.utils.data.DataLoader(trainset_outside,batch_size=1,shuffle=True)

	testset_outside=Hazydataset_OUT(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/SOTS/outdoor/hazy',r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/SOTS/outdoor/gt')
	test_loader_outside=torch.utils.data.DataLoader(testset_outside,batch_size=1,shuffle=False)

	trainset_rain=Raindataset(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/derain/split_img/inp',r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/derain/split_img/gt')
	train_loader_rain=torch.utils.data.DataLoader(trainset_rain,batch_size=opt.batchSize, shuffle=True)

	testset_rain=Raindataset(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/derain/test/inp',r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/dataset/derain/test/gt')
	test_loader_rain=torch.utils.data.DataLoader(testset_rain,batch_size=1,shuffle=False)

	trainset_rain12=Raindataset(r'/home/spl208_rtxtitan/桌面/shenjw/dataset/rain12/inp',r'/home/spl208_rtxtitan/桌面/shenjw/dataset/rain12/gt')
	train_loader_rain12=torch.utils.data.DataLoader(trainset_rain12,batch_size=opt.batchSize, shuffle=True)

	testset_rain12=Raindataset(r'/home/spl208_rtxtitan/桌面/shenjw/dataset/rain12/inp',r'/home/spl208_rtxtitan/桌面/shenjw/dataset/rain12/gt')
	test_loader_rain12=torch.utils.data.DataLoader(testset_rain12,batch_size=1,shuffle=False)

	trainset_toy=Raindataset(r'/home/spl208_rtxtitan/桌面/shenjw/dataset/noise/inpt',r'/home/spl208_rtxtitan/桌面/shenjw/dataset/noise/gt')
	train_loader_toy=torch.utils.data.DataLoader(trainset_toy,batch_size=1, shuffle=True)

	testset_toy=Raindataset(r'/home/spl208_rtxtitan/桌面/shenjw/dataset/noise/inpt',r'/home/spl208_rtxtitan/桌面/shenjw/dataset/noise/gt')
	test_loader_toy=torch.utils.data.DataLoader(testset_toy,batch_size=1,shuffle=False)

	trainset_denoise =  create.denoisingset1(r'/home/spl208_rtxtitan/桌面/shenjw/denoise_dataset/DATASET/train',(500,500))
	train_loader_denoising =torch.utils.data.DataLoader(trainset_denoise,batch_size =6,shuffle = True)
	#print(len(trainset_denoise))

	
	print("===> Building model")
	model = NET.MI_NET()
	
	model = nn.DataParallel(model)
	#cudnn.benchmark = True
	criterion = nn.MSELoss(size_average=False)
	

	print("===> Setting GPU")
	if cuda:
		model = torch.nn.DataParallel(model).cuda()
		criterion = criterion.cuda()

	# optionally resume from a checkpoint
	
	if opt.resume:
		if os.path.isfile(opt.resume):			
			print("===> loading checkpoint: {}".format(opt.resume))
			nn.Module.dump_patches = True
			checkpoint = torch.load(opt.resume)
			opt.start_epoch = checkpoint["epoch"] + 1
			model.load_state_dict(checkpoint["model"].state_dict())
		else:
			print("===> no checkpoint found at {}".format(opt.resume))
	

	# optionally copy weights from a checkpoint
	if opt.pretrained:
		if os.path.isfile(opt.pretrained):
			print("===> load model {}".format(opt.pretrained))
			weights = torch.load(opt.pretrained)
			model.load_state_dict(weights['model'].state_dict())
		else:
			print("===> no model found at {}".format(opt.pretrained))
					
	model=model.cuda()		

	print("===> Setting Optimizer")
	

	optimizer=torch.optim.Adam(model.parameters(), opt.lr, betas=(0.9,0.999))	
	milestones=[i* opt.step for i in range(1,10)]
	scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.6 )


	

	print("===> Training")
	# training for dehazing or deraining
	
	save_checkpoint(model,0)
	
	for epoch in range(opt.start_epoch, opt.nEpochs):
		train(model,training_data_loader,optimizer,epoch)		
		save_checkpoint(model, epoch)		
		test(model,epoch,test_loader)
		# os.system("python eval.py --cuda --model=model/model_epoch_{}.pth".format(epoch))
		

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
	lr = opt.lr * (0.1 ** (epoch  // opt.step))
	return lr

    	
def save_checkpoint(model, epoch):
	model_out_path = "/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/toy/toy_model/" + "model_toy_epoch_{}.pth".format(epoch)
	state = {"epoch": epoch, "model": model}
	#check path status
	if not os.path.exists("toy/toy_model/"):
		os.makedirs("toy/toy_model/")
	#save model
	torch.save(state, model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))



def train(model,train_loader,optimizer,epoch):
	loss=AverageMeter()
	
	model.train()
	#lr = adjust_learning_rate(optimizer, epoch-1)
	#for param_group in optimizer.param_groups:
	#    param_group["lr"] = lr
	path='/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/toy/toy_circle_demo/'
	if not os.path.exists(path):
		os.makedirs(path)

	for i,(input,gr) in enumerate(train_loader):
		output=model(input.cuda())

		loss1=torch.norm(output.cuda()-gr.cuda())
		loss.update(loss1.item())

		optimizer.zero_grad()
		loss1.backward()

		optimizer.step()

		if i%50==0:
			lc_time = time.asctime( time.localtime(time.time()) )
			print("===> {} Epoch[{}]({}/{}):loss_avg:{}".format(lc_time, epoch, i,len(train_loader),loss.avg))
			file =open(path + 'Loss.txt','a')
			file.write("\n===> {} Epoch[{}]({}/{}):loss_avg:{} \n".format(lc_time, epoch, i,len(train_loader),loss.avg))
			file.close()
			
	

def test(model,epoch,test_loader):
	error=AverageMeter()
	model.eval()
	sum=0

	for i,(input,gr) in enumerate(test_loader):		 		
		output=np.squeeze(model(input.cuda()).float().cpu().detach().numpy(),0).transpose(1,2,0)		
		ground_truth=np.squeeze(gr.float().numpy(),0).transpose(1,2,0)		
		#ground_truth=ground_truth[10:470,10:630] #for SOTS_dehaze
		path=r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/toy/toycircle_demo/{}'.format(epoch)
		if not os.path.exists(path):
			os.makedirs(path)

		cv2.imwrite(os.path.join(path,str(i)+'.jpg'),output)
		mse = np.mean( (output - ground_truth) ** 2 )
		error.update(mse)
		if mse < 1.0e-10:
			return 100
		PIXEL_MAX = 255
		psnr=20 * math.log10(PIXEL_MAX / math.sqrt(mse))
		
		sum+=1
		#print("PSNR: {},i:{}".format(psnr,i))
		 

	avg_mse=error.avg

	psnr=20 * math.log10(255 / math.sqrt(avg_mse))
	file =open(r'/home/spl208_rtxtitan/anaconda3/envs/py36/project/shenjw/toy/toycircle_demo/{}/PSNR'.format(epoch),'w')
	file.write('EPOCH: {}  PSNR: {}'.format(epoch, psnr))
	print("avg_psnr:{}".format(psnr))	

if __name__ == "__main__":

	main()
