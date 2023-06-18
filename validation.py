import torch
import cfg
import torch.nn as nn
from tqdm import tqdm
from monai.losses.dice import DiceCELoss
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import os
from utils import *
from function import validation
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

gc.collect()
torch.cuda.empty_cache()

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
print(GPUdevice)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)

net.eval()
test_total_loss, (iou, dice), mDice, class_acc, total_mDice = validation(args, nice_test_loader, net)
print(test_total_loss, iou, dice)
print(mDice,class_acc)
print(total_mDice)
