import os
import sys
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision
import torchvision.transforms as transforms
from skimage import io
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable
from PIL import Image
# from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from dataloader import *
import time
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from utils import *
import function

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)

net.eval()
total_loss = function.validation(args, nice_test_loader, net)
# logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')

