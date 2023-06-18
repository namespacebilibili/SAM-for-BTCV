import sys

import numpy

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import random
import math
import PIL
import matplotlib.pyplot as plt
import seaborn as sns

import collections
import logging
import math
import os
import time
from datetime import datetime

import dateutil.tz

from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
import warnings
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
# from lucent.optvis.param.spatial import pixel_image, fft_image, init_image
# from lucent.optvis.param.color import to_valid_rgb
# from lucent.optvis import objectives, transform, param
# from lucent.misc.io import show
from torchvision.models import vgg19
import torch.nn.functional as F
import cfg

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)



args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)


def get_network(args, net, use_gpu=True, gpu_device=0, distribution=True):
    """ return given network
    """

    if net == 'sam':
        from segment_anything import SamPredictor, sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        if args.net_scale == 'b':
            net = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt).to(device)
        elif args.net_scale == 'h':
            net = sam_model_registry['vit_h'](checkpoint=args.sam_ckpt).to(device)
        elif args.net_scale == 'l':
            net = sam_model_registry['vit_l'](checkpoint=args.sam_ckpt).to(device)
    elif net == 'sam_with_classifier':
        from segment_anything import SamPredictor, sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        from segment_anything import build_sam_vit_b_classifier,build_sam_vit_h_classifier,build_sam_vit_l_classifier
        if args.net_scale == 'b':
            net = build_sam_vit_b_classifier()
        elif args.net_scale == 'h':
            net = build_sam_vit_h_classifier()
        elif args.net_scale == 'l':
            net = build_sam_vit_l_classifier()
        net.to(device)
        net_dict = net.state_dict()
        sam_net = torch.load(args.sam_ckpt)
        state_dict = {k: v for k, v in sam_net.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        net.load_state_dict(net_dict)
        for m in net.mask_decoder.classifier_prediction_head.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean = 0, std = 0.01)
                m.bias.data.fill_(0.0)
    elif net == 'sam_pretrain':
        from segment_anything import SamPredictor, sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        from segment_anything import build_sam_vit_b_classifier,build_sam_vit_h_classifier,build_sam_vit_l_classifier
        if args.net_scale == 'b':
            net = build_sam_vit_b_classifier()
        elif args.net_scale == 'h':
            net = build_sam_vit_h_classifier()
        elif args.net_scale == 'l':
            net = build_sam_vit_l_classifier()
        net.to(device)
        net.load_state_dict(torch.load(args.net_ckpt))

    
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


def get_decath_loader(args):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )



    data_dir = args.data_path
    split_JSON = "dataset_0.json"

    datasets = os.path.join(data_dir, split_JSON)
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files



@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def iou(outputs: np.array, labels: np.array):

    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device=input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None,use_box=False):

    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if (points != None) and (not use_box):
            for i in range(b):
                if args.thd:
                    p = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                if args.use_multi or args.use_pn:
                    for j in range(args.multi_num):
                        gt_masks[i,0,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.5
                        gt_masks[i,1,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.1
                        gt_masks[i,2,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.4
                else:
                    gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                    gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                    gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
                if args.use_pn:
                    gt_masks[i,0,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.5
                    gt_masks[i,1,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.1
                    gt_masks[i,2,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.4
        elif (points != None) and use_box:
            for i in range(b):
                p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                left_up_x = p[i,0]
                left_up_y = p[i,1]
                right_down_x = p[i,2]
                right_down_y = p[i,3]
                # print(gt_masks[i].shape)
                # print(left_up_x, left_up_y, right_down_x, right_down_y)
                gt_masks[i,0,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.5
                gt_masks[i,1,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.1
                gt_masks[i,2,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.4
                gt_masks[i,0,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.5
                gt_masks[i,1,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.1
                gt_masks[i,2,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.4
                gt_masks[i,0,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.5
                gt_masks[i,1,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.1
                gt_masks[i,2,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.4
                gt_masks[i,0,left_up_x:right_down_x,right_down_y-4:right_down_y+4] = 0.5
                gt_masks[i,1,left_up_x:right_down_x,right_down_y-4:right_down_y+4] = 0.1
                gt_masks[i,2,left_up_x:right_down_x,right_down_y-4:right_down_y+2] = 0.4
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return

def vis_image2(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None,use_box=False):

    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if (points != None) and (not use_box):
            for i in range(b):
                p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                if args.use_multi or args.use_pn:
                    for j in range(args.multi_num):
                        gt_masks[i,0,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.5
                        gt_masks[i,1,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.1
                        gt_masks[i,2,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.4
                else:
                    gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                    gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                    gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
                if args.use_pn:
                    gt_masks[i,0,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.5
                    gt_masks[i,1,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.1
                    gt_masks[i,2,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.4
        elif (points != None) and use_box:
            for i in range(b):
                if args.thd:
                    p = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                left_up_x = p[i,0]
                left_up_y = p[i,1]
                right_down_x = p[i,2]
                right_down_y = p[i,3]
                # print(gt_masks[i].shape)
                # print(left_up_x, left_up_y, right_down_x, right_down_y)
                gt_masks[i,0,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.5
                gt_masks[i,1,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.1
                gt_masks[i,2,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.4
                gt_masks[i,0,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.5
                gt_masks[i,1,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.1
                gt_masks[i,2,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.4
                gt_masks[i,0,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.5
                gt_masks[i,1,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.1
                gt_masks[i,2,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.4
                gt_masks[i,0,left_up_x:right_down_x,right_down_y-4:right_down_y+4] = 0.5
                gt_masks[i,1,left_up_x:right_down_x,right_down_y-4:right_down_y+4] = 0.1
                gt_masks[i,2,left_up_x:right_down_x,right_down_y-4:right_down_y+2] = 0.4
        tup = gt_masks[:row_num,:,:,:]
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = tup
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return

def vis_image3(imgs, pred_masks, gt_masks, save_path, reverse=False, points=None,use_box=False):

    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2:
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat((pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if (points != None) and (not use_box):
            for i in range(b):
                p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                if args.use_multi or args.use_pn:
                    for j in range(args.multi_num):
                        gt_masks[i,0,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.5
                        gt_masks[i,1,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.1
                        gt_masks[i,2,p[i,j,0]-5:p[i,j,0]+5,p[i,j,1]-5:p[i,j,1]+5] = 0.4
                else:
                    gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                    gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                    gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
                if args.use_pn:
                    gt_masks[i,0,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.5
                    gt_masks[i,1,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.1
                    gt_masks[i,2,p[i,args.multi_num,0]-5:p[i,args.multi_num,0]+5,p[i,args.multi_num,1]-5:p[i,args.multi_num,1]+5] = 0.4
        elif (points != None) and use_box:
            for i in range(b):
                p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                left_up_x = p[i,0]
                left_up_y = p[i,1]
                right_down_x = p[i,2]
                right_down_y = p[i,3]
                # print(gt_masks[i].shape)
                # print(left_up_x, left_up_y, right_down_x, right_down_y)
                gt_masks[i,0,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.5
                gt_masks[i,1,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.1
                gt_masks[i,2,left_up_x:right_down_x,left_up_y-4:left_up_y+4] = 0.4
                gt_masks[i,0,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.5
                gt_masks[i,1,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.1
                gt_masks[i,2,left_up_x-4:left_up_x+4,left_up_y:right_down_y] = 0.4
                gt_masks[i,0,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.5
                gt_masks[i,1,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.1
                gt_masks[i,2,right_down_x-4:right_down_x+4,left_up_y:right_down_y] = 0.4
                gt_masks[i,0,left_up_x:right_down_x,right_down_y-4:right_down_y+4] = 0.5
                gt_masks[i,1,left_up_x:right_down_x,right_down_y-4:right_down_y+4] = 0.1
                gt_masks[i,2,left_up_x:right_down_x,right_down_y-4:right_down_y+2] = 0.4
        tup = pred_masks[:row_num,:,:,:]
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = tup
        vutils.save_image(compose, fp=save_path, nrow=row_num, padding=10)

    return
def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p[:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p[:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()

        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p[:,0,:,:].squeeze(1).cpu().numpy().astype('int32')

            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()

        return eiou / len(threshold), edice / len(threshold)
