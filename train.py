# train.py
#!/usr/bin/env	python3
import os
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from models.discriminatorlayer import discriminator
from conf import settings
import time
import cfg
from torch.utils.data import random_split
from utils import *
import function 
from dataloader import *
import logging

args = cfg.parse_args()


GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

if args.pretrain:
    weights = torch.load(args.pretrain)
    net.load_state_dict(weights,strict=False)

optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay


# '''load pretrained model'''
# if args.weights != 0:
#     print(f'=> resuming from {args.weights}')
#     assert os.path.exists(args.weights)
#     checkpoint_file = os.path.join(args.weights)
#     assert os.path.exists(checkpoint_file)
#     loc = 'cuda:{}'.format(args.gpu_device)
#     checkpoint = torch.load(checkpoint_file, map_location=loc)
#     start_epoch = checkpoint['epoch']
#     best_tol = checkpoint['best_tol']
    
#     net.load_state_dict(checkpoint['state_dict'],strict=False)

#     path_helper = checkpoint['path_helper']
#     logger = create_logger(path_helper['log_path'])
#     print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')


'''segmentation data'''
transform_train = transforms.Compose([
    transforms.Resize((args.image_size,args.image_size)),
    transforms.ToTensor(),
])

transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
])

transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size,args.out_size)),
    transforms.ToTensor(),
])




nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)


'''checkpoint path and tensorboard'''
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#use tensorboard
if not os.path.exists(settings.LOG_DIR):
    os.mkdir(settings.LOG_DIR)
writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
#create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
for epoch in range(settings.EPOCH):
    if args.mod == 'sam_adpt':
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logging.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logging.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True

            #     save_checkpoint({
            #     'epoch': epoch + 1,
            #     'model': args.net,
            #     'state_dict': sd,
            #     'optimizer': optimizer.state_dict(),
            #     'best_tol': best_tol,
            #     'path_helper': args.path_helper,
            # }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint")
            else:
                is_best = False

writer.close()
