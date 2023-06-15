# train.py
#!/usr/bin/env	python3
import os
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score,confusion_matrix
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from conf import settings
import cfg
from utils import *
from dataloader import *
import os
from utils import *
from function import train_sam, validation
import gc


gc.collect()
torch.cuda.empty_cache()

args = cfg.parse_args()
if not os.path.exists(f'log1/{args.exp}'):
    os.makedirs(f'log1/{args.exp}')
sw1 = SummaryWriter(log_dir=f'log1/{args.exp}')

GPUdevice = torch.device('cuda', args.gpu_device)
print(GPUdevice)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

optimizer = torch.optim.Adam(net.mask_decoder.parameters(),lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# writer = SummaryWriter(log_dir=os.path.join(
#         settings.LOG_DIR, args.net, settings.TIME_NOW))
class_optimizer = torch.optim.Adam(net.mask_decoder.classifier_prediction_head.parameters(), lr=args.classifier_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
if not os.path.exists(f'checkpoint/{args.exp}'):
    os.makedirs(f'checkpoint/{args.exp}')
nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list = get_decath_loader(args)

best_loss = 1e9
net.train()
for epoch in range(args.epoch):
    total_loss = train_sam(args, nice_train_loader, net, optimizer, class_optimizer)
    sw1.add_scalar('train_loss',total_loss.item(),global_step=epoch)
    print("Epoch:",epoch)
    print(total_loss.item())
    if total_loss.item()<best_loss:
        best_loss = total_loss.item()
        torch.save(net.state_dict(),f'checkpoint/{args.exp}/{epoch}.pth')

sw1.close()

net.eval()
test_total_loss, (iou, dice) = validation(args, nice_test_loader, net)
sw1.add_scalar('validation_loss',test_total_loss)
sw1.add_scalar('iou',iou)
sw1.add_scalar('dice',dice)
print(test_total_loss,iou,dice)
