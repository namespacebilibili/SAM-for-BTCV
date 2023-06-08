import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses.dice import DiceCELoss
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import os
from prompt import generate_multi_resize_prompt, msk_preprocess, generate_resize_prompt
from utils import vis_image

def validation(args, val_dataset, net: nn.Module):
    """
    img: (b, c, h, w, d) -> (bd, c, h, w)
    mask: (b, c, h, w, d) -> (b, t, c, h, w, d) -> (bd, t, c, h, w)
    pt: (b*d, t, n, )
    """
    net.eval()
    batch_num = len(val_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    total_loss = 0
    with tqdm(total=batch_num, desc="validation", unit="batch", leave=False) as pbar:
        for idx, data in enumerate(val_dataset):
            # imgsw: (b c h w d), mskw: (b c h w d)
            imgsw = data["image"].to(dtype=torch.float32, device=device)
            masksw = data["label"].to(dtype=torch.float32, device=device)
            masksw = msk_preprocess(masksw) # (b t c h w d)
            names = data["image_meta_dict"]["filename_or_obj"]
            cur = 0
            chunk = args.chunk
            while (cur + chunk) <= imgsw.size(-1):
                imgs = imgsw[:,:,:,:, cur: cur + chunk]
                masks = masksw[:,:,:,:,:, cur: cur + chunk]
                cur += chunk
                imgs = rearrange(imgs, "b c h w d -> (b d) c h w")
                masks = rearrange(masks, "b t c h w d -> (b d) t c h w")
                imgs = imgs.repeat(1, 3, 1, 1)
                imgs = torchvision.transforms.Resize(
                    (args.image_size, args.image_size)
                )(imgs)

                bd, t, c, h, w = masks.size()
                temp_masks = torch.zeros((bd, t, c, args.image_size, args.image_size), device=device)
                for i in range(masks.size(0)):
                    for j in range(masks.size(1)):
                        temp_masks[i][j] = torchvision.transforms.Resize(
                            (args.image_size, args.image_size)
                        )(masks[i][j])
                masks = temp_masks

                if args.use_multi:
                    assert args.multi_size > 1
                    pts = generate_multi_resize_prompt(masks,args.multi_size) # (bd, t, k, n)
                else:
                    pts = generate_resize_prompt(masks) # (bd, t, n)

                bd, t, c, h, w = masks.size()
                temp_masks = torch.zeros((bd, t, c, args.out_size, args.out_size), device=device)
                for i in range(masks.size(0)):
                    for j in range(masks.size(1)):
                        temp_masks[i][j] = torchvision.transforms.Resize(
                            (args.out_size, args.out_size)
                        )(masks[i][j])
                masks = temp_masks

                # imgs: (bd, c, h, w), masks: (bd, t, c, h, w), pt: (bd, t, n)/(bd, t, k, n)
                for i in range(imgs.size(0)): # iter over b*d
                    img = imgs[i]
                    mask = masks[i]
                    pt = pts[i]
                    # img: (c, h, w), mask: (t, c, h, w), pt: (t, n)/(t, k, n)
                    if args.use_multi:
                        able = [
                            i
                            for i in range(pt.size()[0])
                            if not torch.allclose(
                                pt[i][args.multi_size - 1],
                                torch.tensor(
                                    [-1, -1], dtype=torch.float32
                                ),
                            )
                        ]
                    else:
                        able = [
                            i
                            for i in range(pt.size()[0])
                            if not torch.allclose(
                                pt[i],
                                torch.tensor(
                                    [-1, -1], dtype=torch.float32
                                ),
                            )
                        ]
                    type_num = len(able)
                    if not able:
                        continue
                    print(type_num)
                    idx += 1
                    mask_able = mask[able]
                    pt_able = pt[able]
                    if args.use_multi:
                        point_label = torch.ones((type_num,args.multi_size))
                    else:
                        point_label = torch.ones(type_num)
                    img_loss = 0
                    for j in range(type_num): # iter over type
                        mask_use = mask_able[j].unsqueeze(0)
                        img_use = img.unsqueeze(0)

                        pt_use = pt_able[j].unsqueeze(0)
                        point_label_use = point_label[j].unsqueeze(0)

                        show_pt = pt_use
                        point_use = pt_use
                        point_use = torch.as_tensor(point_use, device=device, dtype=torch.float32)
                        point_label_use = torch.as_tensor(point_label_use, device=device, dtype=int)
                        if not args.use_multi:
                            point_use = point_use[None,:,:]
                            point_label_use = point_label_use[None,:]
                        pt_use = (point_use,point_label_use)
                        img_use = img_use.to(device, dtype=torch.float32)
                        with torch.no_grad():
                            img_emb = net.image_encoder(img_use)
                            sparse_emb, dense_emb = net.prompt_encoder(
                                points=pt_use, boxes=None, masks=None
                            )
                            pred, _ = net.mask_decoder(
                                image_embeddings=img_emb,
                                image_pe=net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb,
                                multimask_output=False,
                            )
                            type_loss = loss(pred, mask_use)
                            img_loss += type_loss
                            memory = torch.cuda.memory_allocated()
                            print(f"Current GPU memory usage: {memory / 1024**2:.2f} MB")
                            if idx % 1 == 0:
                                # print(img.device,pred.device,mask.device)
                                vis_image(
                                    img_use,
                                    pred,
                                    mask_use,
                                    f"./image/{idx}_{j}.jpg",
                                    reverse=False,
                                    points=show_pt,
                                )
                        total_loss += (img_loss/type_num)
            pbar.update()
    batch_num *= (imgs.size(-1)//chunk)
    return total_loss / batch_num

def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):
    hard = 0
    epoch_loss = 0
    ind = 0
    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            if args.thd:
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w ')

                imgs = imgs.repeat(1,3,1,1)
                point_labels = torch.ones(imgs.size(0))

                imgs = torchvision.transforms.Resize((args.image_size,args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size,args.out_size))(masks)

            showp = pt

            mask_type = torch.float32
            ind += 1
            b_size,c,w,h = imgs.size()
            longsize = w if w >=h else h

            if point_labels[0] != -1:
                # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)

            '''init'''
            if hard:
                true_mask_ave = (true_mask_ave > 0.5).float()
                #true_mask_ave = cons_tensor(true_mask_ave)
            imgs = imgs.to(dtype=mask_type,device=GPUdevice)

            '''Train'''
            for n, value in net.image_encoder.named_parameters():
                if "Adapter" not in n:
                    value.requires_grad = False
            imge= net.image_encoder(imgs)

            with torch.no_grad():
                # imge= net.image_encoder(imgs)
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None,
                )
            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
            )

            loss = lossfunc(pred, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis:
                if ind % vis == 0:
                    namecat = 'Train'
                    for na in name:
                        namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                    vis_image(imgs,pred,masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)

            pbar.update()

    return loss
