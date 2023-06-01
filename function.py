import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses.dice import DiceCELoss
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import os
from prompt import generate_click_prompt, msk_preprocess
from utils import vis_image

def validation(args, val_dataset, net: nn.Module):
    """
    img: (b, c, h, w, d)
    mask: (b, t, c, h, w, d)
    pt: (b, t, n, d)
    """
    net.eval()
    batch_num = len(val_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    total_loss = 0
    with tqdm(total=batch_num, desc="validation", unit="batch", leave=False) as pbar:
        for idx, data in enumerate(val_dataset):
            imgs = data["image"].to(dtype=torch.float32, device=device)
            masks = data["label"].to(dtype=torch.float32, device=device)
            masks = msk_preprocess(masks)
            # repeat dim 0 of imgs t times:
            imgs = imgs.repeat(masks.size(1), 1, 1, 1, 1)
            # if "pt" not in data.keys():
            #     imgs, pts, masks = generate_prompt(imgs, masks)
            # else:
            #     pts = data["pt"]
            imgs, pts, masks = generate_prompt(imgs, masks)
            names = data["image_meta_dict"]["filename_or_obj"]
            cur = 0
            chunk = args.chunk
            while (cur + chunk) <= imgs.size(-1):
                pt = pts[:, :, cur : cur + chunk]
                img = imgs[..., cur : cur + chunk]
                mask = masks[..., cur : cur + chunk]
                pt = rearrange(pt, "b t n d -> (b d t) n")
                img = rearrange(img, "bt c h w d -> (bt d) c h w")
                mask = rearrange(mask, "b t c h w d -> (b d t) c h w")
                img = img.repeat(1, 3, 1, 1)
                point_labels = torch.ones(img.size(0))
                imgs = torchvision.transforms.Resize(
                    (args.image_size, args.image_size)
                )(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(
                    masks
                )
                # bd, c, w, h = imgs.size()
                idx += 1
                cur += chunk
                show_pt = pt
                point = pt
                point = torch.as_tensor(point, device=device, dtype=torch.float32)
                point_labels = torch.as_tensor(point_labels, device=device, dtype=int)
                pt = (point, point_labels)
                img = img.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    img_emb = net.image_encoder(img)
                    sparse_emb, dense_emb = net.prompt_encoder(
                        points=pt, boxes=None, masks=None
                    )
                    pred, _ = net.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    total_loss += loss(pred, mask)
                # with torch.no_grad():
                #     prediction = [[[] for _ in range(pt[0].size(1))] for _ in range(img.size(0))]
                #     for i in range(img.size(0)):
                #         image = img[i]
                #         prompt = (pt[0][i], pt[1][i])
                #         msk = mask[i]
                #         img_emb = net.image_encoder(image)
                #         dice_loss = 0
                #         type_num = 0
                #         for type in range(prompt[0].size(0)):
                #             if prompt[0][type] == torch.tensor([-1,-1],device=device,dtype=torch.float32):
                #                 continue
                #             type_num += 1
                #             sparse_emb, dense_emb = net.prompt_encoder(
                #                 points=prompt, boxes=None, masks=None
                #             )
                #             pred, _ = net.mask_decoder(
                #                 image_embeddings=img_emb,
                #                 image_pe=net.prompt_encoder.get_dense_pe(),
                #                 sparse_prompt_embeddings=sparse_emb,
                #                 dense_prompt_embeddings=dense_emb,
                #                 multimask_output=False,
                #             )
                #             dice_loss += loss(pred, msk)
                #             prediction[i][type] = pred

                #         dice_loss /= type_num
                #         total_loss += dice_loss

                    if idx % args.vis == 0:
                        for name in names:
                            img_name = name.split("/")[-1].split(".")[0]
                            name = f"Test{img_name}+"
                        vis_image(
                            img,
                            pred,
                            mask,
                            f"{name}.jpg",
                            reverse=False,
                            points=show_pt,
                        )
            pbar.update()
    batch_num *= imgs.size(-1) // chunk
    return total_loss / batch_num

def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
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
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['label'].to(dtype = torch.float32, device = GPUdevice)
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
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)
            
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
