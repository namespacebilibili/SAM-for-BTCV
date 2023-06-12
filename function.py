import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses.dice import DiceCELoss
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import os
from prompt import generate_multi_resize_prompt, msk_preprocess, generate_resize_prompt,generate_prompt
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

                pts, pt_labels, ables = generate_prompt(args,masks) # (bd, t, n)/(bd, t, k, n/(bd, t, 4), (bd, t)/(bd, t, k), (bd, t)
                bd, t, c, h, w = masks.size()
                temp_masks = torch.zeros((bd, t, c, args.out_size, args.out_size), device=device)
                for i in range(masks.size(0)):
                    for j in range(masks.size(1)):
                        temp_masks[i][j] = torchvision.transforms.Resize(
                            (args.out_size, args.out_size)
                        )(masks[i][j])
                masks = temp_masks

                # imgs: (bd, c, h, w), masks: (bd, t, c, h, w), pts: (bd, t, n)/(bd, t, k, n)/(bd, t, 4)
                for i in range(imgs.size(0)): # iter over b*d
                    img = imgs[i]
                    mask = masks[i]
                    pt = pts[i]
                    if not args.use_box:
                        pt_label = pt_labels[i]
                    able = ables[i]
                    type_num = len(able)
                    if not able:
                        continue
                    print(type_num)
                    idx += 1
                    mask_able = mask[able]
                    pt_able = pt[able]
                    if not args.use_box:
                        point_label = pt_label[able]
                    # if args.use_multi:
                    #     point_label = torch.ones((type_num,args.multi_num))
                    # else:
                    #     point_label = torch.ones(type_num)
                    img_loss = 0
                    for j in range(type_num): # iter over type
                        mask_use = mask_able[j].unsqueeze(0)
                        img_use = img.unsqueeze(0)
                        pt_use = pt_able[j].unsqueeze(0)
                        if not args.use_box:
                            point_label_use = point_label[j].unsqueeze(0)
                        show_pt = pt_use
                        point_use = pt_use
                        point_use = torch.as_tensor(point_use, device=device, dtype=torch.float32)
                        if not args.use_box:
                            point_label_use = torch.as_tensor(point_label_use, device=device, dtype=int)
                        if not (args.use_multi or args.use_pn or args.use_box):
                            point_use = point_use[None,:,:]
                            point_label_use = point_label_use[None,:]
                        if not args.use_box:
                            pt_use = (point_use,point_label_use)
                        else:
                            pt_use = point_use[None,:]
                        img_use = img_use.to(device, dtype=torch.float32)
                        with torch.no_grad():
                            img_emb = net.image_encoder(img_use)
                            if args.use_box:
                                sparse_emb, dense_emb = net.prompt_encoder(
                                    points=None, boxes=pt_use, masks=None
                                )
                            else:
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
                                    use_box=args.use_box,
                                )
                        total_loss += (img_loss/type_num)
            pbar.update()
    batch_num *= (imgs.size(-1)//chunk)
    return total_loss / batch_num



def train_sam(args, train_dataset, net: nn.Module, optimizer):
    """
    img: (b, c, h, w, d) -> (bd, c, h, w)
    mask: (b, c, h, w, d) -> (b, t, c, h, w, d) -> (bd, t, c, h, w)
    pt: (b*d, t, n, )
    """
    net.train()
    batch_num = len(train_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer.zero_grad()

    total_loss = 0
    with tqdm(total=batch_num, desc="train", unit="batch", leave=False) as pbar:
        for idx, data in enumerate(train_dataset):
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

                pts, pt_labels, ables = generate_prompt(args,masks) # (bd, t, n)/(bd, t, k, n/(bd, t, 4), (bd, t)/(bd, t, k), (bd, t)
                bd, t, c, h, w = masks.size()
                temp_masks = torch.zeros((bd, t, c, args.out_size, args.out_size), device=device)
                for i in range(masks.size(0)):
                    for j in range(masks.size(1)):
                        temp_masks[i][j] = torchvision.transforms.Resize(
                            (args.out_size, args.out_size)
                        )(masks[i][j])
                masks = temp_masks

                # imgs: (bd, c, h, w), masks: (bd, t, c, h, w), pts: (bd, t, n)/(bd, t, k, n)/(bd, t, 4)
                for i in range(imgs.size(0)): # iter over b*d
                    img = imgs[i]
                    mask = masks[i]
                    pt = pts[i]
                    if not args.use_box:
                        pt_label = pt_labels[i]
                    able = ables[i]
                    type_num = len(able)
                    if not able:
                        continue
                    print(type_num)
                    idx += 1
                    mask_able = mask[able]
                    pt_able = pt[able]
                    if not args.use_box:
                        point_label = pt_label[able]
                    # if args.use_multi:
                    #     point_label = torch.ones((type_num,args.multi_num))
                    # else:
                    #     point_label = torch.ones(type_num)
                    img_loss = 0
                    for j in range(type_num): # iter over type
                        mask_use = mask_able[j].unsqueeze(0)
                        img_use = img.unsqueeze(0)
                        pt_use = pt_able[j].unsqueeze(0)
                        if not args.use_box:
                            point_label_use = point_label[j].unsqueeze(0)
                        show_pt = pt_use
                        point_use = pt_use
                        point_use = torch.as_tensor(point_use, device=device, dtype=torch.float32)
                        if not args.use_box:
                            point_label_use = torch.as_tensor(point_label_use, device=device, dtype=int)
                        if not (args.use_multi or args.use_pn or args.use_box):
                            point_use = point_use[None,:,:]
                            point_label_use = point_label_use[None,:]
                        if not args.use_box:
                            pt_use = (point_use,point_label_use)
                        else:
                            pt_use = point_use[None,:]
                        img_use = img_use.to(device, dtype=torch.float32)
                        with torch.no_grad():
                            img_emb = net.image_encoder(img_use)
                            if args.use_box:
                                sparse_emb, dense_emb = net.prompt_encoder(
                                    points=None, boxes=pt_use, masks=None
                                )
                            else:
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
                        type_loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()
                        img_loss += type_loss
                        memory = torch.cuda.memory_allocated()
                        # print(f"Current GPU memory usage: {memory / 1024**2:.2f} MB")
                        if idx % 10 == 0:   # idx % 1
                            # print(img.device,pred.device,mask.device)
                            vis_image(
                                img_use,
                                pred,
                                mask_use,
                                f"./train_image/{idx}_{j}.jpg",
                                reverse=False,
                                points=show_pt,
                                use_box=args.use_box,
                            )
                        total_loss += (img_loss/type_num)
                # optimizer.step()
                # optimizer.zero_grad()

            pbar.update()
    batch_num *= (imgs.size(-1)//chunk)
    return total_loss / batch_num
