import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses.dice import DiceCELoss
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import os
from prompt import generate_multi_resize_prompt, msk_preprocess, generate_resize_prompt,generate_prompt, msk_label_preprocess
from utils import *
from torch.nn.functional import normalize,threshold
import numpy as np

name = {0: "background", 1: "spleen", 2:"right_kidney", 3:"left_kidney", 4:"gallbladder", 5:"esophagus", 6:"liver", 7:"stomach", 8:"aorta", 9:"inferior_vena_cava", 10:"portal_vein_and_splenic_vein", 11:"pancreas", 12:"right_adrenal_gland", 13:"left_adrenal_gland"}
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
    class_loss = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_class_loss = 0.0
    num = 0
    thre = (0.1, 0.3, 0.5, 0.7, 0.9)
    mix_res = (0, 0, 0, 0)

    right_num = 0
    all_item_num = 0
    with tqdm(total=batch_num, desc="validation", unit="batch", leave=False) as pbar:
        for idx, data in enumerate(val_dataset):
            # imgsw: (b c h w d), mskw: (b c h w d)
            idx_0 = idx
            imgsw = data["image"].to(dtype=torch.float32, device=device)
            masksw = data["label"].to(dtype=torch.float32, device=device)
            masksw, labelsw = msk_label_preprocess(masksw) # (b t c h w d) (b t d t)
            cur = 0
            chunk = args.chunk
            while (cur + chunk) <= imgsw.size(-1):
                imgs = imgsw[:,:,:,:, cur: cur + chunk]
                masks = masksw[:,:,:,:,:, cur: cur + chunk]
                labels = labelsw[:,:,cur: cur + chunk, :]
                cur += chunk
                imgs = rearrange(imgs, "b c h w d -> (b d) c h w")
                masks = rearrange(masks, "b t c h w d -> (b d) t c h w")
                labels = rearrange(labels, "b t d n -> (b d) t n")

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
                # labels: (bd, t, t)
                for i in range(imgs.size(0)): # iter over b*d
                    img = imgs[i]
                    mask = masks[i]
                    pt = pts[i]
                    label = labels[i]
                    if not args.use_box:
                        pt_label = pt_labels[i]
                    able = ables[i]
                    type_num = len(able)
                    if not able:
                        continue
                    num += 1
                    # print(type_num)
                    idx += 1
                    label_able = label[able]
                    mask_able = mask[able]
                    pt_able = pt[able]
                    if not args.use_box:
                        point_label = pt_label[able]
                    # if args.use_multi:
                    #     point_label = torch.ones((type_num,args.multi_num))
                    # else:
                    #     point_label = torch.ones(type_num)
                    img_loss = 0.0
                    img_class_loss = 0.0
                    for j in range(type_num): # iter over type
                        mask_use = mask_able[j].unsqueeze(0)
                        img_use = img.unsqueeze(0)
                        pt_use = pt_able[j].unsqueeze(0)
                        label_use = label_able[j].unsqueeze(0)
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
                            pred, _, class_pred = net.mask_decoder(
                                image_embeddings=img_emb,
                                image_pe=net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb,
                                multimask_output=False,
                            )
                            binary_pred = normalize(threshold(pred,0.0,0)).to(device)
                            type_loss = loss(binary_pred, mask_use)
                            # print("type_loss:",type_loss)
                            class_pred = class_pred.squeeze(0)
                            label_use = label_use.squeeze(0)
                            if torch.argmax(class_pred).item() == label_use.item():
                                right_num += 1
                            class_loss_cal = class_loss(class_pred, label_use.to(device))
                            img_loss += type_loss
                            img_class_loss += class_loss_cal
                            temp = eval_seg(pred, mask_use, thre)
                            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                            # memory = torch.cuda.memory_allocated()
                            # print(f"Current GPU memory usage: {memory / 1024**2:.2f} MB")
                            if idx % 1 == 0:
                                vis_image2(
                                    img_use,
                                    pred,
                                    mask_use,
                                    f"./image2/{idx_0}_{idx}_{name[label_use.item()+1]}.jpg",
                                    reverse=False,
                                    points=None,
                                    use_box=args.use_box,
                                )
                    # print(f"img_loss:{img_loss},type_num:{type_num},img_loss/num:{img_loss/type_num}")
                    total_loss += (img_loss/type_num)
                    total_class_loss += (img_class_loss/type_num)
                    all_item_num += type_num
            pbar.update()
    mix_res = tuple([a/num for a in mix_res])
    print("class_acc: ",right_num/all_item_num)
    print(mix_res)
    return total_loss / num, mix_res



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
    class_loss = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    total_loss = 0.0
    total_class_loss = 0.0
    num = 0
    right_num = 0
    all_item_num = 0
    with tqdm(total=batch_num, desc="train", unit="batch", leave=False) as pbar:
        for idx, data in enumerate(train_dataset):
            # imgsw: (b c h w d), mskw: (b c h w d)
            idx_0 = idx
            imgsw = data["image"].to(dtype=torch.float32, device=device)
            masksw = data["label"].to(dtype=torch.float32, device=device)
            masksw, labelsw = msk_label_preprocess(masksw) # (b t c h w d) (b t d t)
            names = data["image_meta_dict"]["filename_or_obj"]
            cur = 0
            chunk = args.chunk
            while (cur + chunk) <= imgsw.size(-1):
                imgs = imgsw[:,:,:,:, cur: cur + chunk]
                masks = masksw[:,:,:,:,:, cur: cur + chunk]
                labels = labelsw[:,:,cur: cur + chunk, :]
                cur += chunk
                imgs = rearrange(imgs, "b c h w d -> (b d) c h w")
                masks = rearrange(masks, "b t c h w d -> (b d) t c h w")
                labels = rearrange(labels, "b t d n -> (b d) t n")

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
                    label = labels[i]
                    if not args.use_box:
                        pt_label = pt_labels[i]
                    able = ables[i]
                    type_num = len(able)
                    if not able:
                        continue
                    num += 1
                    idx += 1
                    mask_able = mask[able]
                    pt_able = pt[able]
                    label_able = label[able]
                    if not args.use_box:
                        point_label = pt_label[able]
                    # if args.use_multi:
                    #     point_label = torch.ones((type_num,args.multi_num))
                    # else:
                    #     point_label = torch.ones(type_num)
                    img_loss = 0.0
                    img_class_loss = 0.0
                    for j in range(type_num): # iter over type
                        mask_use = mask_able[j].unsqueeze(0)
                        img_use = img.unsqueeze(0)
                        pt_use = pt_able[j].unsqueeze(0)
                        label_use = label_able[j].unsqueeze(0)
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
                        with torch.no_grad():
                            if args.use_box:
                                sparse_emb, dense_emb = net.prompt_encoder(
                                    points=None, boxes=pt_use, masks=None
                                )
                            else:
                                sparse_emb, dense_emb = net.prompt_encoder(
                                    points=pt_use, boxes=None, masks=None
                                )

                        pred, _, class_pred = net.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                        )
                        #upscaled_masks = net.postprocess_masks(pred,(256,256),(1024,1024)).to(device)
                        binary_pred = normalize(threshold(pred,0.0,0)).to(device)
                        type_loss = loss(binary_pred, mask_use)
                        class_pred = class_pred.squeeze(0)

                        label_use = label_use.squeeze(0)
                        # print("class_pred : ", torch.argmax(class_pred) + 1)
                        # print("ground_label : ", label_use)
                        if torch.argmax(class_pred).item() == label_use.item():
                            right_num += 1
                        class_loss_cal = class_loss(class_pred, label_use.to(device))
                        # print("class_loss : ",class_loss_cal)
                        all_loss = type_loss + class_loss_cal
                        optimizer.zero_grad()
                        all_loss.backward()
                        optimizer.step()
                        # for name, para in net.mask_decoder.output_hypernetworks_mlps.named_parameters():
                        #     print(para.grad==0)
                        # print("type_loss:",type_loss)
                        img_loss += type_loss
                        img_class_loss += class_loss_cal
                        if idx_0 % 3 == 0:
                            vis_image(
                                img_use,
                                pred,
                                mask_use,
                                f"./train_image/{idx_0}_{idx}_{name[label_use.item()+1]}.jpg",
                                reverse=False,
                                points=show_pt,
                                use_box=args.use_box,
                            )
                    total_loss += (img_loss/type_num)
                    total_class_loss += (img_class_loss / type_num)
                    all_item_num += type_num
                # optimizer.step()
                # optimizer.zero_grad()

            pbar.update()
    # batch_num *= (imgsw.size(-1)//chunk)
    print("class_acc:",right_num/all_item_num)
    print("total_class_loss:", total_class_loss)
    return total_loss / num
