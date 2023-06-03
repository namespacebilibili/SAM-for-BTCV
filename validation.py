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
# def validation(args, val_dataset, net: nn.Module):
#     """
#     img: (b, c, h, w, d)
#     mask: (b, t, c, h, w, d)
#     pt: (b, t, n, d)
#     """
#     net.eval()
#     batch_num = len(val_dataset)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
#     total_loss = 0
#     with tqdm(total=batch_num, desc="validation", unit="batch", leave=False) as pbar:
#         for idx, data in enumerate(val_dataset):
#             imgs = data["image"].to(dtype=torch.float32, device=device)
#             masks = data["label"].to(dtype=torch.float32, device=device)
#             # masks = msk_preprocess(masks)
#             # repeat dim 0 of imgs t times:
#             # imgs = imgs.repeat(masks.size(1), 1, 1, 1, 1)
#             # if "pt" not in data.keys():
#             #     imgs, pts, masks = generate_prompt(imgs, masks)
#             # else:
#             #     pts = data["pt"]
#             imgs, pts, masks = generate_click_prompt(imgs, masks)
#             names = data["image_meta_dict"]["filename_or_obj"]
#             cur = 0
#             chunk = args.chunk
#             while (cur + chunk) <= imgs.size(-1):
#                 pt = pts[:, :, cur : cur + chunk]
#                 img = imgs[..., cur : cur + chunk]
#                 mask = masks[..., cur : cur + chunk]
#                 pt = rearrange(pt, "b n d -> (b d) n")
#                 img = rearrange(img, "b c h w d -> (b d) c h w")
#                 mask = rearrange(mask, "b c h w d -> (b d) c h w")
#                 img = img.repeat(1, 3, 1, 1)
#                 point_labels = torch.ones(img.size(0))
#                 img = torchvision.transforms.Resize(
#                     (args.image_size, args.image_size)
#                 )(img)
#                 mask = torchvision.transforms.Resize((args.out_size, args.out_size))(
#                     mask
#                 )
#                 # bd, c, w, h = imgs.size()
#                 idx += 1
#                 cur += chunk
#                 show_pt = pt
#                 point = pt
#                 point = torch.as_tensor(point, device=device, dtype=torch.float32)
#                 point_labels = torch.as_tensor(point_labels, device=device, dtype=int)
#                 point,point_labels = point[None,:,:],point_labels[None,:]
#                 pt = (point, point_labels)
#                 img = img.to(device=device, dtype=torch.float32)
#                 with torch.no_grad():
#                     img_emb = net.image_encoder(img)
#                     sparse_emb, dense_emb = net.prompt_encoder(
#                         points=pt, boxes=None, masks=None
#                     )
#                     pred, _ = net.mask_decoder(
#                         image_embeddings=img_emb,
#                         image_pe=net.prompt_encoder.get_dense_pe(),
#                         sparse_prompt_embeddings=sparse_emb,
#                         dense_prompt_embeddings=dense_emb,
#                         multimask_output=False,
#                     )
#                     total_loss += loss(pred, mask)
#                 # with torch.no_grad():
#                 #     prediction = [[[] for _ in range(pt[0].size(1))] for _ in range(img.size(0))]
#                 #     for i in range(img.size(0)):
#                 #         image = img[i]
#                 #         prompt = (pt[0][i], pt[1][i])
#                 #         msk = mask[i]
#                 #         img_emb = net.image_encoder(image)
#                 #         dice_loss = 0
#                 #         type_num = 0
#                 #         for type in range(prompt[0].size(0)):
#                 #             if prompt[0][type] == torch.tensor([-1,-1],device=device,dtype=torch.float32):
#                 #                 continue
#                 #             type_num += 1
#                 #             sparse_emb, dense_emb = net.prompt_encoder(
#                 #                 points=prompt, boxes=None, masks=None
#                 #             )
#                 #             pred, _ = net.mask_decoder(
#                 #                 image_embeddings=img_emb,
#                 #                 image_pe=net.prompt_encoder.get_dense_pe(),
#                 #                 sparse_prompt_embeddings=sparse_emb,
#                 #                 dense_prompt_embeddings=dense_emb,
#                 #                 multimask_output=False,
#                 #             )
#                 #             dice_loss += loss(pred, msk)
#                 #             prediction[i][type] = pred

#                 #         dice_loss /= type_num
#                 #         total_loss += dice_loss

#                     if idx % args.vis == 0:
#                         for name in names:
#                             img_name = name.split("/")[-1].split(".")[0]
#                             name = f"Test{img_name}+"
#                         vis_image(
#                             img,
#                             pred,
#                             mask,
#                             f"{name}.jpg",
#                             reverse=False,
#                             points=show_pt,
#                         )
#             pbar.update()
#     batch_num *= imgs.size(-1) // chunk
#     return total_loss / batch_num

gc.collect()
torch.cuda.empty_cache()

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
print(GPUdevice)
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

nice_train_loader, nice_test_loader, transform_train, transform_val, train_list, val_list =get_decath_loader(args)

net.eval()
total_loss = validation(args, nice_test_loader, net)
print(total_loss)
