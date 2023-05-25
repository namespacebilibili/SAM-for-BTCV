import torch
import torch.nn as nn
import tqdm
from monai.losses import DiceCELoss
from einops import rearrange
import torchvision
import torchvision.transforms as transforms
import os


def validation(args, val_dataset, epoch, net: nn.Module):
    net.eval()
    batch_num = len(val_dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    total_loss = 0
    with tqdm(total=batch_num, desc="validation", unit="batch", leave=False) as pbar:
        for idx, data in enumerate(val_dataset):
            imgs = data["image"].to(dtype=torch.float32, device=device)
            masks = data["label"].to(dtype=torch.float32, device=device)
            if "pt" not in data.keys():
                imgs, pt, masks = generate_prompt(imgs, masks)
            else:
                pts = data["pt"]
                # point_labels = data['point_label']
            names = data["image_meta_dict"]["filename_or_obj"]
            cur = 0
            chunk = args.chunk
            while (cur + chunk) <= imgs.size(-1):
                pt = pts[:, :, cur : cur + chunk]
                img = imgs[..., cur : cur + chunk]
                mask = masks[..., cur : cur + chunk]
                pt = rearrange(pt, "b n d -> (b d) n")
                img = rearrange(img, "b c d h w -> (b d) c h w")
                mask = rearrange(mask, "b c d h w -> (b d) c h w")
                img = img.repeat(1, 3, 1, 1)
                point_labels = torch.ones(img.size(0))
                imgs = torchvision.transforms.Resize(
                    (args.image_size, args.image_size)
                )(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(
                    masks
                )
                bd, c, w, h = imgs.size()
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

                    if idx % args.vis == 0:
                        for name in names:
                            img_name = name.split("/")[-1].split(".")[0]
                            name = f"Test{img_name}+"
                        vis_image(
                            img,
                            pred,
                            mask,
                            os.path.join(
                                args.path_helper["sample_path"],
                                f"{name}epoch+{str(epoch)}.jpg",
                            ),
                            reverse=False,
                            points=show_pt,
                        )
            pbar.update()
            batch_num = batch_num * (imgs.size(-1) // chunk)
            return total_loss / batch_num
