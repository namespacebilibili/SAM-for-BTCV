import torch
import os

# from lucent.optvis.param.spatial import pixel_image, fft_image, init_image
# from lucent.optvis.param.color import to_valid_rgb
# from lucent.optvis import objectives, transform, param
# from lucent.misc.io import show
import cfg
import torch
from monai.transforms import (
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


from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    set_track_meta,
)

args = cfg.parse_args()
device = torch.device('cuda', args.gpu_device)

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
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_size, args.roi_size, args.chunk),
                pos=1,
                neg=1,
                num_samples=args.num_sample,
                image_key="image",
                image_threshold=0,
            ),
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
        num_workers=0,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=2, cache_rate=1.0, num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    set_track_meta(False)

    return train_loader, val_loader, train_transforms, val_transforms, datalist, val_files

print(args)
# train_loader, val_loader, train_transforms, val_transforms, datalist, val_files = get_decath_loader(args)