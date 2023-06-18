# SAM-for-BTCV

SAM for BTCV dataset

## Setup

`./install.sh`; requirements in `env.yml`

## Run

`cfg.py` is all you need. For validation: `validation.py` and for train: `train.py`

validation可使用（注意use_box等参数需根据所用模型修改）
`python validation.py -net_scale='l'  -gpu_device=0 -net_ckpt='checkpint路径 -net='sam_pretrain' -use_box=True -vis_image=True`

参考checkpoint可见于`https://huggingface.co/Zhancun/SAM-for-BTCV`，有验证需求可以提出issue，会更新checkpoint
