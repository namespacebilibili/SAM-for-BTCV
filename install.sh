cd ~
git clone https://github.com/namespacebilibili/SAM-for-BTCV.git
cd SAM-for-BTCV
mkdir data
mkdir image
mkdir log1
mkdir train_image
mkdir runs
mkdir checkpoint
conda create --name myenv --file requirements.txt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
synapse get syn3379050
wget https://drive.google.com/file/d/1t4fIQQkONv7ArTSZe4Nucwkk1KfdUDvW/view?usp=sharing
