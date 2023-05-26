import argparse

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument(
    '-data_path',
    type=str,
    default='./data',
    help='The path of segmentation data')
    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    opt = parser.parse_args()

    return opt
