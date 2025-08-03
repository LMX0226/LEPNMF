import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    device = torch.device('cuda')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--path', type=str, default='./HMDD3')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--GATf', type=int, default=256)
    parser.add_argument('--GATh', type=int, default=4)
    parser.add_argument('--head', type=int, default=4)
    parser.add_argument('--MLPDropout', type=float, default=0.3)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--layernum', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='mydata/')
    return parser.parse_args()