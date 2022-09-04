import torch
import random
from Model.model import SpatialTransformer

def rotation_img(imgs):
    random_int = random.randint(0, imgs.shape[0] - 1)
    target = imgs[random_int].unsqueeze(0)
    target_source = torch.cat([target, imgs[0:random_int]])
    imgs = torch.cat([target_source, imgs[random_int+1::]])
    return imgs

def Noise_warp(τ, imgs, A, spatial, device):
    τ = torch.cat([torch.zeros((1, 2, τ.shape[2], τ.shape[3])).to(device=device), τ], dim=0)
    N = imgs - A
    Nτ = spatial(N, τ)
    Nτ = torch.where(Nτ==0, Nτ + imgs.median(), Nτ)
    return Nτ

