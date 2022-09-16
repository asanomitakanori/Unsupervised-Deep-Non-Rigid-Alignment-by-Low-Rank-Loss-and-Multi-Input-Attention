import torch
import torch.nn as nn
import torch.utils.data

import numpy as np

import utils.losses as losses

from utils.utils import *
from Model.model_parts import SpatialTransformer


def Val(val_loader, model, device, writer):
    '''Validation
        Args:
            val_loader (torch.utils.data): validation data loader
            model (torch.tensor): 3 cascaded UNet architecture 
            device (torch.device): CPU or GPU
        Returns:
            loss_Total (torch.tensor): Total loss of validation data
    '''
    
    model.eval()
    spatial = SpatialTransformer((64, 64), mode="nearest").to(device)

    # Loss setting
    ls = 1./np.sqrt(400)
    alpha = 0.5
    ln = 0.5
    NuclearLoss = losses.NuclearLoss(lw=torch.tensor(alpha*ln))
    L1Loss = losses.L1Loss(lw=torch.tensor(alpha*ls))
    MSELoss = nn.MSELoss()
    NoiseLoss = losses.NoiseLoss().loss
    Smooth = losses.Grad().loss
    NoiseLoss = losses.NoiseLoss().loss
    loss_Total = 0


    for step, batch in enumerate(val_loader):
        imgs = batch['img'].to(device)

        with torch.no_grad():
            A, Aτ, τ, S = model(imgs)  

        # Nτ : warp noise, low_matrix: Low-rank matrix 
        Nτ = Noise_warp(τ, imgs, A, spatial, device) 
        low_matrix = torch.cat([torch.flatten(Aτ[i, 0] + S[i, 0]).unsqueeze(0) for i in range(imgs.shape[0])], dim = 0)

        # Loss 
        mseloss = MSELoss((Aτ[0]).repeat(imgs.shape[0], 1, 1, 1), Aτ) 
        smloss = 2*Smooth(τ, τ)
        rloss = L1Loss(τ)
        loss_low = NuclearLoss(low_matrix)
        loss_noise = NoiseLoss(Nτ, imgs.median(), MSELoss, device)
        loss_sp = L1Loss(S)
        loss_def = 10*mseloss + 0.1*smloss + 0.000001*rloss
        loss = 0.001*loss_low + 50*loss_noise + 1e-4*loss_sp + loss_def
        loss_Total += float(loss)

    return loss_Total
