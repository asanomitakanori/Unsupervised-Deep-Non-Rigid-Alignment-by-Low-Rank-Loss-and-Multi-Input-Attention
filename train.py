import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

import losses

from utils.utils import *
from Model.model import SpatialTransformer

def Train(train_loader, model, device, optimizer, global_step, writer=None):
    model.train()
    spatial = SpatialTransformer((64, 64), mode="nearest").to(device=device)

    # Loss 
    ls = 1./np.sqrt(400)
    alpha = 0.5
    ln = 0.5
    NuclearLoss = losses.NuclearLoss(lw=torch.tensor(alpha*ln))
    L1Loss = losses.L1Loss(lw=torch.tensor(alpha*ls))
    MSELoss = nn.MSELoss()
    NoiseLoss = losses.NoiseLoss().loss
    Smooth = losses.Grad().loss

    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        imgs = batch['img'].to(device)
        imgs = rotation_img(imgs)
        
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
        loss = 0.001*loss_low + 100*loss_noise + 1e-4*loss_sp + loss_def

        loss.backward()
        optimizer.step()

        # tensorboard
        if global_step % 100 == 0:
            for i in range(imgs.shape[0]):
                writer.add_image(f"Input/I{i}", imgs[i], global_step)  
                writer.add_image(f"N/N{i}", Nτ[i], global_step)     
                writer.add_image(f"Aτ/Aτ{i}", Aτ[i], global_step)     
                writer.add_image(f"A/A{i}", A[i], global_step)    
                writer.add_image(f"S/S{i}", S[i], global_step)  
                writer.add_image(f"Aτ+S/Aτ+S{i}", S[i] + Aτ[i], global_step)  

        writer.add_scalar('Loss/train', loss.item(), global_step)
        global_step += 1

    return global_step
