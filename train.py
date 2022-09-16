import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from utils import losses
from utils.utils import *
from Model.model_parts import SpatialTransformer


def Train(train_loader, model, device, optimizer, global_step, epoch, cfg, writer=None):
    '''Training
        Args:
            train_loader (torch.utils.data): training data loader
            model (torch.tensor): 3 cascaded UNet architecture 
            device (torch.device): CPU or GPU
            optimizer (torch.optim): Adam
            global_step (int): Steps
            epoch (int): current epoch
        Returns:
                global_step (int): Updated global steps after training
    '''

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

    with tqdm(total=len(train_loader)*cfg.train.batch_size, desc=f'Epoch {epoch + 1}/{cfg.train.epochs}', unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['img'].to(device)
            imgs = rotation_img(imgs)
            
            A, Aτ, τ, S = model(imgs)

            mseloss = MSELoss((Aτ[0]).repeat(imgs.shape[0], 1, 1, 1), Aτ) 
            smloss = 2*Smooth(τ, τ)
            rloss = L1Loss(τ)

            # low_matrix: Low-rank matrix  of Aτ + S
            low_matrix = torch.cat([torch.flatten(Aτ[i, 0] + S[i, 0]).unsqueeze(0) for i in range(imgs.shape[0])], dim = 0)
            loss_low = NuclearLoss(low_matrix)
            # Nτ : warp noise, low_matrix: Low-rank matrix  of Aτ + S
            Nτ = Noise_warp(τ, imgs, A, spatial, device) 
            loss_noise = NoiseLoss(Nτ, imgs.median(), MSELoss, device)
            loss_sp = L1Loss(S)
            loss_def = 10*mseloss + 0.1*smloss + 0.000001*rloss
            loss = 0.001*loss_low + 50*loss_noise + 1e-4*loss_sp + loss_def

            writer.add_scalar('Loss/train', loss.item(), global_step)
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(imgs.shape[0])
            global_step += 1

            # visualization of result
            if global_step % 100 == 0:
                for i in range(imgs.shape[0]):
                    writer.add_image(f"Input/I{i}", imgs[i], global_step)  
                    writer.add_image(f"Nτ/Nτ{i}", Nτ[i], global_step)     
                    writer.add_image(f"Aτ/Aτ{i}", Aτ[i], global_step)     
                    writer.add_image(f"S/S{i}", S[i], global_step)
                    writer.add_image(f"Aτ+S/Aτ+S{i}", S[i] + Aτ[i], global_step) 

    return global_step
