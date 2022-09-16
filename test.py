import os
import sys

import torch
import torch.utils.data

import hydra 
from hydra.utils import to_absolute_path as abs_path

from utils.image import *
import utils.losses as losses
from utils.seed import set_seed
from utils.loader import MnistLoadTest
from utils.utils import *

from Model.model import LowRankNet
from Model.model_parts import SpatialTransformer


def Test(test_loader,
         model, 
         device
         ):
    '''Test
        Args:
            test_loader (torch.utils.data): test dataloader
            model (torch.tensor): 3 cascaded UNet architecture 
            device (torch.device): CPU or GPU
        Returns:
            Dice Score (torch.tensor): Average of dice score 
    '''
    model.eval()
    spatial = SpatialTransformer((64, 64), mode="nearest").to(device)

    # Diceloss
    diceloss = losses.Dice().loss
    dice_total = 0
    
    for step, batch in enumerate(test_loader):
        imgs = batch['img'].to(device)
        gt = batch['gt'].to(device)
        gt_sparse = batch['img_denoised'].to(device)
        gt = (gt - gt.min()) / (gt.max() - gt.min())   #min-max normalization
        gt_sparse =(gt_sparse - gt_sparse.min()) / (gt_sparse.max() - gt_sparse.min())  #min-max normalization

        with torch.no_grad():
            _, _, τ, S = model(imgs)
            τ = torch.cat([torch.zeros((1, 2, imgs.shape[2], imgs.shape[3])).to(device), τ], dim=0)
            S[S<0.1] = 0

        gts_τ = spatial(gt_sparse, τ)
        gts_τ += S
        dice = diceloss(gt[0].repeat(gts_τ.shape[0], 1, 1, 1), gts_τ)
        dice_total += float(dice)

    dice_total /= (step + 1)
    print(f"Dice Score:{dice_total}")


@hydra.main(config_path='config/config.yaml')
def main(cfg):
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LowRankNet(n_channels=1, n_classes=1)

    if cfg.test.load:
        model.load_state_dict(
            torch.load(abs_path(cfg.test.load), map_location=device)
        )
    else:
        assert f'Model parameter for test is not selected. Change test.load in config.yaml'

    model.to(device=device)

    # Test Dataloder
    test_loader = MnistLoadTest(abs_path(cfg.test.imgs))
    
    try:
        Test(test_loader=test_loader,
             model=model, 
             device=device,
             )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
     main()