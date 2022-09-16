import logging
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

import hydra 
from hydra.utils import to_absolute_path as abs_path

from train import Train
from val import Val
from Model.model import LowRankNet
from utils.seed import set_seed
from utils.loader import MnistLoadTest


def train_net(model, 
          device, 
          cfg
          ):

    if cfg.eval.imgs is not None:
        train_loader = MnistLoadTest(abs_path(cfg.train.imgs))
        val_loader = MnistLoadTest(abs_path(cfg.eval.imgs))
        n_train = len(train_loader)
        n_val = len(train_loader)
    else:
        dataset = MnistLoadTest(abs_path(cfg.train.imgs))
        n_val = int(len(dataset) * cfg.eval.rate)
        n_train = len(dataset) - n_val
        train_loader, val_loader = random_split(dataset, [n_train, n_val])

    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    writer = SummaryWriter(log_dir=abs_path('./logs'), comment=f'LR_{cfg.train.lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Start training:
        Epochs:           {epochs}
        Batch size:       {batch_size}
        Learning rate:    {cfg.train.lr}
        Training size:    {n_train}
        Validation size:  {n_val}
        Device:           {device.type}
        Optimizer         {optimizer.__class__.__name__}
    '''
    )
    
    for epoch in range(epochs):
        global_step = Train(train_loader, model, device, optimizer, global_step, epoch, cfg, writer)
        # if global_step % n_train == 0:
        #     val_loss = Val(val_loader, model, device, writer)

        # logging.info('Validation loss: {}'.format(val_loss))
        # writer.add_scalar('Loss/test', val_loss, global_step)

        if cfg.output.save:
            try:
                os.mkdir(abs_path(cfg.output.dir))
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       abs_path(os.path.join(cfg.output.dir, f'CP_epoch{epoch + 1}.pth')))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


@hydra.main(config_path='config/config.yaml')
def main(cfg):
    set_seed(0)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LowRankNet(n_channels=1, n_classes=1)
    total_param = sum(p.numel() for p in model.parameters())
    logging.info(f'The number of model paramerter is {total_param}')
    logging.info(f'Using device {device}')

    if cfg.load:
        model.load_state_dict(
            torch.load(cfg.load, map_location=device)
        )
        logging.info(f'Model loaded from {cfg.load}')
    
    model.to(device=device)

    try:
        train_net(model=model, 
                  device=device,
                  cfg=cfg)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == '__main__':
     main()
