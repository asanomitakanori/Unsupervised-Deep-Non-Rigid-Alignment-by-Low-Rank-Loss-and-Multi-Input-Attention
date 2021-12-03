from kornia.geometry.transform.imgwarp import get_affine_matrix2d, get_rotation_matrix2d
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import hydra 
import unet_parts
from dataset import DataSet 
from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter
import cv2 
import os
import shutil
import losses
# import model as unet
import torchvision
from model_self_attention import Vox2d as Net
from itertools import chain
from seed import set_seed
from model_self_attention import SpatialTransformer
from image import *
from torchvision import datasets, transforms
import kornia
from test_selfatten import Test
from train_selfatten import train
from pickle_dump_load import *
from pathlib import Path
from sample_code import *
from torch.utils.data.dataset import Subset
from sample_load_test import * 
import time

#  default GPU
torch.cuda.set_device(0)
set_seed(0)

def worker_init_fn(worker_id):
    random.seed(worker_id)

config_path = "./config/config.yaml"
@hydra.main(config_name = "config", config_path = "config")
def main(cfg):
    # path = "/home/asanomi/mnist/"
    path = "/home/asanomi/mnist/self_attention/"
    if os.path.exists(path + "checkpoints"):
        shutil.rmtree(path + "checkpoints")
    writer = SummaryWriter(path + "checkpoints")



    global k
    global m
    m = 0
    k = 0

    # use hydra for importing hyperparameters
    lr = 1e-4
    Validation = cfg.parameters.Validation
    batch_size  = 8
    epoch = 10
    ln = cfg.loss_param.ln
    alpha = cfg.loss_param.alpha
    ls = 1./np.sqrt(400)

    print("lr : {}".format(lr))
    print("batch_size : {}".format(batch_size))
    print("epoch : {}".format(epoch))
    print("ln : {}".format(ln))
    print("alpha : {}".format(alpha))

    # create model
    model = Net(2, 2, batch_size)
    if torch.cuda.is_available():
        model.cuda()

    # read model path
    model_param_path  = "/home/asanomi/mnist/self_attention/initial"
    torch.save(model.state_dict(), model_param_path)

    # optimizer
    optimizer  = getattr(optim, cfg.optim.optimizer2)(model.parameters(), lr=lr)

    # nuclear loss and L1loss
    nucloss = losses.NuclearLoss(lw=torch.tensor(alpha*ln))
    l1loss = losses.L1Loss(lw=torch.tensor(alpha*ls))
    MSEloss = nn.MSELoss()

    early_stop = 0 
    x = 0
    dice = 0
    iou = 0
    memory = Path("/home/asanomi/mnist/self_attention/" + "result.txt")
    memory.touch() 
    test_id = 0

    for std in [0.1, 0.2, 0.3]:
        for trans_mrange in [0.05, 0.075, 0.1]:
        # for trans_mrange in [0.15, 0.2]:
        # for trans_mrange in [0.2]:
            # erase_size 欠損の大きさ
            # for erase_size in [5, 10, 20]:
            erase_size = 6
            trans_grid = 3

            test_dice = 0
            test_iou = 0
            test_dice_sparse = 0
            test_iou_sparse = 0
            # if (std == 0.2 and trans_mrange==0.05) or (std == 0.2 and trans_mrange==0.1):
            #     continue

            for cross_id in range(2):
                data_path = Path(f"/home/asanomi/MNISTdata/MNISTdata3/train{cross_id}/std{std}-erase{erase_size}-grid{trans_grid}-range{trans_mrange}/img/")
                trainval_dataset = MnistLoadTest(data_path)
                test_data_path = Path(f"/home/asanomi/MNISTdata/MNISTdata3/test{cross_id}/std{std}-erase{erase_size}-grid{trans_grid}-range{trans_mrange}/img/")
                test_loader = MnistLoadTest(test_data_path)
                subset3_indices = list(range(0, 500))
                test_loader = Subset(test_loader, subset3_indices)


                subset1_indices = list(range(0, 2000))
                subset2_indices = list(range(2000, 2500))
                trainloader = Subset(trainval_dataset, subset1_indices)
                valloader = Subset(trainval_dataset, subset2_indices)

                # if trans_mrange==0.15:
                #     index=0.1
                # elif trans_mrange==0.2:
                #     index=0.15            

                # if (std ==0.4 and trans_mrange==0.15) or (std ==0.4 and trans_mrange==0.2):
                #     model.load_state_dict(torch.load(path + "output/modelparam/crossid_{}_std_{}_trans_mrange_{}".format(cross_id, 0.4, index)))
                # elif (std ==0.6 and trans_mrange==0.15) or (std ==0.6 and trans_mrange==0.2):
                #     model.load_state_dict(torch.load(path + "output/modelparam/crossid_{}_std_{}_trans_mrange_{}".format(cross_id, 0.6, index)))
                # elif (std ==0.2 and trans_mrange==0.15) or (std ==0.2 and trans_mrange==0.2) :
                #     model.load_state_dict(torch.load(path + "output/modelparam/crossid_{}_std_{}_trans_mrange_{}".format(cross_id, 0.2, index)))
                # else:
                model.load_state_dict(torch.load(model_param_path))
          
                x = 0
                early_stop = 0 
                dice = 0
                iou = 0
                dice_sparse = 0
                iou_sparse = 0

                for i in range(epoch):
                    # k = train(trainloader, model, nucloss, l1loss, optimizer, k, MSEloss, std, writer)
                    m, loss_val, loss_val2 = Test(valloader, model, nucloss, l1loss, MSEloss,  m, cross_id, std, writer)
                    dice += loss_val[0]
                    iou += loss_val[1]
                    dice_sparse += loss_val[2]
                    iou_sparse += loss_val[3]
                    x += 1

                    if i == 0:
                        loss_val_max =  loss_val2
                        temp_parameter = model.state_dict()
                        param_out_path = path + "output/modelparam/crossid_{}_std_{}_trans_mrange_{}".format(cross_id, std, trans_mrange)
                        torch.save(temp_parameter, param_out_path)
                    else:
                        if loss_val_max > loss_val2:
                            loss_val_max =  loss_val2
                            early_stop = 0
                            temp_parameter = model.state_dict()
                            param_out_path = path + "output/modelparam/crossid_{}_std_{}_trans_mrange_{}".format(cross_id, std, trans_mrange)
                            torch.save(temp_parameter, param_out_path)
                        else:
                            early_stop +=1
                            if early_stop == 3:
                                number = np.linspace(0, i, i+1)
                                break

                model.load_state_dict(torch.load(path + "output/modelparam/crossid_{}_std_{}_trans_mrange_{}".format(cross_id, std, trans_mrange)))

                test_id, loss_test, loss_test2 = Test(test_loader, model, nucloss, l1loss, MSEloss, test_id, cross_id, std, writer, test=1)
                test_dice += loss_test[0]
                test_iou += loss_test[1]
                test_dice_sparse += loss_test[2]
                test_iou_sparse += loss_test[3]

                if cross_id == 1:
                    test_dice /= 2
                    test_iou /= 2
                    test_dice_sparse /=2
                    test_iou_sparse /= 2
                    with memory.open("a") as f:
                        print("crossid_{}_std_{}_trans_mrange_{}  :  Dice_mean:{} : IoU_mean:{}\n".format(cross_id, std, trans_mrange, test_dice, test_iou), file=f)
                print("cross_id_{}, std_{}, trans_mrange_{}".format(cross_id, std, trans_mrange))
    writer.close()
    return 0


if __name__ == '__main__':
     main()
