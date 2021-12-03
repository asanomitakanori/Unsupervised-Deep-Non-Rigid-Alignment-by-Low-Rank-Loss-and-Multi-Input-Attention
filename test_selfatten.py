import torch
import numpy as np
import torch.utils.data
import cv2 
from model_self_attention import SpatialTransformer
from image import *
import kornia
import losses
import matplotlib.pyplot as plt
import os
import time 


def Test(val_loader, model, nucloss, l1loss, MSE,  k, crossid, std, train_writer, test=0):
    model.eval()
    dir = "/home/asanomi/mnist/self_attention/output/"
    up1 = torch.nn.Upsample(size=((64, 64)), scale_factor=None, mode='nearest', align_corners=None,)    
    diceloss = losses.Dice().loss
    iouloss = losses.IoU().loss
    loss_dice = 0
    loss_IoU = 0
    loss_dice_sparse = 0
    loss_IoU_sparse = 0
    smooth = losses.Grad().loss
    loss_Total = 0
    device = torch.device('cpu')
    time_all = 0
    spatial = SpatialTransformer((64, 64), mode="nearest").to(device)
    for i, input_img in enumerate(val_loader):
        grandtruth = input_img[2].to(device)
        grandtruth = (grandtruth - grandtruth.min()) / (grandtruth.max() - grandtruth.min())
        input = input_img[0].to(device)

        # Use CNN Network
        if input.shape[0] == 1:
            i -= 1
            break
        start = time.time()
        with torch.no_grad():
            logits, Output, flow, complement = model(input)
        end = time.time()
        print((end - start), 'sec.')
        complement[complement<0.1] = 0
        # warp grandtruth 
        flow = torch.cat([torch.zeros((1, 2, 64, 64)).to(device), flow], dim=0)
        noise1 = (input[:, 0] - logits[:, 0]).unsqueeze(1)
        noise1 = spatial(noise1, flow)
        noise1 = torch.where(noise1==0, noise1 + input.median(), noise1)

        # calculate loss
        K1 = torch.cat([torch.flatten(Output[i, 0] + complement[i, 0]).unsqueeze(0) for i in range(input.shape[0])], dim = 0)
        nuc = nucloss(K1)
        MSElloss = MSE((Output[0]+complement[0]).repeat(input.shape[0], 1, 1, 1), Output+complement) 
        MSElloss2 = MSE((Output[0]).repeat(input.shape[0], 1, 1, 1), Output) 

        M3 = torch.cat([torch.flatten(noise1[i]).unsqueeze(0) for i in range(input.shape[0])], dim = 0)
        a3 = torch.ones(1, input.shape[0]).to(device)
        b3 = torch.full((1, len(logits[0][0])*len(logits[0][0])),fill_value=input.median()).to(device)
        loss5_3 =  MSE(torch.mm(a3, M3)/input.shape[0], b3)
        # l1 = l1loss(logits[:, 1])
        loss_grad = 2*smooth(flow, flow)
        l1_comp = l1loss(complement)
        regist = l1loss(flow)

        loss = 100*loss5_3   + 10* MSElloss2 + 0.001*nuc + 1e-4*l1_comp + 0.000001*regist + 0.1*loss_grad 

        # dice
        if test == 1:
            grandtruth_erased = input_img[1].to(device)
            grandtruth_erased = (grandtruth_erased - grandtruth_erased.min()) / (grandtruth_erased.max() - grandtruth_erased.min())
            warp = spatial(grandtruth_erased[0:grandtruth.shape[0]], flow)
            warp += complement
            dice = diceloss(grandtruth[0].repeat(warp.shape[0], 1, 1, 1), warp)
            IoU = iouloss(grandtruth[0].repeat(warp.shape[0], 1, 1, 1), warp)
            dice_sprase= 0
            IoU_sparse = 0

        else:
            grandtruth_erased = input_img[1].to(device)
            grandtruth_erased = (grandtruth_erased - grandtruth_erased.min()) / (grandtruth_erased.max() - grandtruth_erased.min())
            warp = spatial(grandtruth_erased[0:grandtruth.shape[0]], flow)
            warp += complement
            dice = diceloss(grandtruth[0].repeat(warp.shape[0], 1, 1, 1), warp)
            IoU = iouloss(grandtruth[0].repeat(warp.shape[0], 1, 1, 1), warp)
            dice_sprase= 0
            IoU_sparse = 0          

        warp_temp = warp.permute(0, 2, 3, 1).detach().cpu().numpy()*255
        warp_temp[warp_temp>0] = 255
        inp_temp = grandtruth_erased.permute(0, 2, 3, 1).detach().cpu().numpy()*255
        inp_temp[inp_temp>0] = 255
        for idx in range(8):
            cv2.imwrite(f'/home/asanomi/デスクトップ/cvpr_hikaku/ours_output/{i}_{idx}.png', warp_temp[idx])    
            cv2.imwrite(f'/home/asanomi/デスクトップ/cvpr_hikaku/ours_output/inp{i}_{idx}.png', inp_temp[idx])    
        L = logits[:, 0].unsqueeze(1)
        Output = (Output - Output.min()) / (Output.max() - Output.min())
        noise1 = (noise1 - noise1.min()) / (noise1.max() - noise1.min())
        
        L = L.permute(0, 2, 3, 1).detach().cpu().numpy()*255 
        noise1 = noise1.permute(0, 2, 3, 1).detach().cpu().numpy()*255 
        Output = Output.permute(0, 2, 3, 1).detach().cpu().numpy()*255 
        logits = logits.permute(0, 2, 3, 1).detach().cpu().numpy()*255    
        input = input.permute(0, 2, 3, 1).detach().cpu().numpy()*255
        Warp = warp.permute(0, 2, 3, 1).detach().cpu().numpy()*255
        Grandtruth = grandtruth.permute(0, 2, 3, 1).detach().cpu().numpy()*255
        complement = complement.permute(0, 2, 3, 1).detach().cpu().numpy()*255

        input_temp = np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3))
        noise_temp = np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3))
        # sparse_temp =  np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3))  
        output_temp =  np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3)) 
        complement_temp = np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3)) 
        warp_temp = np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3)) 
        grandtruth_temp = np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3))
        comp_temp = np.zeros((Output.shape[0], Output.shape[1], Output.shape[2], 3))

        for m in range(input.shape[0]):
            input_temp[m] = put_text(input[m], "input",  point = (0, 0))
            warp_temp[m] = put_text(Warp[m], "input τ",  point = (0, 0))
            noise_temp[m] = put_text(noise1[m], "noise",  point = (0, 0))
            # sparse_temp[m] = put_text(sparse[m], "sparse",  point = (0, 0))
            output_temp[m] = put_text(Output[m], " L τ",  point = (0, 0))
            complement_temp[m] = put_text(L[m], " L",  point = (0, 0))
            grandtruth_temp[m] = put_text(Grandtruth[m], " no_noise_input",  point = (0, 0))
            comp_temp[m] = put_text(complement[m], " hokan",  point = (0, 0))

        input = input_temp
        noise3_1 = noise_temp
        output3 = output_temp
        # sparse3 = sparse_temp
        complement3 = complement_temp
        warp = warp_temp
        grandtruth = grandtruth_temp
        complement = comp_temp

        image = get_img_table([input[0], grandtruth[0], warp[0], output3[0], complement3[0], noise3_1[0], complement[0]])[np.newaxis, :, :, :]
        for m in range(1, input.shape[0]):
            image = np.concatenate([image, get_img_table([input[m], grandtruth[m], warp[m], output3[m], complement3[m], noise3_1[m], complement[m]])[np.newaxis, :, :, :]], axis=0)

        for s in range(input.shape[0]):
            cv2.imwrite(dir + "cross{}/".format(crossid) + "{}_{}.png".format(i, s), image[s])
            
        k += 1
        loss_dice += float(dice)
        loss_IoU += float(IoU)
        loss_dice_sparse += float(dice_sprase)
        loss_IoU_sparse += float(IoU_sparse)
        loss_Total += float(loss)

        if test==0:
            if i%200 == 0:
                if i == 0:
                    print("val idx, loss : {},  {}".format(i, loss_Total))
                    print("val idx, Dice : {},  {}".format(i, loss_dice))
                    print("val idx, IoU : {},  {}".format(i, loss_IoU))
                else:
                    print("val idx, loss : {},  {}".format(i, loss_Total/(i+1)))
                    print("val idx, Dice : {},  {}".format(i, loss_dice/(i+1)))
                    print("val idx, IoU : {},  {}".format(i, loss_IoU/(i+1)))
        if test==1:
            if i%200 == 0:
                if i == 0:
                    print("val idx, Dice : {},  {}".format(i, loss_dice))
                    print("val idx, IoU : {},  {}".format(i, loss_IoU))
                else:
                    print("val idx, Dice : {},  {}".format(i, loss_dice/(i+1)))
                    print("val idx, IoU : {},  {}".format(i, loss_IoU/(i+1)))

        loss_Total += float(loss)
        time_all += end - start
        # if i == 200:
        #     break

    # calculate mean loss
    loss_dice /= (i + 1)
    loss_IoU /= (i + 1)
    loss_dice_sparse /= (i + 1)
    loss_IoU_sparse /= (i + 1) 
    print(time_all / (i+1))   

    print("dice:{}".format(loss_dice))
    print("IoU:{}".format(loss_IoU))
    print("loss:{}".format(loss_Total))

    return k, [loss_dice, loss_IoU, loss_dice_sparse, loss_IoU_sparse], loss_Total
