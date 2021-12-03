import torch
import numpy as np
import torch.utils.data
from model_self_attention import SpatialTransformer
from image import *
import kornia
import losses
import random

def train(train_loader, model, nucloss, l1loss, optimizer,  k, MSE, std, train_writer=None):
    model.train()
    loss = 0
    total_loss = 0
    up1 = torch.nn.Upsample(size=((64, 64)), scale_factor=None, mode='nearest', align_corners=None,)    
    spatial = SpatialTransformer((64, 64), mode="nearest").cuda()
    smooth = losses.Grad().loss
    device = torch.device('cuda:0')

    for i, input_img in enumerate(train_loader):
        optimizer.zero_grad()
        input = input_img[0].to(device)
        rand = input.shape[0] - 1
        random_int = random.randint(0, rand)
        inp = input[random_int].unsqueeze(0)
        inp = torch.cat([inp, input[0:random_int]])
        input = torch.cat([inp, input[random_int+1::]])

        # use CNN 
        logits, Output, flow, complement = model(input)

        # warp noise 
        flow = torch.cat([torch.zeros((1, 2, 64, 64)).cuda(), flow], dim=0)
        temp = (input[:, 0] - logits[:, 0]).unsqueeze(1)
        noise_warp = temp
        noise_warp = spatial(noise_warp, flow)
        noise_warp = torch.where(noise_warp==0, noise_warp + input.median(), noise_warp)
        # loss_noise_temp = MSE(noise_warp[0, Output[0]>0.1].repeat(input.shape[0], 1), noise_warp[:, Output[0]>0.1])

        # calculate loss
        K1 = torch.cat([torch.flatten(Output[i, 0] + complement[i, 0]).unsqueeze(0) for i in range(input.shape[0])], dim = 0)
        nuc = nucloss(K1)
        MSElloss = MSE((Output[0]+complement[0]).repeat(input.shape[0], 1, 1, 1), Output+complement) 
        MSElloss2 = MSE((Output[0]).repeat(input.shape[0], 1, 1, 1), Output) 

        M3 = torch.cat([torch.flatten(noise_warp[i]).unsqueeze(0) for i in range(input.shape[0])], dim = 0)
        a3 = torch.ones(1, input.shape[0]).cuda()
        b3 = torch.full((1, len(logits[0][0])*len(logits[0][0])),fill_value=input.median()).cuda()
        loss5_3 =  MSE(torch.mm(a3, M3)/input.shape[0], b3)
        loss_grad = 2*smooth(flow, flow)
        l1_comp = l1loss(complement)
        regist = l1loss(flow)

        # loss = 100*loss5_3 + MSElloss + 10* MSElloss2 + 0.001*nuc + 1e-4*l1_comp + 0.000001*regist + 0.1*loss_grad + 0.5*loss_noise_temp
        loss = 100*loss5_3 + 10* MSElloss2 + 0.001*nuc + 1e-4*l1_comp + 0.000001*regist + 0.1*loss_grad

        loss.backward()
        optimizer.step()

        temp = complement.clone()
        temp[temp<0.1] = 0.8

        # tensorboard
        if k%100 == 0:
            for m in range(input.shape[0]):
                train_writer.add_image("input/input{}".format(m), input[m], k)  
                train_writer.add_image("N/noise{}".format(m), noise_warp[m], k)     
                train_writer.add_image("Lt/Lt{}".format(m), Output[m], k)     
                train_writer.add_image("L/L{}".format(m), logits[m][0].unsqueeze(0), k)    
                # train_writer.add_image("S/S{}".format(m), logits[m][1].unsqueeze(0), k)  
                train_writer.add_image("C/C{}".format(m), complement[m], k)  
                train_writer.add_image("C/temp{}".format(m), temp[m], k)   
                train_writer.add_image("C+L/C+L{}".format(m), complement[m] + Output[m], k)  

        train_writer.add_scalar("loss/MSE", MSElloss, k)
        train_writer.add_scalar("loss/nuclear", nuc, k)
        train_writer.add_scalar("loss/noise", loss5_3, k)
        train_writer.add_scalar("loss/smooth", loss_grad, k)
        train_writer.add_scalar("loss/l1_comp", l1_comp, k)
        train_writer.add_scalar("loss/MSE2", MSElloss2, k)
        train_writer.add_scalar("loss/l1_flow", regist, k)

        k += 1
        total_loss += loss
        loss =0 
        if i%200 == 0:
            print("idx, loss : {},  {}".format(i, total_loss))
            total_loss = 0
    return k

