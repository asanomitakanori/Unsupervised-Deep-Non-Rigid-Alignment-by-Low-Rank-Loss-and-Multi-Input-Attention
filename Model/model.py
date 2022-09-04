""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from Model.model_parts import *
from Model.attention import MultiInputAttetnion
from Model.model_parts import *


class LowRankNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(LowRankNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        num = 16
        self.inc_1 = DoubleConv(n_channels, num)
        self.down1_1 = Down(num, num*2)
        factor = 2 if bilinear else 1
        self.down2_1 = Down(num*2, num*4)
        self.down3_1 = Down( num*4,  num*8)
        self.down4_1 = Down(num*8,  num*16 // factor)
        self.up1_1 = Up(num*16,  num*8 // factor, bilinear)
        self.up2_1 = Up(num*8,  num*4 // factor, bilinear)
        self.up3_1 = Up(num*4,  num*2 // factor, bilinear)
        self.up4_1 = Up(num*2,  num, bilinear)
        self.out_1 = OutConv(num, 1)

        self.inc = DoubleConv(n_channels*2,  num)
        self.down1 = Conv(num, num*2)
        self.down2 = Conv(num*2, num*2)
        self.down3 = Conv(num*2, num*2)
        self.down4 = Conv(num*2, num*2)
        self.up1 = Up(num*4, num*2)
        self.up2 = Up(num*4, num*2)
        self.up3 = Up(num*4, num)
        self.up4 = Up(num*2, 8)
        self.out = DoubleConv(8, 8)
        self.flow = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.activation = nn.LeakyReLU(0.2)
        self.transformer = SpatialTransformer((64, 64))
        self.sigmoid = nn.Sigmoid()

        self.inc_3 = DoubleConv(n_channels, num)
        self.attention = MultiInputAttetnion(num, num)
        self.down1_3 = Down(num, 2*num)
        self.attention1 = MultiInputAttetnion(2*num, 2*num)
        self.down2_3 = Down(2*num, 4*num)
        self.down3_3 = Down(4*num, 8*num)
        self.down4_3 = Down(8*num, 16*num // factor)
        self.up1_3 = Up(16*num,  8*num // factor, bilinear)
        self.up2_3 = Up(8*num,  4*num // factor, bilinear)
        self.up3_3 = Up(4*num , 2*num // factor, bilinear)
        self.up4_3 = Up(2*num, num, bilinear)
        self.out_3 = OutConv(num, 1)

    def forward(self, input):
        # Noise Decomposition
        x1s = self.inc_1(input)
        x2s = self.down1_1(x1s)
        x3s = self.down2_1(x2s)
        x4s = self.down3_1(x3s)
        x5s = self.down4_1(x4s)
        x = self.up1_1(x5s, x4s)
        x = self.up2_1(x, x3s)
        x = self.up3_1(x, x2s)
        x = self.up4_1(x, x1s)
        A = self.out_1(x)
        A = self.sigmoid(A)
        
        # Non-rigid Alignment Network
        target = A[0].unsqueeze(0).repeat(A.shape[0]-1, 1, 1, 1)
        source = A[1:A.shape[0], 0, :, :].unsqueeze(1)
        input2 = torch.cat([source, target], dim=1)
        x1 = self.inc(input2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.activation(x)
        x = self.up2(x, x3)
        x = self.activation(x)
        x = self.up3(x, x2)
        x = self.activation(x)
        x = self.up4(x, x1)
        x = self.activation(x)
        x = self.out(x)
        x = self.activation(x)
        τ = self.flow(x)
        Aτ = self.transformer(source, τ)
        Aτ = torch.cat([target[0].unsqueeze(0), Aτ])

        # Sparse Error Complement Network
        x1t = self.inc_3(Aτ)
        x2t = self.attention(x1t)
        x2t = self.down1_3(x2t)
        x2t = self.attention1(x2t)
        x3t = self.down2_3(x2t)
        x4t = self.down3_3(x3t)
        x5t = self.down4_3(x4t)
        x = self.up1_3(x5t, x4t)
        x = self.up2_3(x, x3t)
        x = self.up3_3(x, x2t)
        x = self.up4_3(x, x1t)
        S = self.out_3(x)
        S = self.sigmoid(S)
        return A, Aτ, τ, S
