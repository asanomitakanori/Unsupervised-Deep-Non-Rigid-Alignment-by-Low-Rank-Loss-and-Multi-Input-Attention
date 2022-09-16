import torch
import torch.nn as nn
from Model.model_parts import *
from Model.attention import MultiInputAttetnion


class Sparse_Complement_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num=16, bilinear=True):
        super().__init__()

        factor = 2 if bilinear else 1
        self.inc_3 = DoubleConv(in_channels, num)
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
        self.out_3 = OutConv(num, out_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, Aτ):
        x1 = self.inc_3(Aτ)
        x1 = self.attention(x1)
        x2 = self.down1_3(x1)
        x2 = self.attention1(x2)
        x3 = self.down2_3(x2)
        x4 = self.down3_3(x3)
        x5 = self.down4_3(x4)
        x = self.up1_3(x5, x4)
        x = self.up2_3(x, x3)
        x = self.up3_3(x, x2)
        x = self.up4_3(x, x1)
        S = self.out_3(x)
        S = self.sigmoid(S)
        return S
