import torch
import torch.nn as nn
from Model.model_parts import *

class Noise_Decomposition_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num=16, bilinear=True):
        super().__init__()

        self.inc = DoubleConv(in_channels, num)
        self.down1 = Down(num, num*2)
        factor = 2 if bilinear else 1
        self.down2 = Down(num*2, num*4)
        self.down3 = Down( num*4,  num*8)
        self.down4 = Down(num*8,  num*16 // factor)
        self.up1 = Up(num*16,  num*8 // factor, bilinear)
        self.up2 = Up(num*8,  num*4 // factor, bilinear)
        self.up3 = Up(num*4,  num*2 // factor, bilinear)
        self.up4 = Up(num*2,  num, bilinear)
        self.out = OutConv(num, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        A = self.out(x)
        A = self.sigmoid(A)
        return A

