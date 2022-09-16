import torch
import torch.nn as nn
from Model.model_parts import *
from torch.distributions.normal import Normal


class Non_Rigid_Alignment_Net(nn.Module):
    def __init__(self, in_channels, out_channels=2, num=16, bilinear=True):
        super().__init__()

        self.inc = DoubleConv(in_channels*2,  num)
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

    def forward(self, A):
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
        return Aτ, τ
