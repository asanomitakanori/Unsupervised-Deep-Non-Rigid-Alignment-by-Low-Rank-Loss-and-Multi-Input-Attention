
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class MultiInputAttetnion(nn.Module):
    """(Attention layer) use convolution instead of FC layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Wq = DoubleConv2(in_channels, out_channels)
        self.Wk = DoubleConv2(in_channels, out_channels)
        self.Wv = DoubleConv2(in_channels, out_channels)
        self.softmax = nn.Softmax(dim =1)

    def forward(self, input):
        query = self.Wq(input).view(input.shape[0], -1)
        key = self.Wk(input).view(input.shape[0], -1)
        value = self.Wv(input).view(input.shape[0], -1)
        attention_weight = torch.matmul(query, key.permute(1, 0)).view(input.shape[0], -1)
        attention_weight = self.softmax(attention_weight)
        weighted_value= torch.matmul(attention_weight, value).reshape(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        return weighted_value + input
