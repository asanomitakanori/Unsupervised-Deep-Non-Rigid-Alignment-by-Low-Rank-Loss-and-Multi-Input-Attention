""" Full assembly of the parts to form the complete network """

from attention import Self_Attention_layer
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
from unet_parts import *
from attention import Self_Attention_layer
import  os

class DoubleConv(nn.Module):
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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class DoubleConv(nn.Module):
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

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.convolution = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        )
    def forward(self, x):
        return self.convolution(x)

class Flow(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.convolution = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    def forward(self, x):
        return self.convolution(x)


class Vox2d(nn.Module):
    def __init__(self, n_channels, n_classes, batch_size, bilinear=True):
        super(Vox2d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batch_size = batch_size
        
        num = 16
        self.inc_1 = DoubleConv(1, num)
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

        num_ch = 16
        self.inc = DoubleConv(n_channels,  num_ch)
        self.down1 = Conv(num_ch, num_ch*2)
        self.down2 = Conv(num_ch*2, num_ch*2)
        self.down3 = Conv(num_ch*2, num_ch*2)
        self.down4 = Conv(num_ch*2, num_ch*2)
        self.up1 = Up(num_ch*4, num_ch*2)
        self.up2 = Up(num_ch*4, num_ch*2)
        self.up3 = Up(num_ch*4, num_ch)
        self.up4 = Up(num_ch*2, 8)
        self.out = DoubleConv(8, 8)

        self.flow = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.activation = nn.LeakyReLU(0.2)
        self.transformer = SpatialTransformer((64, 64))
        self.sigmoid = nn.Sigmoid()

        num2 = 16
        self.inc_3 = DoubleConv(1, num2)
        self.attention = Self_Attention_layer(num2, num2)
        self.down1_3 = Down(num2, 2*num2)
        self.attention1 = Self_Attention_layer(2*num2, 2*num2)
        self.down2_3 = Down(2*num2, 4*num2)
        self.attention2 = Self_Attention_layer(4*num2, 4*num2)
        self.down3_3 = Down(4*num2, 8*num2)
        self.attention3 = Self_Attention_layer(8*num2, 8*num2)
        self.down4_3 = Down(8*num2, 16*num2 // factor)
        self.attention4 = Self_Attention_layer(16*num2 // factor, 16*num2 // factor)
        self.up1_3 = Up(16*num2,  8*num2 // factor, bilinear)
        self.attention5 = Self_Attention_layer(8*num2 // factor, 8*num2 // factor)
        self.up2_3 = Up(8*num2,  4*num2 // factor, bilinear)
        self.attention6 = Self_Attention_layer(4*num2 // factor, 4*num2 // factor)
        self.up3_3 = Up(4*num2 , 2*num2 // factor, bilinear)
        self.attention7 = Self_Attention_layer(2*num2 // factor, 2*num2 // factor)
        self.up4_3 = Up(2*num2, num2, bilinear)
        self.attention8 = Self_Attention_layer(num2, num2)
        self.out_3 = OutConv(num2, 1)

    def forward(self, input):
        x1s = self.inc_1(input)
        x2s = self.down1_1(x1s)
        x3s = self.down2_1(x2s)
        x4s = self.down3_1(x3s)
        x5s = self.down4_1(x4s)
        x = self.up1_1(x5s, x4s)
        x = self.up2_1(x, x3s)
        x = self.up3_1(x, x2s)
        x = self.up4_1(x, x1s)
        logits = self.out_1(x)
        logits = self.sigmoid(logits)
        
        target = logits[0][0].unsqueeze(0).unsqueeze(0).repeat(logits.shape[0]-1, 1, 1, 1)
        source = logits[1:input.shape[0], 0, :, :].unsqueeze(1)
        temp = torch.cat([source, target], dim=1)

        x1 = self.inc(temp)
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
        flow_field = self.flow(x)
        y_source = self.transformer(source, flow_field)
        y_source = torch.cat([target[0].unsqueeze(0), y_source])

        x1t = self.inc_3(y_source)
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
        complement = self.out_3(x)
        complement = self.sigmoid(complement)
        return logits, y_source, flow_field, complement

if __name__=="__main__":
    net = Vox2d(2, 3, 5)
    x = torch.randn((5, 1, 64, 64))  # (B, ch, DEPTH, HEIGHT, WIDTH)
    with torch.no_grad():
        y = net(x)
    print('finished')
    # path = os.getcwd()
    # print(path)