""" Full assembly of the parts to form the complete network """

import torch.nn as nn
from Model.noise_decompositon_Net import Noise_Decomposition_Net
from Model.non_rigid_alignment_Net import Non_Rigid_Alignment_Net
from Model.sparse_complement_Net import Sparse_Complement_Net


class LowRankNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_ch=16, bilinear=True):
        super(LowRankNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.Noise_Decomposition_Net = Noise_Decomposition_Net(n_channels, n_classes, num_ch)
        self.Non_Rigid_Alignment_Net = Non_Rigid_Alignment_Net(n_channels, 2, num_ch)
        self.Sparse_Complement_Net = Sparse_Complement_Net(n_channels, n_classes, num_ch)

    def forward(self, input):
        '''
        Args:
            input (Batch, Channel, Height, Width)
        Return:
            A (torch.): Denoised input imgs, 
            Aτ: Aligned A,
            τ: Flow 
            S: Sparse complement of Aτ
        '''

        A     = self.Noise_Decomposition_Net(input)
        Aτ, τ = self.Non_Rigid_Alignment_Net(A)
        S     = self.Sparse_Complement_Net(Aτ)
        return A, Aτ, τ, S
