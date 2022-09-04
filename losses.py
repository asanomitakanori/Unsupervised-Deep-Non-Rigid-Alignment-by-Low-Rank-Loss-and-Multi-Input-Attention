from numpy.core.numeric import NaN
import torch
import torch.utils.data
import numpy as np 

soft = lambda z, th: z.sign() * (z.abs() - th).max(torch.tensor(0., device="cuda"))


class L1LossFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lw):
        ctx.save_for_backward(input, lw)
        return torch.sum(torch.abs(input)*lw)

    @staticmethod
    def backward(ctx, grad_output):
        input, lw = ctx.saved_tensors
        grad_input = grad_output.clone()
        return (input - soft(input, lw)) * grad_input, torch.abs(input) * grad_input


class L1Loss(torch.nn.Module):
    def __init__(self, lw=torch.tensor(1.0)):
        super(L1Loss, self).__init__()
        self.fn = L1LossFunc.apply
        self.lw = torch.nn.Parameter(lw, requires_grad=lw.requires_grad)

    def forward(self, input):
        return self.fn(input, self.lw)
        
class NuclearLossFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lw):
        u, s, v = torch.svd(input)
        ctx.save_for_backward(input, lw, u, s, v)
        return torch.sum(s*lw)

    @staticmethod
    def backward(ctx, grad_output):
        input, lw, u, s, v = ctx.saved_tensors
        grad_input = grad_output.clone()
        svt_input = torch.matmul(torch.matmul(u, torch.diag_embed(soft(s,lw))), torch.transpose(v, -2, -1))
        return (input - svt_input) * grad_input, s * grad_input


class NuclearLoss(torch.nn.Module):
    def __init__(self, lw=torch.tensor(1.0)):
        super(NuclearLoss, self).__init__()
        self.fn = NuclearLossFunc.apply
        self.lw = torch.nn.Parameter(lw, requires_grad=lw.requires_grad)

    def forward(self, input):
        return self.fn(input, self.lw)


class NoiseLoss:

    def loss(self, Nτ, imgs_median, MSELoss, device):
        M3 = torch.cat([torch.flatten(Nτ[i]).unsqueeze(0) for i in range(Nτ.shape[0])], dim = 0)
        a3 = torch.ones(1, Nτ.shape[0]).to(device=device)
        b3 = torch.full((1, Nτ.shape[2]*Nτ.shape[3]),fill_value=imgs_median).to(device=device)
        loss_noise =  MSELoss(torch.mm(a3, M3)/Nτ.shape[0], b3)
        return loss_noise


class Grad:
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l2', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:,  :] - y_pred[:, :, :-1,  :]) 
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) 

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class Dice:
    """
    N-D dice for mnist
    """
    def loss(self, y_true, y_pred):
        temp = y_true.clone()
        temp2 = y_pred.clone()

        temp[temp>0] = 1
        temp2[temp2>0] = 1
        return  2*torch.sum(temp[temp2==1]==1) / (torch.sum(temp) + torch.sum(temp2))

