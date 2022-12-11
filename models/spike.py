import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.backbones import FCN, DeepConvLSTM


class AvgMeter:

    def __init__(self):
        self.value = 0
        self.number = 0

    def add(self, v, n):
        self.value += v
        self.number += n

    def avg(self):
        return self.value / self.number


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        # tmp = torch.ones_like(input)
        # tmp = torch.where(input.abs() < 0.5, 1., 0.)
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bp = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bp)) / \
                 (2 * np.tanh(self.region * temp)) + 0.5
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.5, gamma=1.0, dspike=False, soft_reset=True):
        """
        Implementing the LIF neurons.
        @param thresh: firing threshold;
        @param tau: membrane potential decay factor;
        @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
        @param dspike: whether using rectangular gradient of dspike gradient;
        @param soft_reset: whether using soft-reset or hard-reset.
        """
        super(LIFSpike, self).__init__()
        if not dspike:
            self.act = ZIF.apply
        else:
            # using the surrogate gradient function from Dspike: 
            # https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf
            self.act = DSPIKE(region=1.0)
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.soft_reset = soft_reset

    def forward(self, x):
        mem = 0
        spike_out = []
        T = x.shape[2]
        for t in range(T):
            mem = mem * self.tau + x[:, :, t]
            spike = self.act(mem - self.thresh, self.gamma)
            mem = mem - spike * self.thresh if self.soft_reset else (1 - spike) * mem
            spike_out.append(spike)

        return torch.stack(spike_out, dim=2)


class SFCN(FCN):

    def __init__(self, n_channels, n_classes, out_channels=128, backbone=True, **kwargs):
        super(SFCN, self).__init__(n_channels, n_classes, out_channels, backbone)
        self.conv_block1 = nn.Sequential(nn.Conv1d(n_channels, 32, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(32),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1),
                                         nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(nn.Conv1d(64, out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                                         nn.BatchNorm1d(out_channels),
                                         LIFSpike(**kwargs),
                                         nn.AvgPool1d(kernel_size=2, stride=2, padding=1))


class SDCL(DeepConvLSTM):

    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=True, **snn_p):
        super(SDCL, self).__init__(n_channels, n_classes, conv_kernels, kernel_size, LSTM_units, backbone)
        self.act1 = LIFSpike(**snn_p)
        self.act2 = LIFSpike(**snn_p)
        self.act3 = LIFSpike(**snn_p)
        self.act4 = LIFSpike(**snn_p)

        self.bn1 = nn.BatchNorm2d(conv_kernels)
        self.bn2 = nn.BatchNorm2d(conv_kernels)
        self.bn3 = nn.BatchNorm2d(conv_kernels)
        self.bn4 = nn.BatchNorm2d(conv_kernels)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[-1, :, :]

        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x



