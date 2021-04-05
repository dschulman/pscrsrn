import torch
import torch.nn as nn
import torch.nn.utils.rnn as tnur
from typing import List
from . import seq

class Block(nn.Module):
    def __init__(self,
            channels, kernel_size, stride,
            leak=0.0, batch_norm=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.leak = leak
        self.batch_norm = batch_norm
        self.conv1 = seq.Conv(
            in_channels = channels,
            out_channels = channels * 2,
            kernel_size = kernel_size,
            stride = stride,
            pad_delta = 1,
            bias = not batch_norm)
        self.norm1 = seq.BatchNorm(channels * 2) if batch_norm else None
        self.conv2 = seq.Conv(
            in_channels = channels,
            out_channels = channels,
            kernel_size = kernel_size,
            stride = 1,
            pad_delta = 1,
            bias = not batch_norm)
        self.norm2 = seq.BatchNorm(channels) if batch_norm else None
        self.act = nn.LeakyReLU(leak) if leak > 0 else nn.ReLU()

    def forward(self, x, N):
        c = self.channels
        lr, N = self.conv1(x, N)
        if self.norm1 is not None:
            lr = self.norm1(lr, N)
        l = lr[:,:c]
        r = lr[:,c:]
        r = self.act(r)
        r, _ = self.conv2(r, N)
        if self.norm2 is not None:
            r = self.norm2(r, N)
        return l + self.act(r), N

def _stride(layer, n_layers, stride_on):
    if stride_on == 'all':
        return True
    elif stride_on == 'first':
        return layer == 0
    elif stride_on == 'last':
        return layer == (n_layers - 1)
    else:
        raise ValueError(f'bad stride_on: {stride_on}')

class Classify(nn.Module):
    def __init__(self,
            features, classes,
            inproj_size=7, inproj_stride=4, inproj_norm=True,
            hidden=64, kernel_size=5, stride=2, layers=2,
            outproj_size=64,
            stride_on='all', dropout=0.2, leak=0.0, batch_norm=True):
        super().__init__()
        self.inproj_conv = seq.Conv(
            in_channels = features,
            out_channels = hidden,
            kernel_size = inproj_size,
            stride = inproj_stride,
            pad_delta = 1,
            bias = not inproj_norm)
        self.inproj_norm = seq.BatchNorm(hidden) if inproj_norm else None
        self.inproj_act = nn.LeakyReLU(leak) if leak>0.0 else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(
                channels=hidden,
                kernel_size=kernel_size,
                stride=stride if _stride(l, layers, stride_on) else 1,
                leak=leak,
                batch_norm=batch_norm)
            for l in range(layers)])
        self.outproj_lin1 = nn.Linear(
            in_features = hidden * 2,
            out_features = outproj_size)
        self.outproj_act = nn.LeakyReLU(leak) if leak>0.0 else nn.ReLU()
        self.outproj_lin2 = nn.Linear(
            in_features = outproj_size,
            out_features = classes)

    def forward(self, xs: List[torch.Tensor]):
        N = torch.tensor([x.shape[0] for x in xs], device=xs[0].device, dtype=torch.int)
        x = tnur.pad_sequence(xs, batch_first=True).transpose(1,2)
        N, sorted_indices = torch.sort(N)
        x = x[sorted_indices]
        h, N = self.inproj_conv(x, N)
        if self.inproj_norm is not None:
            h = self.inproj_norm(h, N)
        h = self.inproj_act(h)
        for block in self.blocks:
            h = self.drop(h)
            h, N = block(h, N)
        hmean = torch.sum(h, dim=2) / N.unsqueeze(1)
        hmax = torch.max(h, dim=2)[0]  ## TODO handle masking (for leaky ReLU)
        h = torch.cat((hmean, hmax), dim=1)
        h = self.outproj_lin1(h)
        h = self.outproj_act(h)
        z = self.outproj_lin2(h)
        return z[seq.invert_permutation(sorted_indices)]
