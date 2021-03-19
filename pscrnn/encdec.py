import torch
import torch.nn as nn
from . import pst

class Encode(nn.Module):
    def __init__(self,
            features, latent,
            inproj_size=8, inproj_stride=4,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            dropout=0.0, leak=0.0, weight_norm=True):
        super().__init__()
        self.inproj_size = inproj_size
        self.inproj_stride = inproj_stride
        self.inproj = nn.Conv1d(
            in_channels = features,
            out_channels = hidden,
            kernel_size = inproj_size,
            stride = inproj_stride)
        self.reduce = pst.Reduce(
            hidden = hidden,
            kernel_size = kernel_size,
            stride = stride,
            layers = layers,
            depth_variant = depth_variant,
            dropout = dropout,
            leak = leak,
            weight_norm = weight_norm)
        self.outproj = nn.Linear(
            in_features = hidden,
            out_features = latent)

    def forward(self, x, N):
        h = self.inproj(x)
        N = torch.floor(((N - self.inproj_size) / self.inproj_stride) + 1).long()
        h = self.reduce(h, N)
        return self.outproj(h)

class Decode(nn.Module):
    def __init__(self,
            features, latent,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            dropout=0.0, leak=0.2, weight_norm=True,
            outproj_size=5):
        super().__init__()
        self.inproj = nn.Linear(
            in_features = latent,
            out_features = hidden)
        self.expand = pst.Expand(
            hidden = hidden,
            kernel_size = kernel_size,
            stride = stride,
            layers = layers,
            depth_variant = depth_variant,
            dropout = dropout,
            leak = leak,
            weight_norm = weight_norm)
        self.outproj = pst.SeqConv(
            in_channels = hidden,
            out_channels = features,
            kernel_size = outproj_size,
            stride = 1,
            pad_delta = 1)

    def forward(self, x, N):
        h = self.inproj(x)
        h = self.expand(h, N)
        y, _ = self.outproj(h, N)
        return y
