import torch
import torch.nn as nn
from . import pst

class Generate(nn.Module):
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
