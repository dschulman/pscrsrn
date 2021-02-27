import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _seq_mask(x, N):
    mask = torch.arange(x.shape[2], device=N.device).unsqueeze(0) < N.unsqueeze(1)
    return x * mask.unsqueeze(1)

class Reduce(nn.Module):
    def __init__(self,
            n_hidden, kernel_size, stride,
            depth_variant, dropout):
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 3")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if stride < 2:
            raise ValueError("stride must be >= 2")
        if stride > kernel_size:
            raise ValueError("stride must be <= kernel_size")
        super().__init__()
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth_variant = depth_variant
        self.dropout = dropout
        self.conv1 = nn.Conv1d(
            in_channels = n_hidden + (1 if depth_variant else 0),
            out_channels = n_hidden * 2,
            kernel_size = kernel_size,
            stride = stride,
            padding = (kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(
            in_channels = n_hidden,
            out_channels = n_hidden,
            kernel_size = kernel_size,
            stride = 1,
            padding = (kernel_size - 1) // 2)
        self.act = nn.ReLU()

    def _reduce(self, h, N, depth):
        nh = self.n_hidden
        ks = self.kernel_size
        stride = self.stride
        if self.depth_variant:
            d = torch.full((h.shape[0], 1, h.shape[2]), math.log1p(depth), device=h.device)
            h = torch.cat([h, d], dim=1)
        h = _seq_mask(h, N)
        lr = self.conv1(h)
        N = torch.floor((N - 1) / stride).long() + 1
        l = lr[:,:nh]
        r = lr[:,nh:]
        r = self.act(r)
        r = _seq_mask(r, N)
        r = self.conv2(r)
        h = self.act(l + r)
        return h, N

    def forward(self, h, N):
        mask = None
        if self.training and self.dropout > 0.0:
            mask = torch.rand(h.shape[0], h.shape[1], 1, device=h.device) > self.dropout
            mask = mask / (1 - self.dropout)
        out = []
        depth = 0
        while h.shape[0] > 0:
            if mask is not None:
                h = mask * h
            h, N = self._reduce(h, N, depth)
            reduced = (N <= 1)
            out.append(h[reduced, :, 0])
            h = h[~reduced]
            N = N[~reduced]
            if mask is not None:
                mask = mask[~reduced]
            depth += 1
        return torch.cat(out, dim=0)
