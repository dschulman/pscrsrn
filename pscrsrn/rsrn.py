import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as tnur
from typing import List
from . import seq

def _depth_cat(h, depth: int):
    dshape = (h.shape[0], 1, h.shape[2])
    d = torch.full(dshape, math.log1p(depth), device=h.device)
    return torch.cat([h,d], dim=1)

class Block(nn.Module):
    def __init__(self, seq_conv_cls,
            channels, kernel_size, stride, pad_delta,
            leak=0.0, layer_norm=True, depth_variant=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_delta = pad_delta
        self.leak = leak
        self.layer_norm = layer_norm
        self.depth_variant = depth_variant
        self.conv1 = seq_conv_cls(
            in_channels = channels + (1 if depth_variant else 0),
            out_channels = channels * 2,
            kernel_size = kernel_size,
            stride = stride,
            pad_delta = pad_delta)
        self.norm1 = seq.LayerNorm() if layer_norm else None
        self.conv2 = seq_conv_cls(
            in_channels = channels + (1 if depth_variant else 0),
            out_channels = channels,
            kernel_size = kernel_size,
            stride = 1,
            pad_delta = 1)
        self.norm2 = seq.LayerNorm() if layer_norm else None
        self.act = nn.LeakyReLU(leak) if leak > 0 else nn.ReLU()

    def forward(self, h, N, depth: int):
        c = self.channels
        if self.depth_variant:
            h = _depth_cat(h, depth)
        lr, N = self.conv1(h, N)
        l = lr[:,:c]
        r = lr[:,c:]
        if self.norm1 is not None:
            r = self.norm1(r, N)
        r = self.act(r)
        if self.depth_variant:
            r = _depth_cat(r, depth)
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

class Reduce(nn.Module):
    def __init__(self, 
            hidden, kernel_size, stride, layers, depth_variant, 
            stride_on, leak, dropout, layer_norm):
        super().__init__()
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.layers = layers
        self.depth_variant = depth_variant
        self.stride_on = stride_on
        self.leak = leak
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.blocks = nn.ModuleList([
            Block(
                seq_conv_cls = seq.Conv, 
                channels = hidden, 
                kernel_size = kernel_size, 
                stride = stride if _stride(l, layers, stride_on) else 1,
                pad_delta = 1,
                leak = leak,
                depth_variant = depth_variant,
                layer_norm = layer_norm)
            for l in range(layers)])

    def forward(self, h, N):
        if self.training and self.dropout > 0.0:
            mask_shape = (h.shape[0], h.shape[1], 1)
            mask = torch.rand(mask_shape, device=h.device) > self.dropout
            mask = mask / (1 - self.dropout)
        else:
            mask = None
        out = []
        depth = 0
        while h.shape[0] > 0:
            if mask is not None:
                h = mask * h
            for block in self.blocks:
                h, N = block(h, N, depth)
            reduced = (N <= 1)
            out.append(h[reduced, :, 0])
            h = h[~reduced]
            N = N[~reduced]
            if mask is not None:
                mask = mask[~reduced]
            depth += 1
        return torch.cat(out, dim=0)

class Expand(nn.Module):
    def __init__(self, 
            hidden, kernel_size, stride, layers, depth_variant, 
            stride_on, leak, dropout, layer_norm):
        super().__init__()
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.layers = layers
        self.depth_variant = depth_variant
        self.stride_on = stride_on
        self.leak = leak
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.blocks = nn.ModuleList([
            Block(
                seq_conv_cls = seq.ConvTranspose, 
                channels = hidden, 
                kernel_size = kernel_size, 
                stride = stride if _stride(l, layers, stride_on) else 1,
                pad_delta = 3, 
                leak = leak,
                depth_variant = depth_variant,
                layer_norm = layer_norm)
            for l in range(layers)])

    def forward(self, h, N):
        mask = None
        if self.training and self.dropout > 0.0:
            mask_shape = (h.shape[0], h.shape[1], 1)
            mask = torch.rand(mask_shape, device=h.device) > self.dropout
            mask = mask / (1 - self.dropout)
        N_orig = N
        h = h.unsqueeze(-1)
        M = torch.ones_like(N)
        out_h = []
        out_M = []
        depth = 0
        while h.shape[0] > 0:
            if mask is not None:
                h = mask * h
            for block in self.blocks:
                h, M = block(h, M, depth)
            expanded = (M >= N)
            out_h.extend(torch.unbind(h[expanded]))
            out_M.append(M[expanded])
            h = h[~expanded]
            M = M[~expanded]
            N = N[~expanded]
            if mask is not None:
                mask = mask[~expanded]
            depth += 1
        M = torch.cat(out_M)
        offset = (M - N_orig) // 2
        hs = [hi.T[oi:(oi+Ni)] for hi, oi, Ni in zip(out_h, offset, N_orig)]
        return tnur.pad_sequence(hs, batch_first=True).transpose(1,2)

class Classify(nn.Module):
    def __init__(self,
            features, classes,
            inproj_size=7, inproj_stride=4, inproj_norm=True,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            stride_on='first',
            outproj_size=64,
            dropout=0.2, leak=0.0, layer_norm=True):
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
        self.reduce = Reduce(
            hidden = hidden,
            kernel_size = kernel_size,
            stride = stride,
            layers = layers,
            depth_variant = depth_variant,
            stride_on = stride_on,
            dropout = dropout,
            leak = leak,
            layer_norm = layer_norm)
        self.outproj_lin1 = nn.Linear(
            in_features = hidden,
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
        h = self.reduce(h, N)
        h = self.outproj_lin1(h)
        h = self.outproj_act(h)
        z = self.outproj_lin2(h)
        return z[seq.invert_permutation(sorted_indices)]

class Generate(nn.Module):
    def __init__(self,
            latent, features,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            stride_on='all',
            outproj_size=5,
            dropout=0.0, leak=0.1, layer_norm=True):
        super().__init__()
        self.inproj = nn.Linear(
            in_features = latent,
            out_features = hidden)
        self.expand = Expand(
            hidden = hidden, 
            kernel_size = kernel_size, 
            stride = stride, 
            layers = layers, 
            depth_variant = depth_variant, 
            stride_on = stride_on, 
            leak = leak, 
            dropout = dropout, 
            layer_norm = layer_norm)
        self.outproj = seq.Conv(
            in_channels = hidden,
            out_channels = features,
            kernel_size = outproj_size,
            stride = 1,
            pad_delta = 1)

    def forward(self, h, N):
        N, sorted_indices = torch.sort(N)
        h = h[sorted_indices]
        h = self.inproj(h)
        h = self.expand(h, N)
        y, _ = self.outproj(h, N)
        y = y * seq.mask(y, N)
        return y[seq.invert_permutation(sorted_indices)]
