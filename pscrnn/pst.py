import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as tnur

def seq_mask(x, N):
    mask = torch.arange(x.shape[2], device=N.device) < N.unsqueeze(1)
    return mask.unsqueeze(1)

class _SeqConvBase(nn.Module):
    def __init__(self, conv_cls, 
            in_channels, out_channels, kernel_size, stride, pad_delta, weight_norm):
        if (kernel_size % 2) == 0:
            raise ValueError('kernel_size should be odd')
        if stride > kernel_size:
            raise ValueError('stride should be <= kernel_size')
        if pad_delta < 1:
            raise ValueError('pad_delta should be >= 1')
        if (pad_delta % 2) == 0:
            raise ValueError('pad_delta should be odd')
        if pad_delta > kernel_size:
            raise ValueError('pad_delta should be <= kernel_size')
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_delta = pad_delta
        self.weight_norm = weight_norm
        self.conv = conv_cls(
            in_channels, out_channels, kernel_size,
            stride = stride,
            padding = (kernel_size - pad_delta) // 2)
        if weight_norm:
            nn.utils.weight_norm(self.conv)
    
class SeqConv(_SeqConvBase):
    def __init__(self,
            in_channels, out_channels, kernel_size, stride, pad_delta,
            weight_norm = False):
        super().__init__(nn.Conv1d, in_channels, out_channels, kernel_size, stride, pad_delta, weight_norm)

    def forward(self, x, N):
        x = self.conv(x * seq_mask(x, N))
        N = torch.floor((N - self.pad_delta) / self.stride).long() + 1
        return x, N

class SeqConvTranspose(_SeqConvBase):
    def __init__(self, 
            in_channels, out_channels, kernel_size, stride, pad_delta,
            weight_norm = False):
        super().__init__(nn.ConvTranspose1d, in_channels, out_channels, kernel_size, stride, pad_delta, weight_norm)

    def forward(self, x, N):
        x = self.conv(x * seq_mask(x, N))
        N = ((N-1) * self.stride) + self.pad_delta
        return x, N

class SeqLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.log_weight = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, N):
        x = x * seq_mask(x, N)
        N = N.unsqueeze(-1).unsqueeze(-1)
        mu = torch.sum(x, dim=[1,2], keepdim=True) / x.shape[1] / N
        z = x - mu
        var = torch.sum(z*z, dim=[1,2], keepdim=True) / x.shape[1] / N
        w = torch.exp(self.log_weight)
        b = self.bias
        return w * z * torch.rsqrt(var + self.eps) + b 

class Block(nn.Module):
    def __init__(self, seq_conv_cls,
            channels, kernel_size, stride, pad_delta,
            leak=0.0, weight_norm=True, layer_norm=True, depth_variant=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_delta = pad_delta
        self.leak = leak
        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        self.depth_variant = depth_variant
        self.conv1 = seq_conv_cls(
            in_channels = channels + (1 if depth_variant else 0),
            out_channels = channels * 2,
            kernel_size = kernel_size,
            stride = stride,
            pad_delta = pad_delta,
            weight_norm = weight_norm)
        self.norm1 = SeqLayerNorm() if layer_norm else None
        self.conv2 = seq_conv_cls(
            in_channels = channels + (1 if depth_variant else 0),
            out_channels = channels,
            kernel_size = kernel_size,
            stride = 1,
            pad_delta = 1,
            weight_norm = weight_norm)
        self.norm2 = SeqLayerNorm() if layer_norm else None
        self.act = nn.LeakyReLU(leak) if leak > 0 else nn.ReLU()

    def _depth_cat(self, h, depth):
        if self.depth_variant:
            dshape = (h.shape[0], 1, h.shape[2])
            d = torch.full(dshape, math.log1p(depth), device=h.device)
            h = torch.cat([h,d], dim=1)
        return h

    def forward(self, h, N, depth):
        c = self.channels
        h = self._depth_cat(h, depth)
        lr, N = self.conv1(h, N)
        l = lr[:,:c]
        r = lr[:,c:]
        if self.norm1 is not None:
            r = self.norm1(r, N)
        r = self.act(r)
        r = self._depth_cat(r, depth)
        r, _ = self.conv2(r, N)
        if self.norm2 is not None:
            r = self.norm2(r, N)
        return self.act(l + r), N

class Reduce(nn.Module):
    def __init__(self, 
            hidden, kernel_size, stride, layers, depth_variant, 
            leak, dropout, weight_norm, layer_norm):
        super().__init__()
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.layers = layers
        self.depth_variant = depth_variant
        self.leak = leak
        self.dropout = dropout
        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        self.blocks = nn.ModuleList([
            Block(
                seq_conv_cls = SeqConv, 
                channels = hidden, 
                kernel_size = kernel_size, 
                stride = stride, 
                pad_delta = 1,
                leak = leak,
                depth_variant = depth_variant, 
                weight_norm = weight_norm,
                layer_norm = layer_norm)
            for _ in range(layers)])

    def forward(self, h, N):
        mask = None
        if self.training and self.dropout > 0.0:
            mask_shape = (h.shape[0], h.shape[1], 1)
            mask = torch.rand(mask_shape, device=h.device) > self.dropout
            mask = mask / (1 - self.dropout)
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
            leak, dropout, weight_norm, layer_norm):
        super().__init__()
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.layers = layers
        self.depth_variant = depth_variant
        self.leak = leak
        self.dropout = dropout
        self.weight_norm = weight_norm
        self.layer_norm = layer_norm
        self.blocks = nn.ModuleList([
            Block(
                seq_conv_cls = SeqConvTranspose, 
                channels = hidden, 
                kernel_size = kernel_size, 
                stride = stride,
                pad_delta = 3, 
                leak = leak,
                depth_variant = depth_variant, 
                weight_norm = weight_norm,
                layer_norm = layer_norm)
            for _ in range(layers)])

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

def invert_permutation(permutation):
    output = torch.empty_like(permutation)
    output.scatter_(
        0, permutation,
        torch.arange(0, permutation.numel(), device=permutation.device))
    return output

class Classify(nn.Module):
    def __init__(self,
            features, classes,
            inproj_size=7, inproj_stride=4,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            outproj_size=64,
            dropout=0.2, leak=0.0, weight_norm=True, layer_norm=True):
        super().__init__()
        self.inproj_conv = SeqConv(
            in_channels = features,
            out_channels = hidden,
            kernel_size = inproj_size,
            stride = inproj_stride,
            pad_delta = 1,
            weight_norm = weight_norm)
        self.inproj_norm = SeqLayerNorm() if layer_norm else None
        self.inproj_act = nn.LeakyReLU(leak) if leak>0.0 else nn.ReLU()
        self.reduce = Reduce(
            hidden = hidden,
            kernel_size = kernel_size,
            stride = stride,
            layers = layers,
            depth_variant = depth_variant,
            dropout = dropout,
            leak = leak,
            weight_norm = weight_norm,
            layer_norm = layer_norm)
        self.outproj_lin1 = nn.Linear(
            in_features = hidden,
            out_features = outproj_size)
        self.outproj_act = nn.LeakyReLU(leak) if leak>0.0 else nn.ReLU()
        self.outproj_lin2 = nn.Linear(
            in_features = outproj_size,
            out_features = classes)

    def forward(self, x, N):
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
        return z[invert_permutation(sorted_indices)]
