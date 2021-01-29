import torch
import torch.nn as nn
import torch.nn.functional as F

def _prefix(h, M):
    return h[...,:M.max()]

def _suffix_mask(h, N, M):
    shape = list(h.shape)
    for i in range(1, len(shape)-1):
        shape[i] = 1
    out = torch.zeros(*shape, dtype=torch.bool)
    for i, (m, n) in enumerate(zip(M, N)):
        out[i, ..., m:n] = True
    return out.to(h.device)

def _suffix_cat(h1, N1, h2, N2, M):
    N = N1 + N2 - M
    h = F.pad(h1, (0, N.max().item() - h1.shape[-1]))
    src_mask = _suffix_mask(h2, N2, M)
    dst_mask = _suffix_mask(h, N, N1)
    h.masked_scatter_(dst_mask, h2.masked_select(src_mask))
    return h, N

def _batch_split_and_unpad(h, N):
    _,B = torch.unique_consecutive(N, return_counts=True)
    Ns = torch.split(N, B.tolist(), dim=0)
    hs = torch.split(h, B.tolist(), dim=0)
    return [h[...,:N[0].item()] for h,N in zip(hs,Ns)]

class _ReduceBase(nn.Module):
    def __init__(self,
            n_hidden, dropout, init_gate_bias,
            conv_cls, kernel_size, stride, padding):
        super().__init__()
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.init_gate_bias = init_gate_bias
        self.drop = nn.Dropout(dropout)
        self.conv = conv_cls(
            in_channels = n_hidden,
            out_channels = n_hidden * 3,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding)
        if init_gate_bias is not None:
            self.conv.bias.data[(n_hidden*2):].fill_(init_gate_bias)

    def _reduce(self, h):
        h = self.drop(h)
        nh = self.n_hidden
        lrg = self.conv(h)
        l = lrg[:,:nh]
        r = torch.tanh(lrg[:,nh:(nh*2)])
        g = torch.sigmoid(lrg[:,(nh*2):])
        return l*g + r*(1-g)

    def forward(self, h, N):
        Nfp2 = 2 ** N.float().log2().floor().long()
        M = 2 * (N - Nfp2)
        if torch.any(M > 0):
            hp = self._reduce(_prefix(h, M))
            h, N = _suffix_cat(hp, M//2, h, N, M)
        hs = _batch_split_and_unpad(h, N)
        for i in range(len(hs)-1):
            while hs[i].shape[-1] > hs[i+1].shape[-1]:
                hs[i] = self._reduce(hs[i])
            hs[i+1] = torch.cat((hs[i], hs[i+1]), dim=0)
        h = hs[-1]
        while h.shape[-1] > 1:
            h = self._reduce(h)
        return h.squeeze(-1)

class Reduce(_ReduceBase):
    def __init__(self, n_hidden, dropout=0.0, init_gate_bias=1.0):
        super().__init__(
            n_hidden = n_hidden,
            dropout = dropout,
            init_gate_bias = init_gate_bias,
            conv_cls = nn.Conv1d,
            kernel_size = 2,
            stride = 2,
            padding = 0)

class ConvReduce(_ReduceBase):
    def __init__(self, n_hidden,
            dropout=0.0, init_gate_bias=1.0,
            kernel_size = 3):
        if kernel_size % 2 == 0:
            raise ValueError('ConvReduce kernel_size must be odd')
        super().__init__(
            n_hidden = n_hidden,
            dropout = dropout,
            init_gate_bias = init_gate_bias,
            conv_cls = nn.Conv2d,
            kernel_size = (kernel_size, 2),
            stride = (1, 2),
            padding = ((kernel_size - 1) // 2, 0))
