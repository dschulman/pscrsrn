import torch
import torch.nn as nn
import torch.nn.functional as F

class Reduce(nn.Module):
    def __init__(self,
            n_hidden, kernel_size,
            dropout, init_gate_bias):
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        super().__init__()
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.init_gate_bias = init_gate_bias
        self.conv = nn.Conv1d(
            in_channels = n_hidden,
            out_channels = n_hidden * 3,
            kernel_size = kernel_size,
            stride = kernel_size)
        if init_gate_bias is not None:
            self.conv.bias.data[(n_hidden*2):].fill_(init_gate_bias)

    def _reduce(self, h, N):
        nh = self.n_hidden
        ks = self.kernel_size
        Nmax = h.shape[2]
        mask = torch.arange(Nmax, device=N.device).unsqueeze(0) < N.unsqueeze(1)
        h = h * mask.unsqueeze(1)
        if (Nmax % ks) != 0:
            h = F.pad(h, (0, self.kernel_size - (Nmax % ks)))
        lrg = self.conv(h)
        l = lrg[:,:nh]
        r = torch.tanh(lrg[:,nh:(nh*2)])
        g = torch.sigmoid(lrg[:,(nh*2):])
        h = l*g + r*(1-g)
        N = torch.ceil(N / ks).long()
        return h, N

    def forward(self, h, N):
        mask = None
        if self.training and self.dropout > 0.0:
            mask = torch.rand(h.shape[0], h.shape[1], 1, device=h.device) > self.dropout
            mask = mask / (1 - self.dropout)
        out = []
        while h.shape[0] > 0:
            if mask is not None:
                h = mask * h
            h, N = self._reduce(h, N)
            reduced = (N <= 1)
            out.append(h[reduced, :, 0])
            h = h[~reduced]
            N = N[~reduced]
            if mask is not None:
                mask = mask[~reduced]
        return torch.cat(out, dim=0)
