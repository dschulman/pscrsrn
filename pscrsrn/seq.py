import torch
import torch.nn as nn

def mask(x, N):
    return (torch.arange(x.shape[2], device=N.device) < N.unsqueeze(1)).unsqueeze(1)

class _ConvBase(nn.Module):
    def __init__(self, conv_cls, 
            in_channels, out_channels, kernel_size, stride, pad_delta):
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
        self.conv = conv_cls(
            in_channels, out_channels, kernel_size,
            stride = stride,
            padding = (kernel_size - pad_delta) // 2)

class Conv(_ConvBase):
    def __init__(self,
            in_channels, out_channels, kernel_size, stride, pad_delta):
        super().__init__(nn.Conv1d, in_channels, out_channels, kernel_size, stride, pad_delta)

    def forward(self, x, N):
        x = self.conv(x * mask(x, N))
        N = torch.floor((N - self.pad_delta) / self.stride).long() + 1
        return x, N

class ConvTranspose(_ConvBase):
    def __init__(self, 
            in_channels, out_channels, kernel_size, stride, pad_delta):
        super().__init__(nn.ConvTranspose1d, in_channels, out_channels, kernel_size, stride, pad_delta)

    def forward(self, x, N):
        x = self.conv(x * mask(x, N))
        N = ((N-1) * self.stride) + self.pad_delta
        return x, N

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.log_weight = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, N):
        m = mask(x, N)
        N = N.unsqueeze(-1).unsqueeze(-1)
        mu = torch.sum(x * m, dim=[1,2], keepdim=True) / x.shape[1] / N
        z = x - mu
        var = torch.sum(z*z * m, dim=[1,2], keepdim=True) / x.shape[1] / N
        w = torch.exp(self.log_weight)
        b = self.bias
        return w * z * torch.rsqrt(var + self.eps) + b 

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.log_weight = nn.Parameter(torch.zeros(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x, N):
        m = mask(x, N)
        denom = x.shape[0] * x.shape[2]
        mu = torch.sum(x * m, dim=[0,2], keepdim=True) / denom
        z = x - mu
        var = torch.sum(z*z * m, dim=[0,2], keepdim=True) / denom
        if self.training:
            mom = self.momentum
            if self.num_batches_tracked == 0:
                self.running_mean = mu
                self.running_var = var
            else:
                self.running_mean = ((1-mom) * self.running_mean) + (mom * mu)
                self.running_var = ((1-mom) * self.running_var) + (mom * var)
            self.num_batches_tracked += 1
            y = z * torch.rsqrt(var + self.eps)
        else:
            y = (x - self.running_mean) * torch.rsqrt(self.running_var + self.eps)
        return y * torch.exp(self.log_weight) + self.bias
