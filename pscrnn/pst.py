import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SeqConv(nn.Module):
    def __init__(self, 
            in_channels, out_channels, kernel_size, stride,
            weight_norm = False):
        if (kernel_size % 2) == 0:
            raise ValueError('kernel_size should be odd')
        if stride > kernel_size:
            raise ValueError('stride should be <= kernel_size')
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_norm = weight_norm
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride = stride,
            padding = (kernel_size - 1) // 2)
        if weight_norm:
            nn.utils.weight_norm(self.conv)
    
    def forward(self, x, N):
        mask = torch.arange(x.shape[2], device=N.device) < N.unsqueeze(1)
        x = self.conv(x * mask.unsqueeze(1))
        N = torch.floor((N-1) / self.stride).long() + 1
        return x, N

class ReduceBlock(nn.Module):
    def __init__(self,
            channels, kernel_size, stride, 
            weight_norm=True, depth_variant=True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight_norm = weight_norm
        self.depth_variant = depth_variant
        self.conv1 = SeqConv(
            in_channels = channels + (1 if depth_variant else 0),
            out_channels = channels * 2,
            kernel_size = kernel_size,
            stride = stride,
            weight_norm = weight_norm)
        self.conv2 = SeqConv(
            in_channels = channels,
            out_channels = channels,
            kernel_size = kernel_size,
            stride = 1,
            weight_norm = weight_norm)
        self.act = nn.ReLU()

    def forward(self, h, N, depth):
        c = self.channels
        if self.depth_variant:
            dshape = (h.shape[0], 1, h.shape[2])
            d = torch.full(dshape, math.log1p(depth), device=h.device)
            h = torch.cat([h,d], dim=1)
        lr, N = self.conv1(h, N)
        l = lr[:,:c]
        r = lr[:,c:]
        r = self.act(r)
        r, _ = self.conv2(r, N)
        return self.act(l + r), N

class Reduce(nn.Module):
    def __init__(self, 
            hidden, kernel_size, stride, layers, depth_variant, 
            dropout, weight_norm):
        super().__init__()
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.layers = layers
        self.depth_variant = depth_variant
        self.dropout = dropout
        self.weight_norm = weight_norm
        self.blocks = nn.ModuleList([
            ReduceBlock(hidden, kernel_size, stride, depth_variant, weight_norm)
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

class Classify(pl.LightningModule):
    def __init__(self,
            features, classes,
            inproj_size=8, inproj_stride=4,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            dropout=0.2, weight_norm=True,
            lr=1e-3, weight_decay=1e-2,
            exhparams={}):
        super().__init__()
        self.save_hyperparameters({
            'inproj_size': inproj_size,
            'inproj_stride': inproj_stride,
            'hidden': hidden,
            'kernel_size': kernel_size,
            'stride': stride,
            'layers': layers,
            'depth_variant': depth_variant,
            'dropout': dropout,
            'weight_norm': weight_norm,
            'lr': lr,
            'weight_decay': weight_norm,
            **exhparams
        })
        self.metrics = [
            'acc/train', 'acc/val',
            'f1/train', 'f1/val']
        self.inproj_size = inproj_size
        self.inproj_stride = inproj_stride
        self.lr = lr
        self.weight_decay = weight_decay
        self.inproj = nn.Conv1d(
            in_channels = features,
            out_channels = hidden,
            kernel_size = inproj_size,
            stride = inproj_stride)
        self.reduce = Reduce(
            hidden = hidden,
            kernel_size = kernel_size,
            stride = stride,
            layers = layers,
            depth_variant = depth_variant,
            dropout = dropout,
            weight_norm = weight_norm)
        self.outproj = nn.Linear(
            in_features = hidden,
            out_features = classes)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.val_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.train_f1 = pl.metrics.F1(classes, average='macro', compute_on_step=False)
        self.val_f1 = pl.metrics.F1(classes, average='macro', compute_on_step=False)

    def forward(self, x, N):
        h = self.inproj(x)
        N = torch.floor(((N - self.inproj_size) / self.inproj_stride) - 1).long()
        h = self.reduce(h, N)
        return self.outproj(h)

    def training_step(self, batch, batch_idx):
        x, N, y = batch
        z = self(x, N)
        loss = self.loss(z, y)
        self.log('loss/train', loss, on_step=False, on_epoch=True)
        self.train_acc(torch.argmax(z, dim=1), y)
        self.train_f1(z, y)
        return loss

    def training_epoch_end(self, outs):
        self.log('acc/train', self.train_acc.compute())
        self.log('f1/train', self.train_f1.compute())

    def validation_step(self, batch, batch_idx):
        x, N, y = batch
        z = self(x, N)
        loss = self.loss(z, y)
        self.log('loss/val', loss, on_step=False, on_epoch=True)
        self.val_acc(torch.argmax(z, dim=1), y)
        self.val_f1(z, y)
        return loss

    def validation_epoch_end(self, outs):
        self.log('acc/val', self.val_acc.compute())
        self.log('f1/val', self.val_f1.compute())

    def configure_optimizers(self):
        return optim.AdamW(
            params = self.parameters(), 
            lr = self.lr,
            weight_decay = self.weight_decay)
