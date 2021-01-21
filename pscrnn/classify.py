import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from . import pst

class Classify(pl.LightningModule):
    def __init__(self, n_in, n_classes, n_hidden=128, init_gate_bias=1.0, lr=1e-3):
        super().__init__()
        self.n_hidden = n_hidden
        self.init_gate_bias = init_gate_bias
        self.lr = lr
        self.inproj = nn.Conv1d(
            in_channels = n_in,
            out_channels = n_hidden,
            kernel_size = 1)
        self.reduce = pst.Reduce(
            n_hidden = n_hidden,
            init_gate_bias = init_gate_bias)
        self.outproj = nn.Linear(
            in_features = n_hidden,
            out_features = n_classes)
        self.train_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.val_acc = pl.metrics.Accuracy(compute_on_step=False)
        self.train_f1 = pl.metrics.F1(n_classes, average='macro', compute_on_step=False)
        self.val_f1 = pl.metrics.F1(n_classes, average='macro', compute_on_step=False)

    def forward(self, x, N):
        h = self.inproj(x)
        h = self.reduce(h, N)
        return self.outproj(h)
    
    def training_step(self, batch, batch_idx):
        x, N, y = batch
        z = self(x, N)
        loss = F.cross_entropy(z, y)
        self.log('loss/train', loss, on_step=False, on_epoch=True)
        self.train_acc(z, y)
        self.train_f1(z, y)
        return loss

    def training_epoch_end(self, outs):
        self.log('acc/train', self.train_acc.compute())
        self.log('f1/train', self.train_f1.compute())

    def validation_step(self, batch, batch_idx):
        x, N, y = batch
        z = self(x, N)
        loss = F.cross_entropy(z, y)
        self.log('loss/val', loss, on_step=False, on_epoch=True)
        self.val_acc(z, y)
        self.val_f1(z, y)
        return loss

    def validation_epoch_end(self, outs):
        self.log('acc/val', self.val_acc.compute())
        self.log('f1/val', self.val_f1.compute())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
