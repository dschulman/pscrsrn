import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from . import pst

class Classify(pl.LightningModule):
    def __init__(self,
            features, classes,
            inproj_size=8, inproj_stride=4,
            hidden=64, kernel_size=5, stride=2, layers=2, depth_variant=True,
            dropout=0.2, leak=0.0, weight_norm=True,
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
            'leak': leak,
            'dropout': dropout,
            'weight_norm': weight_norm,
            'lr': lr,
            'weight_decay': weight_decay,
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
        self.reduce = pst.Reduce(
            hidden = hidden,
            kernel_size = kernel_size,
            stride = stride,
            layers = layers,
            depth_variant = depth_variant,
            leak = leak,
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
