import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from . import pst

class Classify(pl.LightningModule):
    def __init__(self, n_in, classes, input_dropout=0.0, n_hidden=128, dropout=0.0, init_gate_bias=1.0, lr=1e-3):
        super().__init__()
        n_classes = len(classes)
        self.classes = classes
        self.input_dropout = input_dropout
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.init_gate_bias = init_gate_bias
        self.lr = lr
        self.indrop = nn.Dropout(
            p = input_dropout)
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
        self.train_cm = pl.metrics.ConfusionMatrix(n_classes, compute_on_step=False)
        self.val_cm = pl.metrics.ConfusionMatrix(n_classes, compute_on_step=False)

    def forward(self, x, N):
        x = self.indrop(x)
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
        self.train_cm(z, y)
        return loss

    def training_epoch_end(self, outs):
        self.log('acc/train', self.train_acc.compute())
        self.log('f1/train', self.train_f1.compute())
        cm = self.train_cm.compute()
        self.logger.experiment.add_figure('cm/train', self._plot_cm(cm), self.trainer.global_step)

    def validation_step(self, batch, batch_idx):
        x, N, y = batch
        z = self(x, N)
        loss = F.cross_entropy(z, y)
        self.log('loss/val', loss, on_step=False, on_epoch=True)
        self.val_acc(z, y)
        self.val_f1(z, y)
        self.val_cm(z, y)
        return loss

    def validation_epoch_end(self, outs):
        self.log('acc/val', self.val_acc.compute())
        self.log('f1/val', self.val_f1.compute())
        if not self.trainer.running_sanity_check:
            cm = self.val_cm.compute()
            self.logger.experiment.add_figure('cm/val', self._plot_cm(cm), self.trainer.global_step)
        else:
            self.val_cm.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def _plot_cm(self, cm):
        cm = cm.cpu().detach().numpy()
        fig = plt.figure(figsize=(8,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        ticks = np.arange(len(self.classes))
        plt.xticks(ticks, self.classes)
        plt.yticks(ticks, self.classes)
        cm_norm = np.around(
            cm.astype(np.float) / cm.sum(axis=1, keepdims=True), 
            decimals=2)
        wb_threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, cm_norm[i, j], 
                    horizontalalignment = 'center',
                    color = 'white' if cm[i,j]>wb_threshold else 'black')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return fig
