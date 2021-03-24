import torch
import torch.nn as nn
import torchmetrics as tmet
from . import data, pst, train

class Metrics(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.accuracy = tmet.Accuracy(compute_on_step=False)
        self.precision = tmet.Precision(n_classes, average='macro', compute_on_step=False)
        self.recall = tmet.Recall(n_classes, average='macro', compute_on_step=False)
        self.f1 = tmet.F1(n_classes, average='macro', compute_on_step=False)
        self.auc = tmet.AUROC(n_classes, average='macro', compute_on_step=False)

    scalars = ['accuracy','precision','recall','f1','auc']

    def reset(self):
        for c in self.children():
            c.reset()

    def forward(self, z, y):
        with torch.no_grad():
            zmax = torch.argmax(z, dim=1)
            self.accuracy(zmax, y)
            self.precision(zmax, y)
            self.recall(zmax, y)
            self.f1(zmax, y)
            self.auc(torch.softmax(z, dim=1), y)

    def compute(self):
        return {
            'accuracy': self.accuracy.compute().item(),
            'precision': self.precision.compute().item(),
            'recall': self.recall.compute().item(),
            'f1': self.f1.compute().item(),
            'auc': self.auc.compute().item()
        }

def main():
    d = data.Cinc2017()
    train.run(
        default_out = 'outputs',
        default_conf = 'default.yaml',
        model_con = lambda **hparams: pst.Classify(d.n_features, d.n_classes, **hparams),
        data_con = d,
        loss_con = nn.CrossEntropyLoss,
        metrics_con = lambda: Metrics(d.n_classes))

if __name__ == '__main__':
    main()
