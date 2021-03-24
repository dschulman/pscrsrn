import torch
import torch.nn as nn
import torchmetrics as tmet
from . import cm, data, pst, train

class Metrics(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        n_classes = len(classes)
        self.accuracy = tmet.Accuracy(compute_on_step=False)
        self.f1 = tmet.F1(n_classes, average='macro', compute_on_step=False)
        self.auc = tmet.AUROC(n_classes, average='macro', compute_on_step=False)
        self.cm = tmet.ConfusionMatrix(n_classes, normalize='true', compute_on_step=False)

    scalars = ['accuracy','f1','auc']

    def reset(self):
        for c in self.children():
            c.reset()

    def forward(self, z, y):
        with torch.no_grad():
            zmax = torch.argmax(z, dim=1)
            self.accuracy(zmax, y)
            self.f1(zmax, y)
            self.auc(torch.softmax(z, dim=1), y)
            self.cm(zmax, y)

    def compute(self):
        return {
            'accuracy': self.accuracy.compute().item(),
            'f1': self.f1.compute().item(),
            'auc': self.auc.compute().item(),
            'cm': cm.plot(self.cm.compute().cpu().numpy(), self.classes)
        }

def main():
    d = data.Cinc2017()
    train.run(
        default_out = 'outputs',
        default_conf = 'default.yaml',
        model_con = lambda **hparams: pst.Classify(d.n_features, d.n_classes, **hparams),
        data_con = d,
        loss_con = nn.CrossEntropyLoss,
        metrics_con = lambda: Metrics(d.CATS))

if __name__ == '__main__':
    main()
