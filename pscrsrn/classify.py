import torch
import torch.nn as nn
import torchmetrics as tmet
from . import cm, data, resnet, rnn, rsrn, train

class Task(train.Task):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
        n_classes = len(classes)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            "accuracy": tmet.Accuracy(compute_on_step=False),
            "f1": tmet.F1(n_classes, average='macro', compute_on_step=False),
            "auc": tmet.AUROC(n_classes, average='macro', compute_on_step=False),
            "cm": tmet.ConfusionMatrix(n_classes, normalize='true', compute_on_step=False)
        })

    scalars = ['accuracy','f1','auc']

    def start_stage(self, stage):
        for metric in self.metrics.values():
            metric.reset()

    def step(self, model, batch):
        x, y = batch
        z = model(x)
        with torch.no_grad():
            zmax = torch.argmax(z, dim=1)
            self.metrics.accuracy(zmax, y)
            self.metrics.f1(zmax, y)
            self.metrics.auc(torch.softmax(z, dim=1), y)
            self.metrics.cm(zmax, y)
        return self.loss(z, y), y.shape[0]

    def finish_stage(self, stage):
        return {
            'accuracy': self.metrics.accuracy.compute().item(),
            'f1': self.metrics.f1.compute().item(),
            'auc': self.metrics.auc.compute().item(),
            'cm': cm.plot(self.metrics.cm.compute().cpu().numpy(), self.classes)
        }

def run(hparams=None):
    d = data.Cinc2017()
    def model(**hparams):
        mtype = hparams['type']
        hparams = hparams.copy()
        del hparams['type']
        if mtype == 'rsrn':
            return rsrn.Classify(d.n_features, d.n_classes, **hparams)
        elif mtype == 'rnn':
            return rnn.Classify(d.n_features, d.n_classes, **hparams)
        elif mtype == 'resnet':
            return resnet.Classify(d.n_features, d.n_classes, **hparams)
        raise ValueError('unknown model type: ' + mtype)
    train.run(
        hparams = hparams,
        conf_dir = 'conf/classify',
        default_out = 'outputs',
        model_con = model,
        data_con = d,
        task_con = lambda **hparams: Task(d.CATS, **hparams),
        gpu = True)

if __name__ == '__main__':
    run()
