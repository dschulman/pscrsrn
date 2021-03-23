import torch
import torch.nn as nn
import torchmetrics as tmet
from . import data, pst, train

def main():
    d = data.Cinc2017()
    def m(**hparams):
        return pst.Classify(
            features = d.n_features, 
            classes = d.n_classes,
            **hparams)
    train.run(
        default_out = 'outputs',
        default_conf = 'default.yaml',
        model_con = m,
        data_con = d,
        loss_con = nn.CrossEntropyLoss,
        metrics = {
            'accuracy': tmet.Accuracy(compute_on_step=False),
            'f1': tmet.F1(d.n_classes, average='macro', compute_on_step=False),
            'precision': tmet.Precision(d.n_classes, average='macro', compute_on_step=False),
            'recall': tmet.Recall(d.n_classes, average='macro', compute_on_step=False)
        })

if __name__ == '__main__':
    main()
