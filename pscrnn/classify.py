import torch
import torch.nn as nn
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
        loss_con = nn.CrossEntropyLoss)

if __name__ == '__main__':
    main()
