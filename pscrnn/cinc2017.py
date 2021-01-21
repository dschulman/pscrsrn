import pytorch_lightning as pl
from . import classify, data

def run():
    dm = data.Cinc2017Data()
    m = classify.Classify(
        n_in = dm.N_IN, 
        n_classes = dm.N_CLASSES,
        n_hidden = 32)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(m, datamodule=dm)

if __name__ == '__main__':
    run()
