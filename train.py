import hydra
import pytorch_lightning as pl
from pscrnn import classify, data

@hydra.main(config_name='config')
def run(cfg):
    dm = data.Cinc2017Data(
        base_path = hydra.utils.get_original_cwd(),
        **cfg.get('data', {}))
    m = classify.Classify(
        n_in = dm.n_in, 
        classes = dm.CLASSES,
        **cfg.get('model', {}))
    logger = pl.loggers.TensorBoardLogger(save_dir='.', name='')
    early_stop = pl.callbacks.EarlyStopping('loss/val', **cfg.get('early_stop', {}))
    checkpoint = pl.callbacks.ModelCheckpoint(**cfg.get('checkpoint', {}))
    trainer = pl.Trainer(
        logger = logger,
        callbacks = [early_stop, checkpoint],
        **cfg.get('trainer', {}))
    trainer.fit(m, datamodule=dm)

if __name__=='__main__':
    run()
