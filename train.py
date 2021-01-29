import hydra
import pytorch_lightning as pl
from pscrnn import classify, data

@hydra.main(config_path='conf', config_name='config')
def run(cfg):
    dm = data.Cinc2017Data(
        base_path = hydra.utils.get_original_cwd(),
        trans = hydra.utils.instantiate(cfg['trans']),
        **cfg.get('data', {}))
    m = classify.Classify(
        n_in = dm.n_features, 
        classes = dm.CLASSES,
        model = cfg['model'],
        loss_type = cfg['loss'],
        optim = cfg['optim'],
        sched = cfg['sched'])
    trainer = pl.Trainer(
        logger = pl.loggers.TensorBoardLogger(save_dir='.', name=''),
        callbacks = [
            pl.callbacks.EarlyStopping('loss/val', **cfg.get('early_stop', {})),
            pl.callbacks.ModelCheckpoint(**cfg.get('checkpoint', {})),
            pl.callbacks.LearningRateMonitor('epoch')
        ],
        **cfg.get('trainer', {}))
    trainer.fit(m, datamodule=dm)

if __name__=='__main__':
    run()
