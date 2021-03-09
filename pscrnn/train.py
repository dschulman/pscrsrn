import hydra
import pytorch_lightning as pl
from . import data, classify

class LogHparamsCallback(pl.Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        metrics = { m: 0 for m in pl_module.metrics }
        self.logger.log_hyperparams(pl_module.hparams, metrics)

@hydra.main(config_path='../conf', config_name='default')
def run(cfg):
    dm = data.Cinc2017(
        base_path = hydra.utils.get_original_cwd(),
        **cfg['data'])
    m = classify.Classify(
        features = dm.n_features,
        classes = dm.n_classes,
        exhparams = {**dm.hparams, **cfg['train'] },
        **cfg['model'])
    tb_logger = pl.loggers.TensorBoardLogger('.', name='', version='log', default_hp_metric=False)
    csv_logger = pl.loggers.CSVLogger('.', name='', version='log')
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='checkpoint')
    hp_cb = LogHparamsCallback(tb_logger)
    trainer = pl.Trainer(
        logger = [tb_logger, csv_logger],
        callbacks = [ckpt_cb, hp_cb],
        gpus=1,
        **cfg['train'])
    trainer.fit(model=m, datamodule=dm)

if __name__=='__main__':
    run()
