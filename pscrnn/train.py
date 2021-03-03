import hydra
import pytorch_lightning as pl
from . import data, pst

class LogHparamsCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        metrics = {
            'm'+'/final': trainer.callback_metrics[m]
            for m in pl_module.metrics
        }
        trainer.logger.log_hyperparams(pl_module.hparams, metrics)

@hydra.main(config_path='../conf', config_name='default')
def run(cfg):
    dm = data.Cinc2017(
        base_path = hydra.utils.get_original_cwd(),
        **cfg['data'])
    m = pst.Classify(
        features = dm.n_features,
        classes = dm.n_classes,
        **cfg['model'])
    m.hparams.update(dm.hparams)
    m.hparams.update(**cfg['train'])
    tb_logger = pl.loggers.TensorBoardLogger('.', name='', version='log', default_hp_metric=False)
    csv_logger = pl.loggers.CSVLogger('.', name='', version='log')
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='checkpoint')
    hp_cb = LogHparamsCallback()
    trainer = pl.Trainer(
        logger = [tb_logger, csv_logger],
        callbacks = [ckpt_cb, hp_cb],
        gpus=1,
        **cfg['train'])
    trainer.fit(model=m, datamodule=dm)

if __name__=='__main__':
    run()
