import hydra
import pytorch_lightning as pl
from pscrnn import classify, data

class LogHparamsCallback(pl.Callback):
    def __init__(self, hparams, metrics):
        self.hparams = hparams
        self.metrics = metrics

    def on_train_end(self, trainer, pl_module):
        hparams = {
            **pl_module.hparams,
            **self.hparams
        }
        metrics = {
            k: trainer.callback_metrics[v]
            for k, v in self.metrics.items()
        }
        trainer.logger.log_hyperparams(pl_module.hparams, metrics)

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
        logger = pl.loggers.TensorBoardLogger(save_dir='.', name='', default_hp_metric=False),
        callbacks = [
            pl.callbacks.EarlyStopping('loss/val', **cfg.get('early_stop', {})),
            pl.callbacks.ModelCheckpoint(**cfg.get('checkpoint', {})),
            pl.callbacks.LearningRateMonitor('epoch'),
            LogHparamsCallback( ## TODO shouldn't hardcode metrics
                hparams = {'data': cfg['data'], 'trans': cfg['trans']},
                metrics = {'acc/metric': 'acc/val', 'f1/metric': 'f1/val' })
        ],
        **cfg.get('trainer', {}))
    trainer.fit(m, datamodule=dm)

if __name__=='__main__':
    run()
