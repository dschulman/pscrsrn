import pytorch_lightning as pl

class LogHparamsCallback(pl.Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        metrics = { m: 0 for m in pl_module.metrics }
        self.logger.log_hyperparams(pl_module.hparams, metrics)