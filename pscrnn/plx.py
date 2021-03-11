import pytorch_lightning as pl
import torch.utils.tensorboard as tensorboard

class TboardHParamsCallback(pl.Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_train_start(self, trainer, pl_module):
        params = pl_module.hparams
        metrics = {m: 0 for m in pl_module.metrics}
        exp, ssi, sei = tensorboard.summary.hparams(params, metrics)
        writer = self.logger.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
