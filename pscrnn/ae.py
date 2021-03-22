import hydra
import pytorch_lightning as pl
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.optim as optim
from . import data, encdec, plx

class AE(pl.LightningModule):
    def __init__(self,
            features, latent,
            inproj_size=8, inproj_stride=4,
            enc_hidden=64, enc_kernel_size=5, enc_stride=2, enc_layers=2, enc_depth_variant=True,
            enc_dropout=0.0, enc_leak=0.0, enc_weight_norm=True, enc_layer_norm=True,
            dec_hidden=64, dec_kernel_size=5, dec_stride=2, dec_layers=2, dec_depth_variant=True,
            dec_dropout=0.0, dec_leak=0.2, dec_weight_norm=True, dec_layer_norm=True,
            outproj_size=5,
            lr=1e-3, weight_decay=1e-2,
            exhparams={}):
        super().__init__()
        self.save_hyperparameters({
            'latent': latent,
            'inproj_size': inproj_size,
            'inproj_stride': inproj_stride,
            'enc_hidden': enc_hidden,
            'enc_kernel_size': enc_kernel_size,
            'enc_stride': enc_stride,
            'enc_layers': enc_layers,
            'enc_depth_variant': enc_depth_variant,
            'enc_leak': enc_leak,
            'enc_dropout': enc_dropout,
            'enc_weight_norm': enc_weight_norm,
            'enc_layer_norm': enc_layer_norm,
            'dec_hidden': dec_hidden,
            'dec_kernel_size': dec_kernel_size,
            'dec_stride': dec_stride,
            'dec_layers': dec_layers,
            'dec_depth_variant': dec_depth_variant,
            'dec_dropout': dec_dropout,
            'dec_leak': dec_leak,
            'dec_weight_norm': dec_weight_norm,
            'dec_layer_norm': dec_layer_norm,
            'outproj_size': outproj_size,
            'lr': lr,
            'weight_decay': weight_decay,
            **exhparams
        })
        self.metrics = [
            'loss/train',
            'loss/val'
        ]
        self.latent = latent
        self.lr = lr
        self.weight_decay = weight_decay
        self.encoder = encdec.Encode(
            features = features,
            latent = latent,
            inproj_size = inproj_size,
            inproj_stride = inproj_stride,
            hidden = enc_hidden,
            kernel_size = enc_kernel_size,
            stride = enc_stride,
            layers = enc_layers,
            depth_variant = enc_depth_variant,
            leak = enc_leak,
            dropout = enc_dropout,
            weight_norm = enc_weight_norm,
            layer_norm = enc_layer_norm)
        self.decoder = encdec.Decode(
            features = features,
            latent = latent,
            hidden = dec_hidden,
            kernel_size = dec_kernel_size,
            stride = dec_stride,
            layers = dec_layers,
            depth_variant = dec_depth_variant,
            leak = dec_leak,
            dropout = dec_dropout,
            weight_norm = dec_weight_norm,
            layer_norm = dec_layer_norm,
            outproj_size = outproj_size)
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, x, N):
        z = self.encoder(x, N)
        y = self.decoder(z, N)
        return y

    def step(self, batch, batch_idx, log_suffix):
        x, N, _ = batch
        y = self(x, N)
        mask = torch.arange(x.shape[2], device=N.device) < N.unsqueeze(1)
        loss = self.loss(y, x) * mask.unsqueeze(1)
        loss = torch.sum(loss) / torch.sum(N) / x.shape[1]
        self.log(f'loss/{log_suffix}', loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, 'val')

    def configure_optimizers(self):
        return optim.AdamW(
            params = self.parameters(),
            lr = self.lr,
            weight_decay = self.weight_decay)

@hydra.main(config_path='../conf', config_name='ae')
def run(cfg):
    dm = data.Cinc2017(
        base_path = hydra.utils.get_original_cwd(),
        **cfg['data'])
    m = AE(
        features = dm.n_features,
        exhparams = {**dm.hparams, **cfg['train'] },
        **cfg['model'])
    tb_logger = pl.loggers.TensorBoardLogger('.', name='', version='log', default_hp_metric=False)
    csv_logger = pl.loggers.CSVLogger('.', name='', version='log')
    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath='checkpoint')
    tbhp_cb = plx.TboardHParamsCallback(tb_logger)
    trainer = pl.Trainer(
        logger = [tb_logger, csv_logger],
        callbacks = [ckpt_cb, tbhp_cb],
        gpus=1,
        **cfg['train'])
    trainer.fit(model=m, datamodule=dm)

if __name__=='__main__':
    run()
