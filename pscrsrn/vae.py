import torch
import torch.nn as nn
import torch.nn.utils.rnn as tnur
from . import data, rsrn, train

class VAE(nn.Module):
    def __init__(self, features, latent, encode, decode, kl_coeff=0.1):
        super().__init__()
        self.latent = latent
        self.kl_coeff = kl_coeff
        self.encode = rsrn.Classify(
            features = features, 
            classes = 2*latent,
            **encode)
        self.decode = rsrn.Generate(
            latent = latent, 
            features = features,
            **decode)

    def forward(self, xs):
        nl = self.latent
        N = torch.tensor([x.shape[0] for x in xs], device=xs[0].device, dtype=torch.int)
        mlv = self.encode(xs)
        mu = mlv[:,:nl]
        logvar = mlv[:,nl:]
        kld = torch.sum(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
        z = mu + (torch.randn_like(mu) * torch.exp(logvar / 2))
        xr = self.decode(z, N)
        x = tnur.pad_sequence(xs, batch_first=True).transpose(1,2)
        recon_loss = torch.sum(torch.sum((xr - x)**2, dim=[1,2]) / N)
        loss = recon_loss + self.kl_coeff * kld
        return x, xr, z, loss, kld, recon_loss

class Task(train.Task):
    scalars = ['loss', 'kl_loss', 'recon_loss']

    def start_stage(self, stage):
        self.total_count = 0
        self.total_kl_loss = 0.0
        self.total_recon_loss = 0.0

    def step(self, model, batch):
        xs, y = batch
        _, _, _, loss, kl_loss, recon_loss = model(xs)
        self.total_count += y.shape[0]
        self.total_kl_loss += kl_loss.item()
        self.total_recon_loss += recon_loss.item()
        return loss, y.shape[0]

    def finish_stage(self, stage):
        return {
            'kl_loss': self.total_kl_loss / self.total_count,
            'recon_loss': self.total_recon_loss / self.total_count
        }

def run(hparams=None):
    d = data.Cinc2017()
    train.run(
        hparams = hparams,
        conf_dir = 'conf/vae',
        default_out = 'outputs',
        model_con = lambda **hparams: VAE(d.n_features, **hparams),
        data_con = d,
        task_con = lambda **hparams: Task(),
        gpu = True)

if __name__ == '__main__':
    run()
