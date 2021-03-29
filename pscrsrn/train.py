import argparse
import csv
import datetime
import matplotlib.pyplot as plt
import omegaconf as oc
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.utils.tensorboard as tut
import torchmetrics as tmet
from tqdm.auto import tqdm, trange

def _parse_args(hparams, default_out):
    if hparams is not None:
        return default_out, oc.OmegaConf.create(hparams)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-o', '--out', default=default_out,
            help='Output directory')
        parser.add_argument(
            'base',
            help='Base hyperparameter file')
        parser.add_argument(
            'override', nargs='*', 
            help='Hyperparameter overrides')
        args = parser.parse_args()
        hparams = oc.OmegaConf.merge(
            oc.OmegaConf.load(args.base),
            oc.OmegaConf.from_cli(args.override))
        return args.out, hparams

def _nparams(model):
    return sum(torch.numel(p) for p in model.parameters() if p.requires_grad)

def _prefixed(s, prefix):
    return s if prefix is None else prefix+'/'+s

def _flat_hparams(hparams, prefix=None):
    out = {}
    for k,v in hparams.items():
        if isinstance(v, oc.DictConfig):
            out.update(_flat_hparams(v, _prefixed(k, prefix)))
        else:
            out[_prefixed(k, prefix)] = v
    return out

def _tboard_hparams(tb, hparams, metrics):
    hparams = _flat_hparams(hparams)
    metrics = [m+'/train' for m in metrics] + [m+'/val' for m in metrics]
    metrics = {m:0.0 for m in metrics}
    for s in tut.summary.hparams(hparams, metrics):
        tb.file_writer.add_summary(s)

def _tboard_metrics(tb, mets, suffix, e):
    for k,v in mets.items():
        if isinstance(v, float):
            tb.add_scalar(k+suffix, v, e)
        elif isinstance(v, plt.Figure):
            tb.add_figure(k+suffix, v, e)

def _csv_metrics(mets):
    return {k:v for k,v in mets.items() if isinstance(v,float)}

def _batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, tuple):
        return tuple(_batch_to_device(b, device) for b in batch)
    elif isinstance(batch, list):
        return [_batch_to_device(b, device) for b in batch]
    else:
        raise ValueError()

def run(
        hparams,
        default_out,
        model_con,
        data_con,
        loss_con,
        metrics_con,
        gpu = True,
        val_every_n_epochs = 1):
    output, hparams = _parse_args(hparams, default_out)
    name = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    output = os.path.join(output, name)
    os.makedirs(output, exist_ok = True)
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model = model_con(**hparams.get('model', {}))
    hparams['nparams'] = _nparams(model)
    oc.OmegaConf.save(hparams, os.path.join(output, 'hparams.yaml'))
    model.to(device)
    train_data, val_data = data_con(**hparams.get('data', {}))
    loss_fn = loss_con(**hparams.get('loss', {}))
    loss_fn.to(device)
    train_metrics = metrics_con()
    train_metrics.to(device)
    val_metrics = metrics_con()
    val_metrics.to(device)
    train_hparams = hparams['train']
    optimizer = optim.AdamW(
        params = model.parameters(),
        lr = train_hparams['lr'],
        weight_decay = train_hparams['weight_decay'])
    csvpath = os.path.join(output, 'metrics.csv')
    with open(csvpath, 'w', newline='') as csvf, tut.SummaryWriter(output) as tb:
        csvw = csv.DictWriter(csvf, ['epoch','stage','loss'] + train_metrics.scalars)
        csvw.writeheader()
        _tboard_hparams(tb, hparams, ['loss'] + train_metrics.scalars)
        with trange(train_hparams['epochs'], desc='Epoch') as et:
            for e in et:
                model.train()
                total_loss = 0.0
                total_len = 0
                with tqdm(train_data, desc='Train', leave=False) as bt:
                    for b, batch in enumerate(bt):
                        x, y = _batch_to_device(batch, device)
                        optimizer.zero_grad()
                        y_pred = model(x)
                        loss = loss_fn(y_pred, y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        total_len += y.shape[0]
                        bt.set_postfix(Loss=total_loss/total_len)
                        train_metrics(y_pred, y)
                train_loss = total_loss / total_len
                train_mets = train_metrics.compute()
                train_metrics.reset()
                csvw.writerow({'epoch':e, 'stage':'train', 'loss':train_loss, **_csv_metrics(train_mets)})
                csvf.flush()
                tb.add_scalar('loss/train', train_loss, e)
                _tboard_metrics(tb, train_mets, '/train', e)
                if ((e+1) % val_every_n_epochs) == 0:
                    model.eval()
                    with torch.no_grad():
                        total_loss = 0.0
                        total_len = 0
                        with tqdm(val_data, desc='Val', leave=False) as bt:
                            for b, batch in enumerate(bt):
                                x, y = _batch_to_device(batch, device)
                                y_preds = model(x)
                                y_preds = torch.stack(torch.chunk(y_preds, y.shape[0]))
                                y_pred = torch.mean(y_preds, dim=1)
                                loss = loss_fn(y_pred, y)
                                total_loss += loss.item()
                                total_len += y.shape[0]
                                bt.set_postfix(Loss=total_loss/total_len)
                                val_metrics(y_pred, y)
                    val_loss = total_loss / total_len
                    val_mets = val_metrics.compute()
                    val_metrics.reset()
                    csvw.writerow({'epoch':e, 'stage':'val', 'loss':val_loss, **_csv_metrics(val_mets)})
                    csvf.flush()
                    tb.add_scalar('loss/val', val_loss, e)
                    _tboard_metrics(tb, val_mets, '/val', e)
                    et.set_postfix(Train=train_loss, Val=val_loss)
                torch.save(
                    {'epoch':e, 'model':model.state_dict(), 'optim':optimizer.state_dict()},
                    os.path.join(output, 'checkpoint.pt'))
