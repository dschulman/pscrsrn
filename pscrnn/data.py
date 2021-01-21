import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import requests
import scipy.io as spio
import shutil
import torch
import torch.nn.utils.rnn as tnur
import torch.utils.data as tud
import zipfile

class _Cinc2017Dataset(tud.Dataset):
    _CATS = ['N','A','O','~']

    def __init__(
            self,
            data_path = 'data/cinc2017',
            ref_file = 'REFERENCE.csv'):
        self.data_path = data_path
        self.ref_path = os.path.join(data_path, ref_file)
        self.ref = pd.read_csv(self.ref_path, names=['id','label'])
        self.ref['label'] = pd.Categorical(self.ref['label'], self._CATS)
        self.ref['code'] = self.ref['label'].cat.codes

    def __len__(self):
        return self.ref.shape[0]

    def __getitem__(self, idx):
        mat_id, _, y = self.ref.iloc[idx]
        mat_path = os.path.join(self.data_path, mat_id+'.mat')
        x = spio.loadmat(mat_path)['val'][0].astype(np.float32)
        x = (x - x.mean()) / x.std()
        return x, y

    def stratify(self):
        return [
            tud.Subset(self, self.ref.index[self.ref['label']==c].values)
            for c in self._CATS
        ]

def _collate(batch):
    xs, Ns, ys = zip(*((torch.tensor(x), x.shape[0], y) for x, y in batch))
    x = tnur.pad_sequence(xs, batch_first=True)
    N = torch.tensor(Ns, dtype=torch.int)
    y = torch.tensor(ys)
    N, sorted_indices = torch.sort(N, descending=True)
    return x[sorted_indices], N, y[sorted_indices]

class Cinc2017Data(pl.LightningDataModule):
    def __init__(
            self,
            url = 'https://www.physionet.org/files/challenge-2017/1.0.0/training2017.zip?download',
            data_path = 'data/cinc2017',
            train_pct = 0.7,
            split_seed = 12345,
            batch_size = 64):
        super().__init__()
        self.url = url
        self.data_path = data_path
        self.train_pct = train_pct
        self.split_seed = split_seed
        self.batch_size = batch_size

    def prepare_data(self):
        if not os.path.exists(self.data_path):
            tmp_path = self.data_path + '_tmp'
            os.makedirs(tmp_path, exist_ok=True)
            zip_path = os.path.join(tmp_path, 'training2017.zip')
            if not os.path.exists(zip_path):
                with requests.get(self.url, stream=True) as r:
                    with open(zip_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
            shutil.move(os.path.join(tmp_path, 'training2017'), self.data_path)
            shutil.rmtree(tmp_path)

    def setup(self):
        ds = _Cinc2017Dataset(self.data_path)
        gen = torch.Generator().manual_seed(self.split_seed)
        splits = []
        for ds1 in ds.stratify():
            train_size = int(self.train_pct * len(ds1))
            val_size = len(ds1) - train_size
            splits.append(tud.random_split(ds1, [train_size, val_size], gen))
        train_datas, val_datas = zip(*splits)
        self.train_data = tud.ConcatDataset(train_datas)
        self.val_data = tud.ConcatDataset(val_datas)

    def train_dataloader(self):
        return tud.DataLoader(
            self.train_data, self.batch_size, shuffle=True,
            collate_fn=_collate)

    def val_dataloader(self):
        return tud.DataLoader(
            self.val_data, self.batch_size, shuffle=False,
            collate_fn=_collate)
