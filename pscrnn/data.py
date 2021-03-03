import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import requests
import scipy.io as spio
import scipy.signal as spsig
import shutil
import torch
import torch.nn.utils.rnn as tnur
import torch.utils.data as tud
import tqdm
import zipfile

class Cinc2017Dataset(tud.Dataset):
    def __init__(self, xs, y, weights, trim_prob, trim_min=0.5):
        self.xs = xs
        self.y = y
        self.weights = weights
        self.trim_prob = trim_prob
        self.trim_min = trim_min

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x = self.xs[idx]
        if self.trim_prob and (np.random.random() < self.trim_prob):
            length = x.shape[0]
            trimmed = np.random.randint(int(length*self.trim_min), length)
            offset = np.random.randint(length-trimmed)
            x = x[offset:(offset+trimmed)]
        return x, self.y[idx]

class Cinc2017(pl.LightningDataModule):
    CATS = ['N','A','O','~']
    URL = 'https://www.physionet.org/files/challenge-2017/1.0.0/training2017.zip?download'
    PATH = 'data/cinc2017'

    def __init__(self,
            base_path,
            batch_size = 32,
            trim_prob = 0.9,
            trim_min = 0.5,
            balanced_sampling = True,
            train_pct = 0.7,
            split_seed = 1234):
        super().__init__()
        self.path = os.path.join(base_path, self.PATH)
        self.batch_size = batch_size
        self.trim_prob = trim_prob
        self.trim_min = trim_min
        self.balanced_sampling = balanced_sampling
        self.train_pct = train_pct
        self.split_seed = split_seed
        self.hparams = {
            'batch_size': batch_size,
            'trim_prob': trim_prob,
            'trim_min': trim_min,
            'balanced_sampling': balanced_sampling
        }
        self.n_features = 1
        self.n_classes = len(self.CATS)

    def prepare_data(self):
        if not os.path.exists(self.path):
            tmp_path = self.path + '_tmp'
            os.makedirs(tmp_path, exist_ok=True)
            zip_path = os.path.join(tmp_path, 'training2017.zip')
            if not os.path.exists(zip_path):
                with requests.get(self.URL, stream=True) as r:
                    chunk_size = 1024
                    total = int(r.headers.get('content-length', 0)) // chunk_size
                    with open(zip_path, 'wb') as f:
                        for chunk in tqdm.tqdm(r.iter_content(chunk_size), desc='Downloading Data', total=total):
                            f.write(chunk)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
            shutil.move(os.path.join(tmp_path, 'training2017'), self.path)
            shutil.rmtree(tmp_path)

    def setup(self, stage=None):
        ref_path = os.path.join(self.path, 'REFERENCE.csv')
        ref = pd.read_csv(ref_path, names=['id','label'])
        xs = []
        for mat_id in tqdm.tqdm(ref['id'], desc='Loading Data'):
            path = os.path.join(self.path, f'{mat_id}.mat')
            x = spio.loadmat(path)['val'][0].astype(np.float32)
            x = np.expand_dims(x, -1)
            x = (x - x.mean()) / x.std()
            xs.append(x)
        xs = np.array(xs, dtype=np.object)
        y = pd.Categorical(ref['label'], self.CATS).codes
        counts = np.unique(y, return_counts=True)[1]
        weights = (1/counts)[y]
        rng = np.random.default_rng(self.split_seed)
        train_indices = []
        val_indices = []
        for c in range(len(self.CATS)):
            indices = np.nonzero(y==c)[0]
            rng.shuffle(indices)
            train_size = int(self.train_pct * len(indices))
            train_indices.append(indices[:train_size])
            val_indices.append(indices[train_size:])
        train_indices = np.concatenate(train_indices)
        train_indices.sort()
        val_indices = np.concatenate(val_indices)
        val_indices.sort()
        self.train_ds = Cinc2017Dataset(
            xs[train_indices], y[train_indices], weights[train_indices],
            trim_prob = self.trim_prob, trim_min = self.trim_min)
        self.val_ds = Cinc2017Dataset(
            xs[val_indices], y[val_indices], weights[val_indices],
            trim_prob = 0.0)

    @staticmethod
    def _collate(batch):
        xs, Ns, ys = zip(*((torch.tensor(x), x.shape[0], y) for x, y in batch))
        x = tnur.pad_sequence(xs, batch_first=True).transpose(1,2)
        N = torch.tensor(Ns, dtype=torch.int)
        y = torch.tensor(ys, dtype=torch.long)
        N, sorted_indices = torch.sort(N)
        return x[sorted_indices], N, y[sorted_indices]

    def train_dataloader(self):
        tds = self.train_ds
        if self.balanced_sampling:
            return tud.DataLoader(
                tds, self.batch_size,
                sampler = tud.WeightedRandomSampler(tds.weights, len(tds)),
                collate_fn = self._collate)
        else:
            return tud.DataLoader(
                tds, self.batch_size,
                shuffle = True,
                collate_fn = self._collate)

    def val_dataloader(self):
        return tud.DataLoader(
            self.val_ds, self.batch_size,
            shuffle = False,
            collate_fn = self._collate)
