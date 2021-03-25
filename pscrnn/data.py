import numpy as np
import os
import pandas as pd
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
    def __init__(self, xs, y, weights, trim_min=0.5):
        self.xs = xs
        self.y = y
        self.weights = weights
        self.trim_min = trim_min

    def __len__(self):
        return self.xs.shape[0]

    def _augment(self, x):
        length = x.shape[0]
        trimmed = torch.randint(int(length*self.trim_min), length, ()).item()
        offset = torch.randint(length-trimmed, ()).item()
        return x[offset:(offset+trimmed)]

class Cinc2017TrainDataset(Cinc2017Dataset):
    def __init__(self, xs, y, weights, trim_prob, trim_min=0.5):
        super().__init__(xs, y, weights, trim_min)
        self.trim_prob = trim_prob

    def __getitem__(self, idx):
        x = self.xs[idx]
        if self.trim_prob and (torch.rand(()).item() < self.trim_prob):
            x = self._augment(x)
        return x, self.y[idx]

class Cinc2017TestDataset(Cinc2017Dataset):
    def __init__(self, xs, y, weights, augments, trim_min=0.5):
        super().__init__(xs, y, weights, trim_min)
        self.augments = augments

    def __getitem__(self, idx):
        xs = [self.xs[idx]]
        for _ in range(self.augments):
            xs.append(self._augment(xs[0]))
        return xs, self.y[idx]

class Cinc2017:
    CATS = ['N','A','O','~']
    URL = 'https://www.physionet.org/files/challenge-2017/1.0.0/training2017.zip?download'
    PATH = 'data/cinc2017'

    def __init__(self, base_path='.', train_pct=0.7, split_seed=1234):
        self.path = os.path.join(base_path, self.PATH)
        self.train_pct = train_pct
        self.split_seed = split_seed
        self.n_features = 1
        self.n_classes = len(self.CATS)

    def _download(self):
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

    def _setup(self, trim_prob, val_augments, trim_min):
        ref_path = os.path.join(self.path, 'REFERENCE.csv')
        ref = pd.read_csv(ref_path, names=['id','label'])
        xs = []
        for mat_id in tqdm.tqdm(ref['id'], desc='Loading Data'):
            path = os.path.join(self.path, f'{mat_id}.mat')
            x = spio.loadmat(path)['val'][0].astype(np.float32)
            x = np.expand_dims(x, -1)
            x = (x - x.mean()) / x.std()
            xs.append(torch.tensor(x))
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
        train_ds = Cinc2017TrainDataset(
            xs[train_indices], y[train_indices], weights[train_indices],
            trim_prob = trim_prob, trim_min = trim_min)
        val_ds = Cinc2017TestDataset(
            xs[val_indices], y[val_indices], weights[val_indices],
            augments = val_augments, trim_min = trim_min)
        return train_ds, val_ds

    @staticmethod
    def _collate(batch):
        xs, ys = zip(*batch)
        if isinstance(xs[0], list):
            xs = sum(xs, []) ## flatten
        y = torch.tensor(ys, dtype=torch.long)
        return xs, y

    def _train_dataloader(self, train_ds, batch_size, balanced_sampling):
        if balanced_sampling:
            return tud.DataLoader(
                train_ds, batch_size,
                sampler = tud.WeightedRandomSampler(train_ds.weights, len(train_ds)),
                collate_fn = self._collate)
        else:
            return tud.DataLoader(
                train_ds, batch_size,
                shuffle = True,
                collate_fn = self._collate)

    def _val_dataloader(self, val_ds, batch_size):
        return tud.DataLoader(
            val_ds, batch_size,
            shuffle = False,
            collate_fn = self._collate)

    def __call__(self, batch_size, trim_prob, val_augments, trim_min, balanced_sampling):
        self._download()
        train_ds, val_ds = self._setup(trim_prob, val_augments, trim_min)
        train_dl = self._train_dataloader(train_ds, batch_size, balanced_sampling)
        val_dl = self._val_dataloader(val_ds, batch_size)
        return train_dl, val_dl
