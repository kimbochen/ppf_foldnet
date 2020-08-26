from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PPFDataset(Dataset):
    def __init__(self, dataset_dir, num_images=None):
        assert Path(f'data/{dataset_dir}/labels.csv').is_file()
        df = pd.read_csv(f'data/{dataset_dir}/labels.csv')

        if num_images == None:
            num_images = len(df)
        else:
            assert num_images < len(df)
        num_patches = pd.Series(df['num_patches'][:num_images], dtype='int32')
        filename = pd.Series(df['filename'][:num_images])

        self.filenames = [
            f'{fname}_{i:03d}.npy'
            for fname, imax in zip(filename, num_patches)
            for i in range(imax)
        ]

        assert Path(f'data/{dataset_dir}/image').is_dir()
        self.ds_dir = f'data/{dataset_dir}'

        self.dataset_size = len(self.filenames)

    def __getitem__(self, idx):
        assert Path(f'{self.ds_dir}/image/{self.filenames[idx]}').is_file()

        with open(f'{self.ds_dir}/image/{self.filenames[idx]}', 'rb') as file:
            img_ppf = np.load(file)

        return img_ppf

    def __len__(self):
        return self.dataset_size


class PPFPairDataset(Dataset):
    def __init__(self, dataset_dir, num_images=None):
        assert Path(f'data/{dataset_dir}/labels.csv').is_file()
        df = pd.read_csv(f'data/{dataset_dir}/labels.csv')

        if num_images == None:
            num_images = len(df)
        else:
            assert num_images < len(df)
        num_patches = pd.Series(df['num_patches'][:num_images], dtype='int32')
        filename = pd.Series(df['filename'][:num_images])

        self.filenames = [
            f'{fname}_{i:03d}.npy'
            for fname, imax in zip(filename, num_patches)
            for i in range(imax)
        ]

        assert Path(f'data/{dataset_dir}/image').is_dir()
        assert Path(f'data/{dataset_dir}/model').is_dir()
        self.ds_dir = f'data/{dataset_dir}'

        self.dataset_size = len(self.filenames)

    def __getitem__(self, idx):
        assert Path(f'{self.ds_dir}/image/{self.filenames[idx]}').is_file()
        with open(f'{self.ds_dir}/image/{self.filenames[idx]}', 'rb') as file:
            img_ppf = np.load(file)

        assert Path(f'{self.ds_dir}/model/{self.filenames[idx]}').is_file()
        with open(f'{self.ds_dir}/model/{self.filenames[idx]}', 'rb') as file:
            md_ppf = np.load(file)

        return img_ppf, md_ppf

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from statistics import mean, median, stdev
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from configs import DEVICE

    train_ds = PPFPairDataset('ycbv_obj_000001_train', 50)
    train_dl = DataLoader(train_ds, batch_size=1)
    print(f'Training set size: {train_ds.__len__()}')

    x, y = next(iter(train_dl))
    print(x[0, :5, :], y[0, :5, :], sep='\n')
    print(x.size(), y.size())

    loss_func = torch.nn.CosineSimilarity(dim=0)
    loss = loss_func(x, y) 
    print(loss.size(), loss, sep='\n')

    # losses = [
    #     torch.sum(loss_func(x.to(DEVICE), y.to(DEVICE))).item()
    #     for x, y in tqdm(train_dl)
    # ]
    # print(f'Mean: {mean(losses):.3f} / Median: {median(losses):.3f} / Stdev: {stdev(losses):.3f}')
