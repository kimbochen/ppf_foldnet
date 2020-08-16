from pathlib import Path

import torch
import numpy as np
from open3d.open3d.geometry import KDTreeFlann, PointCloud
from open3d.open3d.io import read_point_cloud
from open3d.open3d.utility import Vector3dVector
from torch_cluster import fps
from torch.utils.data import Dataset
from tqdm import tqdm

from bop_toolkit_lib.inout import load_json


class PPFDataset(Dataset):
    def __init__(self, dataset_dir, num_patches):

        assert Path(f'data/{dataset_dir}/image').is_dir()
        img_dir_glob = Path(f'data/{dataset_dir}/image').glob('*.npy')
        img_dir = list(sorted(img_dir_glob))

        assert Path(f'data/{dataset_dir}/model').is_dir()
        md_dir_glob = Path(f'data/{dataset_dir}/model').glob('*.npy')
        md_dir = list(sorted(md_dir_glob))

        assert num_patches <= len(md_dir) and num_patches <= len(img_dir)
        self.image_dir = img_dir[:num_patches]
        self.model_dir = md_dir[:num_patches]
        self.dataset_size = num_patches

    def __getitem__(self, idx):
        assert Path(self.image_dir[idx]).is_file()
        with open(self.image_dir[idx], 'rb') as file:
            img_ppf = np.load(file)

        assert Path(self.model_dir[idx]).is_file()
        with open(self.model_dir[idx], 'rb') as file:
            md_ppf = np.load(file)

        return img_ppf, md_ppf

    def __len__(self):
        return self.dataset_size


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_ds = PPFDataset('ycbv_obj_000001_train', 2000)
    print(train_ds.__len__())

    train_dl = DataLoader(train_ds, batch_size=8)

    x, y = next(iter(train_dl))
    print(x.size(), y.size())
