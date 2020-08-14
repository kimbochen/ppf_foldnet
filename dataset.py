from pathlib import Path
from time import time

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
    def __init__(self, ppf_dir, num_images):
        assert Path(f'{ppf_dir}/image').is_dir()
        img_dir = list(sorted(Path(f'{ppf_dir}/image').glob('*.npy')))
        assert num_images <= len(img_dir)
        self.image_dir = img_dir[:num_images]

        assert Path(f'{ppf_dir}/model').is_dir()
        md_dir = list(sorted(Path(f'{ppf_dir}/model').glob('*.npy')))
        assert num_images <= len(md_dir)
        self.model_dir = md_dir[:num_images]

        self.dataset_size = num_images

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


class PointCloudDataset(Dataset):
    def __init__(
        self, image_dir, num_images, model_file,
        voxel_size, device, num_patches, num_pts_per_patch
    ):
        assert Path(model_file).is_file()
        model_pcd = read_point_cloud(model_file).voxel_down_sample(voxel_size)
        self.md_pcd_pts = np.asarray(model_pcd.points)

        self.vox_sz = voxel_size
        self.num_patches = num_patches
        self.K = num_pts_per_patch + 1
        self.dev = device

        assert Path(image_dir).is_dir()
        self.img_dir = image_dir
        metadata = load_json(f'{image_dir}/metadata.json')

        assert num_images <= len(metadata)
        self.dataset_size = num_images
        self.data = [
            self.encode_ppf(*sample) for sample in tqdm(metadata[:num_images])
        ]

    def encode_ppf(self, pcd_path, cam_R, cam_t):
        assert Path(f'{self.img_dir}/{pcd_path}.ply').is_file()
        img_pcd_file = f'{self.img_dir}/{pcd_path}.ply'
        img_pcd = read_point_cloud(img_pcd_file).voxel_down_sample(self.vox_sz)

        # Select reference points on image using farthest point sampling
        img_pcd_pts_fps = torch.as_tensor(img_pcd.points).to(self.dev)
        ratio = self.num_patches / img_pcd_pts_fps.size(0)
        img_ref_idxs = fps(img_pcd_pts_fps, ratio=ratio).to('cpu').numpy()

        # Calculate model reference points
        img_ref_pts = np.asarray(img_pcd.points)[img_ref_idxs]
        cam_R, cam_t = np.asarray(cam_R), np.asarray(cam_t)
        md_ref_pts = (img_ref_pts - cam_t.T) @ np.linalg.inv(cam_R)

        # Recreate model point cloud
        md_ref_idxs = np.arange(md_ref_pts.shape[0])
        md_pcd_pts = np.concatenate([md_ref_pts, self.md_pcd_pts], axis=0)
        md_pcd = PointCloud()
        md_pcd.points = Vector3dVector(md_pcd_pts)

        # Calculate PPFs
        img_ppf = create_local_patches(img_pcd, img_ref_idxs, self.K)
        md_ppf = create_local_patches(md_pcd, md_ref_idxs, self.K)

        return img_ppf, md_ppf

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.dataset_size


def create_local_patches(pcd, ref_idxs, K):
    '''
    Create a local patch representation of the input point cloud.
    Arguments:
    pcd: Target point cloud.
    '''
    # Estimate normals
    pcd.estimate_normals()
    pcd_ns = np.asarray(pcd.normals)

    # Select reference points
    ref_pcd = pcd.select_down_sample(ref_idxs)

    # Create a local patch of PPF around each reference point
    pcd_kdtree = KDTreeFlann(pcd)
    pcd_pts = np.asarray(pcd.points)
    patches = []

    for ref_pt, ref_n in zip(ref_pcd.points, ref_pcd.normals):
        idxs = pcd_kdtree.search_knn_vector_3d(ref_pt, K)[1][1:]
        patch = encode_local_patch(ref_pt, ref_n, pcd_pts[idxs], pcd_ns[idxs])
        patches.append(patch)

    return np.stack(patches, axis=0)


def encode_local_patch(ref_pt, ref_n, nbr_pts, nbr_ns):
    '''
    Returns local patch using the dark arts of linear algebra.
    Dimensions:
    ref_pt, ref_n: [3, ]
    nbr_pts, nbr_ns: [num_pts_per_patch, 3]
    '''
    d = ref_pt - nbr_pts  # [num_pts_per_patch, 3]

    def compute_angle(v1, v2, dot_prod):
        cross_norm = np.linalg.norm(np.cross(v1, v2), axis=1)
        return np.arctan2(cross_norm, dot_prod)

    n1_d = compute_angle(d, ref_n, d @ ref_n)
    n2_d = compute_angle(d, nbr_ns, np.einsum('ij, ij->i', d, nbr_ns))
    n1_n2 = compute_angle(nbr_ns, ref_n, nbr_ns @ ref_n)

    d = np.linalg.norm(d, axis=1)
    d_norm = (d - d.min()) * np.pi / (d.max() - d.min())

    return np.float32([n1_d, n2_d, n1_n2, d_norm]).T


def reshape_batch(batch):
    x_batch = np.concatenate([x for x, _ in batch], axis=0)
    y_batch = np.concatenate([y for _, y in batch], axis=0)
    return torch.as_tensor(x_batch), torch.as_tensor(y_batch)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # tau = 0.01
    # diameter = 172.063

    # ds = PointCloudDataset(
    #     image_dir='pcd_data/train/obj_000001',
    #     num_images=40,
    #     model_file='/home/jim/Core7/ycbv/models/obj_000001.ply',
    #     voxel_size=tau * diameter,
    #     num_patches=128,
    #     num_pts_per_patch=2048,
    #     device='cuda:0'
    # )
    # dl = DataLoader(ds, batch_size=8, num_workers=8, collate_fn=reshape_batch)

    ds = PPFDataset('ppf_data/train', 2000)
    dl = DataLoader(ds, batch_size=8, num_workers=8, collate_fn=reshape_batch)
    x, y = next(iter(dl))
    print(x.size(), y.size(), sep='\n')
