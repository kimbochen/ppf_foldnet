from multiprocessing import Pool, set_start_method
from pathlib import Path
from time import time

import torch
import numpy as np
from numpy.ma import masked_where
from open3d.open3d.io import read_point_cloud
from open3d.open3d.camera import PinholeCameraIntrinsic as PHCamIntrinsic
from open3d.open3d.geometry import Image, PointCloud, KDTreeFlann
from open3d.open3d.utility import Vector3dVector
from torch_cluster import fps
from tqdm import tqdm

import bop_toolkit_lib.inout as io
from configs import DATASET, OBJ_ID, DEVICE, NUM_PTS_PER_PATCH


SPLIT_NAME = 'test'
SCENE_ID_RANGE = (48, 49)
IM_SIZE = (640, 480)
D = 172.063

VOXEL_SIZE = D * 0.01
FPS_RATIO = 0.01
K = [(0, 0), (1, 1), (0, 2), (1, 2)]  # FX, FY, CX, CY

SAVE_DIR = f'data/{DATASET}_obj_{OBJ_ID:06d}_{SPLIT_NAME.split("_")[0]}'
DATASET_DIR = f'/home/jim/Core7/{DATASET}/{SPLIT_NAME}'


def create_patch_pair(depth_path, mask_path, im_cam, gt, save_name, md_pcd_pts):
    raw_depth = io.load_depth(depth_path)
    mask = io.load_im(mask_path)

    img_pcd = PointCloud.create_from_depth_image(
        depth=Image(masked_where(mask == 0.0, raw_depth).filled(0.0)),
        intrinsic=PHCamIntrinsic(*IM_SIZE, *[im_cam['cam_K'][i] for i in K]),
        depth_scale=im_cam['depth_scale'],
        depth_trunc=150000
    )
    img_pcd.voxel_down_sample(VOXEL_SIZE)

    if np.asarray(img_pcd.points).shape[0] > 10000:
        cam_R, cam_t = gt['cam_R_m2c'], gt['cam_t_m2c']

        # Select reference points on image using farthest point sampling
        img_pcd_pts_fps = torch.as_tensor(img_pcd.points).to(DEVICE)
        img_ref_idxs = fps(img_pcd_pts_fps, ratio=FPS_RATIO).to('cpu').numpy()

        # Calculate model reference points
        img_ref_pts = np.asarray(img_pcd.points)[img_ref_idxs]
        md_ref_pts = (img_ref_pts - cam_t.T) @ np.linalg.inv(cam_R)

        # Recreate model point cloud
        md_ref_idxs = np.arange(md_ref_pts.shape[0])
        md_pcd_pts = np.concatenate([md_ref_pts, md_pcd_pts], axis=0)
        md_pcd = PointCloud()
        md_pcd.points = Vector3dVector(md_pcd_pts)

        # Calculate and save PPFs
        img_save_path = f'image/{save_name}'
        create_local_patches(img_pcd, img_ref_idxs, img_save_path)

        md_save_path = f'model/{save_name}'
        create_local_patches(md_pcd, md_ref_idxs, md_save_path)

        return 1
    else:
        return 0


def create_local_patches(pcd, ref_idxs, save_path):
    # Estimate normals
    pcd.estimate_normals()
    pcd_ns = np.asarray(pcd.normals)

    # Select reference points
    ref_pcd = pcd.select_down_sample(ref_idxs)

    # Create a local patch of PPF around each reference point
    pcd_kdtree = KDTreeFlann(pcd)
    pcd_pts = np.asarray(pcd.points)
    k = NUM_PTS_PER_PATCH + 1

    def compute_angle(v1, v2, dot_prod):
        cross_norm = np.linalg.norm(np.cross(v1, v2), axis=1)
        return np.arctan2(cross_norm, dot_prod)

    for i, (ref_pt, ref_n) in enumerate(zip(ref_pcd.points, ref_pcd.normals)):
        idxs = pcd_kdtree.search_knn_vector_3d(ref_pt, k)[1][1:]
        nbr_pts, nbr_ns = pcd_pts[idxs], pcd_ns[idxs]
        d = ref_pt - nbr_pts  # [num_pts_per_patch, 3]

        n1_d = compute_angle(d, ref_n, d @ ref_n)
        n2_d = compute_angle(d, nbr_ns, np.einsum('ij, ij->i', d, nbr_ns))
        n1_n2 = compute_angle(nbr_ns, ref_n, nbr_ns @ ref_n)

        d = np.linalg.norm(d, axis=1)
        d_norm = (d - d.min()) * np.pi / (d.max() - d.min())

        local_patch = np.float32([n1_d, n2_d, n1_n2, d_norm]).T

        with open(f'{SAVE_DIR}/{save_path}_{i:03d}.npy', 'wb+') as file:
            np.save(file, local_patch)


def preprocess_data():
    # Create directory
    Path(f'{SAVE_DIR}/image').mkdir(parents=True, exist_ok=True)
    Path(f'{SAVE_DIR}/model').mkdir(parents=True, exist_ok=True)

    # Parse metadata and load information of images with the target object
    img_info = []

    for sc_id in range(*SCENE_ID_RANGE):
        assert Path(f'{DATASET_DIR}/{sc_id:06d}/scene_gt.json').is_file()
        scene_gt = io.load_scene_gt(f'{DATASET_DIR}/{sc_id:06d}/scene_gt.json')

        assert Path(f'{DATASET_DIR}/{sc_id:06d}/scene_camera.json').is_file()
        scene_cam = io.load_scene_camera(f'{DATASET_DIR}/{sc_id:06d}/scene_camera.json')

        for (im_id, im_gt), im_cam in zip(scene_gt.items(), scene_cam.values()):
            for gt_id, gt in enumerate(im_gt):
                if int(gt['obj_id']) == OBJ_ID:
                    assert Path(f'{DATASET_DIR}/{sc_id:06d}/depth/{im_id:06d}.png').is_file()
                    depth_path = f'{DATASET_DIR}/{sc_id:06d}/depth/{im_id:06d}.png'

                    assert Path(f'{DATASET_DIR}/{sc_id:06d}/mask_visib/{im_id:06d}_{gt_id:06d}.png').is_file()
                    mask_path = f'{DATASET_DIR}/{sc_id:06d}/mask_visib/{im_id:06d}_{gt_id:06d}.png'

                    save_name = f'{sc_id:06d}_{im_id:06d}_{gt_id:06d}'

                    img_info.append([depth_path, mask_path, im_cam, gt, save_name])

    # Read model point cloud
    model_file = f'{Path(DATASET_DIR).parent}/models/obj_{OBJ_ID:06d}.ply'
    assert Path(model_file).is_file()
    model_pcd = read_point_cloud(model_file).voxel_down_sample(VOXEL_SIZE)
    md_pcd_pts = np.asarray(model_pcd.points)

    # Create point cloud from image information
    t1 = time()
    set_start_method('spawn')
    with Pool() as pool:
        jobs = [
            pool.apply_async(create_patch_pair, (*info, md_pcd_pts))
            for info in img_info
        ]
        num_patch_pairs = sum([j.get() for j in jobs])
        t2 = time()
        print(f'Created {num_patch_pairs} patch_pairs. Time elapsed: {t2 - t1 :.3f}')


if __name__ == '__main__':
    preprocess_data()
