import json
from multiprocessing import Pool
from pathlib import Path

import torch
import numpy as np
from numpy.ma import masked_where
from open3d.open3d.camera import PinholeCameraIntrinsic as ph_cam_intrinsic
from open3d.open3d.geometry import Image, PointCloud, KDTreeFlann
from open3d.open3d.io import write_point_cloud, read_point_cloud
from open3d.open3d.utility import set_verbosity_level, VerbosityLevel, Vector3dVector
from torch_cluster import fps
from tqdm import tqdm

import bop_toolkit_lib.inout as io


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            ret = obj.tolist()
        else:
            ret = json.JSONEncoder.default(self, obj)
        return ret


def create_point_cloud_data(obj_id, scene_ids, data_dir, save_dir, im_size):
    '''Create point cloud data of target object of a given dataset.
    Arguments:
    obj_id: Object ID.
    scene_ids: The IDs of the scenes to load data from.
    data_dir: The path to the data directory.
    save_dir: The path to the directory where point cloud data is saved.
    im_size: Size of the images.
    '''
    obj_dir = Path(f'{save_dir}/obj_{obj_id:06d}')
    obj_dir.mkdir(parents=True, exist_ok=True)

    K = [(0, 0), (1, 1), (0, 2), (1, 2)]  # FX, FY, CX, CY

    set_verbosity_level(VerbosityLevel.Error)

    if Path(f'{obj_dir}/metadata.json').is_file():
        metadata = io.load_json(f'{obj_dir}/metadata.json', keys_to_int=True)
    else:
        metadata = []

    print(f'Loading {len(scene_ids)} scenes ...')
    for sc_id in tqdm(scene_ids):
        assert Path(f'{data_dir}/{sc_id:06d}/scene_gt.json').is_file()
        scene_gt = io.load_scene_gt(f'{data_dir}/{sc_id:06d}/scene_gt.json')

        assert Path(f'{data_dir}/{sc_id:06d}/scene_camera.json').is_file()
        scene_cam = io.load_scene_camera(f'{data_dir}/{sc_id:06d}/scene_camera.json')

        with Pool() as pool:
            scene = zip(scene_gt.items(), scene_cam.values())
            jobs = []

            for (im_id, im_gt), im_cam in scene:
                jobs.append(pool.apply_async(
                    load_scene,
                    (sc_id, im_id, im_gt, im_cam, obj_id, data_dir, K, im_size, obj_dir)
                ))

            for j in jobs:
                for row in j.get():
                    metadata.append(row)

    print(f'Writing metadata, {len(metadata)} in total.')
    with open(f'{obj_dir}/metadata.json', 'w+') as file:
        json.dump(metadata, file, cls=NumpyEncoder)


def load_scene(sc_id, im_id, im_gt, im_cam, obj_id, data_dir, K, im_size, obj_dir):
        assert Path(f'{data_dir}/{sc_id:06d}/depth/{im_id:06d}.png').is_file()
        raw_d = io.load_depth(f'{data_dir}/{sc_id:06d}/depth/{im_id:06d}.png')
        cam_K = ph_cam_intrinsic(*im_size, *[im_cam['cam_K'][i] for i in K])
        metadata = []

        for gt_id, gt in enumerate(im_gt):
            if obj_id == int(gt['obj_id']):
                mask_path = '{}/{:06d}/mask_visib/{:06d}_{:06d}.png'.format(
                    data_dir, sc_id, im_id, gt_id
                )
                assert Path(mask_path).is_file()
                mask = io.load_im(mask_path)

                pcd = PointCloud.create_from_depth_image(
                    depth=Image(masked_where(mask == 0.0, raw_d).filled(0.0)),
                    intrinsic=cam_K,
                    depth_scale=im_cam['depth_scale'],
                    depth_trunc=150000
                )

                if np.asarray(pcd.points).shape[0] > 10000:
                    file_name = f'{sc_id:06d}_{im_id:06d}_{gt_id:06d}'
                    write_point_cloud(f'{obj_dir}/{file_name}.ply', pcd)
                    metadata.append([file_name, gt['cam_R_m2c'], gt['cam_t_m2c']])

        return metadata


def create_patch_pairs(
    image_dir, num_images, model_file, voxel_size,
    num_patches, num_pts_per_patch, device
):
    Path('ppf_data/test/image').mkdir(parents=True, exist_ok=True)  # TODO
    Path('ppf_data/test/model').mkdir(parents=True, exist_ok=True)  # TODO

    assert Path(model_file).is_file()
    model_pcd = read_point_cloud(model_file).voxel_down_sample(voxel_size)
    md_pcd_pts = np.asarray(model_pcd.points)

    assert Path(image_dir).is_dir()
    metadata = io.load_json(f'{image_dir}/metadata.json')

    assert num_images <= len(metadata)
    K = num_pts_per_patch + 1

    with Pool() as pool:
        jobs = []
        for pcd_path, cam_R, cam_t in tqdm(metadata[:num_images]):
            jobs.append(pool.apply_async(
                encode_ppf,
                (pcd_path, cam_R, cam_t, image_dir, 
                voxel_size, device, num_patches, md_pcd_pts, K)
            ))

        for j in jobs: j.get()

def encode_ppf(
    pcd_path, cam_R, cam_t, img_dir, voxel_size,
    device, num_patches, md_pcd_pts, K
):
    assert Path(f'{img_dir}/{pcd_path}.ply').is_file()
    img_pcd_file = f'{img_dir}/{pcd_path}.ply'
    img_pcd = read_point_cloud(img_pcd_file).voxel_down_sample(voxel_size)

    # Select reference points on image using farthest point sampling
    img_pcd_pts_fps = torch.as_tensor(img_pcd.points).to(device)
    ratio = num_patches / img_pcd_pts_fps.size(0)
    img_ref_idxs = fps(img_pcd_pts_fps, ratio=ratio).to('cpu').numpy()

    # Calculate model reference points
    img_ref_pts = np.asarray(img_pcd.points)[img_ref_idxs]
    cam_R, cam_t = np.asarray(cam_R), np.asarray(cam_t)
    md_ref_pts = (img_ref_pts - cam_t.T) @ np.linalg.inv(cam_R)

    # Recreate model point cloud
    md_ref_idxs = np.arange(md_ref_pts.shape[0])
    md_pcd_pts = np.concatenate([md_ref_pts, md_pcd_pts], axis=0)
    md_pcd = PointCloud()
    md_pcd.points = Vector3dVector(md_pcd_pts)

    # Calculate and save PPFs
    create_local_patches(img_pcd, img_ref_idxs, K, 'image')
    create_local_patches(md_pcd, md_ref_idxs, K, 'model')

    # img_ppf = create_local_patches(img_pcd, img_ref_idxs, K, 'image')
    # with open(f'ppf_data/test/image/{pcd_path}.npy', 'wb') as file:  # TODO
    #     np.save(file, img_ppf)

    # md_ppf = create_local_patches(md_pcd, md_ref_idxs, K)
    # with open(f'ppf_data/test/model/{pcd_path}.npy', 'wb') as file:  # TODO
    #     np.save(file, md_ppf)

def create_local_patches(pcd, ref_idxs, K, pair):
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

    for i, ref_pt, ref_n in enumerate(zip(ref_pcd.points, ref_pcd.normals)):
        idxs = pcd_kdtree.search_knn_vector_3d(ref_pt, K)[1][1:]
        patch = encode_local_patch(ref_pt, ref_n, pcd_pts[idxs], pcd_ns[idxs])
        with open(f'ppf_data/test/{pair}/{pcd_path}_{i}.npy', 'wb') as file:  # TODO
            np.save(file, patch)

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


if __name__ == '__main__':
    # create_point_cloud_data(
    #     obj_id=1,
    #     scene_ids=range(5),
    #     data_dir='/home/jim/Core7/ycbv/train_pbr',
    #     save_dir='pcd_data/train',
    #     im_size=(640, 480),
    # )

    create_patch_pairs(  # SAVE DIRECTORY IS HARD-CODED, REMEMBER TO CHANGE IT!
        image_dir='pcd_data/test/obj_000001',
        num_images=500,
        model_file='/home/jim/Core7/ycbv/models/obj_000001.ply',
        voxel_size=1.72063,
        num_patches=128,
        num_pts_per_patch=2048,
        device='cuda:1'
    )
