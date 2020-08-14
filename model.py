import torch
import numpy as np
import torch.nn as nn

from chamfer_distance import ChamferDistance


class AutoEncoder(nn.Module):
    def __init__(self, num_pts_per_patch, **kwargs):
        super().__init__()
        self.encoder = Encoder(num_pts_per_patch)
        self.decoder = Decoder(num_pts_per_patch, **kwargs)

    def forward(self, x):
        codeword = self.encoder(x)
        output = self.decoder(codeword)
        return output


class Encoder(nn.Module):
    def __init__(self, num_pts_per_patch):
        super().__init__()
        
        self.d = num_pts_per_patch

        layer_dims1 = [4, 64, 128, 256]
        self.mlp1 = nn.ModuleList(mlp_block(layer_dims1, num_pts_per_patch))

        self.max_pool1 = nn.MaxPool1d(kernel_size=num_pts_per_patch)

        layer_dims2 = [704, 512, 512]
        self.mlp2 = nn.Sequential(*mlp_block(layer_dims2, num_pts_per_patch))

        self.max_pool2 = nn.MaxPool1d(kernel_size=num_pts_per_patch)

    def forward(self, x):
        features = []
        for fc_layer in self.mlp1:
            x = fc_layer(x)
            features.append(x)

        x = self.max_pool1(x.transpose(1, 2))
        x = x.transpose(1, 2).repeat(1, self.d, 1)

        x = torch.cat([x, *features], dim=-1)
        x = self.mlp2(x)

        codeword = self.max_pool2(x.transpose(1, 2))
        codeword.transpose_(1, 2)

        return codeword


class Decoder(nn.Module):
    def __init__(self, num_pts_per_patch, device):
        super().__init__()

        m_root = int(np.sqrt(num_pts_per_patch))
        self.m = m_root ** 2

        d = torch.linspace(0, np.pi, steps=m_root, device=device)
        self.grid = torch.stack(torch.meshgrid([d, d]), 2).reshape(-1, 2)

        layer_dims1 = [514, 256, 128, 64, 32, 4]
        self.mlp1 = nn.Sequential(*mlp_block(layer_dims1, self.m))

        layer_dims2 = [516, 256, 128, 64, 32, 4]
        self.mlp2 = nn.Sequential(*mlp_block(layer_dims2, self.m))

    def forward(self, x):
        replicated_x = x.repeat(1, self.m, 1)

        batch_grid = torch.stack([self.grid] * x.size(0))
        x = torch.cat([replicated_x, batch_grid], dim=-1)
        fold1 = self.mlp1(x)

        x = torch.cat([replicated_x, fold1], dim=-1)
        fold2 = self.mlp2(x)

        return fold2



def mlp_block(layer_dims, C):
    mlp = []

    for in_f, out_f in zip(layer_dims, layer_dims[1:]):
        layer = nn.Sequential(
            nn.Linear(in_f, out_f),
            nn.BatchNorm1d(C),
            nn.ReLU()
        )
        mlp.append(layer)

    return mlp


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.chamfer_distance = ChamferDistance()

    def forward(self, pred, gt):
        dist1, dist2 = self.chamfer_distance(pred, gt)
        loss = torch.mean(dist1) + torch.mean(dist2)
        return loss

class MyChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        dist_matrix = torch.cdist(pred, gt, p=2)

        chamfer1 = torch.mean(dist_matrix.min(dim=1)[0])
        chamfer2 = torch.mean(dist_matrix.min(dim=2)[0])

        loss = chamfer1 + chamfer2

        return loss


if __name__ == '__main__':
    from dataset import PointCloudDataset, PPFDataset, reshape_batch
    from torch.utils.data import DataLoader

    tau = 0.01
    diameter = 172.063

    ds = PointCloudDataset(
        image_dir='pcd_data/train/obj_000001',
        num_images=10,
        model_file='/home/jim/Core7/ycbv/models/obj_000001.ply',
        voxel_size=tau * diameter,
        num_patches=32,
        num_pts_per_patch=2048,
        device='cuda:1'
    )
    dl = DataLoader(ds, batch_size=1, num_workers=12, collate_fn=reshape_batch)

    # ds = PPFDataset('ppf_data/train', 2000)
    # dl = DataLoader(ds, batch_size=1, num_workers=8, collate_fn=reshape_batch)

    ae = AutoEncoder(2048, device='cuda:1').to('cuda:1')

    loss_func = ChamferLoss()

    x, y = next(iter(dl))
    x, y = x.to('cuda:1'), y.to('cuda:1')
    print(f'input: {x.size()}')

    x_hat = ae(x)
    print(f'output: {x_hat.size()}')

    loss = loss_func(x_hat, x)
    print(f'loss: {loss}')
