import torch
import numpy as np
import torch.nn as nn
from configs import DEVICE


class AutoEncoder(nn.Module):
    def __init__(self, num_pts_per_patch):
        super().__init__()
        self.encoder = Encoder(num_pts_per_patch)
        self.decoder = Decoder(num_pts_per_patch)

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
    def __init__(self, num_pts_per_patch):
        super().__init__()

        m_root = int(np.sqrt(num_pts_per_patch))
        self.m = m_root ** 2

        d = torch.linspace(0, np.pi, steps=m_root, device=DEVICE)
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


if __name__ == '__main__':
    from dataset import PPFDataset
    from torch.utils.data import DataLoader
    from chamfer_distance import ChamferLoss
    from configs import NUM_PTS_PER_PATCH, DEVICE

    train_ds = PPFDataset('ycbv_obj_000001_train', 2000)
    train_dl = DataLoader(train_ds, batch_size=32)
    print(f'Training set size: {train_ds.__len__()}')

    x, y = next(iter(train_dl))
    x, y = x.to(DEVICE), y.to(DEVICE)
    print(x.size(), y.size())

    model = AutoEncoder(NUM_PTS_PER_PATCH).to(DEVICE)
    loss_func = ChamferLoss()

    y_hat = model(x)
    loss = loss_func(y_hat, y)
    print(loss)
