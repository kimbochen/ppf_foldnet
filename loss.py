import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        dist_matrix = torch.cdist(pred, gt)

        chamfer_dist1 = torch.mean(dist_matrix.min(dim=1).values)
        chamfer_dist2 = torch.mean(dist_matrix.min(dim=2).values)

        loss = chamfer_dist1 + chamfer_dist2

        return loss
