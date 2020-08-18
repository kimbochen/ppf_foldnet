import torch
import torch.nn as nn

from chamfer_distance import ChamferDistance


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
        dist_matrix = torch.cdist(pred, gt)

        chamfer_dist1 = torch.mean(dist_matrix.min(dim=1).values)
        chamfer_dist2 = torch.mean(dist_matrix.min(dim=2).values)

        loss = chamfer_dist1 + chamfer_dist2

        return loss
