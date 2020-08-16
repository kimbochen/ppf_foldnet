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
