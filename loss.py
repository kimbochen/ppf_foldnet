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


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    a = torch.stack([a, a + 1])
    print(f'a:{a.size()}\n{a}')

    b = torch.tensor([[2, 4, 6], [8, 10, 12], [14, 16, 18]], dtype=torch.float)
    b = torch.stack([b, b + 3])
    print(f'b:{b.size()}\n{b}')

    dist_matrix = torch.cdist(a, b)
    print(f'dist_matrix:{dist_matrix.size()}\n{dist_matrix}')
    print(f'dist_matrix.min(0):{dist_matrix.min(0).values.size()}\n{dist_matrix.min(0).values}')
    print(f'dist_matrix.min(1):{dist_matrix.min(1).values.size()}\n{dist_matrix.min(1).values}')
    print(f'dist_matrix.min(2):{dist_matrix.min(2).values.size()}\n{dist_matrix.min(2).values}')

    loss = ChamferLoss()(a, b)
    print(f'\nloss: {loss}')
