from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chamfer_distance import ChamferLoss
from dataset import PPFDataset
from model import Encoder
from configs import *


# CUDA settings
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = DEBUG
torch.backends.cudnn.benchmark = not DEBUG

print(f'Using device {DEVICE}')

# Model
model = Encoder(NUM_PTS_PER_PATCH).to(DEVICE)
model.load_state_dict(torch.load(TRAINER_ARGS.checkpoint_path))
model.eval()

loss_func = ChamferLoss()

# Data loader
test_ds = PPFDataset(TEST_DS_ARGS)
test_dl = DataLoader(test_ds, **TEST_DL_ARGS)
print(f'Testing set: {test_ds.__len__()}')

loss_hist = []

for x_batch, y_batch in tqdm(test_dl):
    with torch.no_grad():
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        pred = model(x_batch)
        loss = loss_func(pred, y_batch)
        loss_hist.append(loss.item())

print(f'Testing set loss: {mean(loss_hist):.4f}')
