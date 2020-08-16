from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from chamfer_distance import ChamferLoss
from dataset import PPFDataset
from model import AutoEncoder
from configs import *


def train(model, loss_func, optimizer, scheduler, data_loaders, args):
    train_dl, val_dl = data_loaders
    loss = 0.0

    for epoch in range(args.num_epochs):
        print('Training ...')
        model.train()
        for x_batch, y_batch in tqdm(train_dl):
            batch_forward(model, loss_func, x_batch, y_batch)
            optimizer.step()
            optimizer.zero_grad()

        print('Evaluating ...')
        model.eval()
        with torch.no_grad():
            loss_hist = [
                batch_forward(model, loss_func, x_batch, y_batch)
                for x_batch, y_batch in tqdm(val_dl)
            ]

        if (epoch + 1) % args.scheduler_interval == 0:
            scheduler.step()

        if mean(loss_hist) < loss and (epoch + 1) > args.checkpoint_epoch:
            print('Saving checkpoint ...')
            torch.save(model.state_dict(), args.checkpoint_path)

        loss = mean(loss_hist)
        print(f'Epoch {epoch:03d} validation loss: {loss:.4f}\n')


def batch_forward(model, loss_func, x_batch, y_batch):
    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

    pred = model(x_batch)
    loss = loss_func(pred, y_batch)

    if model.training:
        loss.backward()
    else:
        return loss.item()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # CUDA settings
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = DEBUG
    torch.backends.cudnn.benchmark = not DEBUG

    print(f'Using device {DEVICE}')

    # Model
    model = AutoEncoder(NUM_PTS_PER_PATCH)
    model.apply(init_weights).to(DEVICE)

    loss_func = ChamferLoss()
    optimizer = Adam(model.parameters(), **OPTIMIZER_ARGS)
    scheduler = ExponentialLR(optimizer, SCHEDULER_GAMMA)

    # Data loaders
    train_ds = PPFDataset(**TRAIN_DS_ARGS)
    train_dl = DataLoader(train_ds, **DATALOADER_ARGS)

    val_ds = PPFDataset(**TEST_DS_ARGS)
    val_dl = DataLoader(val_ds, **DATALOADER_ARGS)

    print('Training set: {}, Validation set: {}.\n'.format(
        train_ds.__len__(), val_ds.__len__()
    ))

    Path(TRAIN_ARGS.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Training
    train(model, loss_func, optimizer, scheduler, (train_dl, val_dl), TRAIN_ARGS)
