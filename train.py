from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PPFDataset
from loss import ChamferLoss
from model import AutoEncoder
from configs import *


def train(model, loss_func, optimizer, scheduler, data_loaders, args):
    train_dl, val_dl = data_loaders
    min_loss = 309.805

    for epoch in range(args.num_epochs):
        print('Training ...')
        model.train()
        train_loss = [
            batch_forward(model, loss_func, x_batch, optimizer, scheduler)
            for x_batch in tqdm(train_dl)
        ]

        print('Evaluating ...')
        model.eval()
        with torch.no_grad():
            val_loss = [
                batch_forward(model, loss_func, x_batch)
                for x_batch in tqdm(val_dl)
            ]

        if (epoch + 1) > args.checkpoint_epoch and mean(val_loss) < min_loss:
            print('Saving checkpoint ...')
            torch.save(model.encoder.state_dict(), args.checkpoint_path)
            min_loss = mean(val_loss)

        log = 'Epoch {:03d} | Train loss: {:.4f} | Validation loss: {:.4f}\n'.format(
            epoch, mean(train_loss), mean(val_loss)
        )
        print(log)


def batch_forward(model, loss_func, x_batch, optimizer=None, scheduler=None):
    x_batch = x_batch.to(DEVICE)

    pred = model(x_batch)
    loss = loss_func(pred, x_batch)

    if model.training:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    return loss.item()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    # CUDA settings
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = DEBUG
    torch.backends.cudnn.benchmark = not DEBUG

    print(f'Using device {DEVICE}')

    # Data loaders
    train_ds = PPFDataset(**TRAIN_DS_ARGS)
    train_dl = DataLoader(train_ds, **TRAIN_DL_ARGS)

    val_ds = PPFDataset(**VAL_DS_ARGS)
    val_dl = DataLoader(val_ds, **VAL_DL_ARGS)

    print('Training set: {} Validation set: {}\n'.format(
        train_ds.__len__(), val_ds.__len__()
    ))

    # Model
    model = AutoEncoder(NUM_PTS_PER_PATCH)
    model.apply(init_weights).to(DEVICE)

    loss_func = ChamferLoss()
    optimizer = Adam(model.parameters(), LR)
    scheduler = OneCycleLR(
        optimizer, MAX_LR, total_steps=len(train_dl)*TRAINER_ARGS.num_epochs
    )

    Path(TRAINER_ARGS.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Training
    train(model, loss_func, optimizer, scheduler, (train_dl, val_dl), TRAINER_ARGS)
