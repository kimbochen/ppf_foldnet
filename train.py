from pathlib import Path
from statistics import mean

import hydra
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PointCloudDataset, reshape_batch
from model import AutoEncoder, ChamferLoss


def train(model, loss_func, optimizer, scheduler, data_loaders, device, args):
    train_dl, val_dl = data_loaders
    loss = 0.0

    for epoch in range(args.num_epochs):
        print('Training ...')
        model.train()
        for x_batch, y_batch in tqdm(train_dl):
            batch_forward(model, loss_func, x_batch, y_batch, device)
            optimizer.step()
            optimizer.zero_grad()

        print('Evaluating ...')
        model.eval()
        with torch.no_grad():
            loss_hist = [
                batch_forward(model, loss_func, x_batch, y_batch, device)
                for x_batch, y_batch in tqdm(val_dl)
            ]

        if (epoch + 1) % args.scheduler_interval == 0:
            scheduler.step()

        if mean(loss_hist) < loss and (epoch + 1) > args.checkpoint_epoch:
            print('Saving checkpoint ...')
            torch.save(model.state_dict(), args.checkpoint_path)

        loss = mean(loss_hist)
        print(f'Epoch {epoch:03d} validation loss: {loss:.4f}\n')


def batch_forward(model, loss_func, x_batch, y_batch, device):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

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


@hydra.main(config_path='config.yaml')
def main(cfg):
    # CUDA settings
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.debug
    torch.backends.cudnn.benchmark = not cfg.debug

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Model
    model = hydra.utils.instantiate(cfg.model)
    model.apply(init_weights).to(device)

    loss_func = ChamferLoss()
    optimizer = Adam(model.parameters(), **cfg.optimizer)
    scheduler = ExponentialLR(optimizer, cfg.scheduler_gamma)

    # Data loaders
    print('Loading training set ...')
    train_ds = hydra.utils.instantiate(cfg.train_ds)
    train_dl = DataLoader(train_ds, collate_fn=reshape_batch, **cfg.dl)

    print('Loading validation set ...')
    val_ds = hydra.utils.instantiate(cfg.val_ds)
    val_dl = DataLoader(val_ds, collate_fn=reshape_batch, **cfg.dl)

    print('\nTraining set: {}, Validation set: {}.'.format(
        train_ds.__len__(), val_ds.__len__()
    ))

    Path(cfg.train.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Training
    train(model, loss_func, optimizer, scheduler, (train_dl, val_dl), device, cfg.train)


if __name__ == '__main__':
    main()
