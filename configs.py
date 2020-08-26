import torch
from easydict import EasyDict


# Dataset configurations
DATASET = 'ycbv'
OBJ_ID = 1
NUM_PTS_PER_PATCH = 1024

# CUDA configurations
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
DEBUG = True
SEED = 309805


# Data loading configurations
DL_ARGS = {
    'batch_size': 64,
    'num_workers': 8,
}

TRAIN_DS_ARGS = {
    'dataset_dir': f'{DATASET}_obj_{OBJ_ID:06d}_train',
    'num_images': 60
}
TRAIN_DL_ARGS = {**DL_ARGS, 'shuffle': True}

VAL_DS_ARGS = {
    'dataset_dir': f'{DATASET}_obj_{OBJ_ID:06d}_val',
    'num_images': 60
}
VAL_DL_ARGS = {**DL_ARGS}

TEST_DS_ARGS = f'{DATASET}_obj_{OBJ_ID:06d}_test'
TEST_DL_ARGS = {
    'batch_size': 400,
    'num_workers': 8,
}


# Training configurations
TRAINER_ARGS = EasyDict({
    'num_epochs': 100,
    'checkpoint_epoch': 10,
    'checkpoint_path': f'checkpoints/ckpt_{DATASET}_obj_{OBJ_ID:06d}.pth'
})

# Learning rate
LR = 1e-4
MAX_LR = 3e-2
