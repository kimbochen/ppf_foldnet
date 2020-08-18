import torch
from easydict import EasyDict


# Dataset configurations
DATASET = 'ycbv'
OBJ_ID = 1
NUM_PTS_PER_PATCH = 2048

# CUDA configurations
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
DEBUG = True
SEED = 309805


# Data loading configurations
TRAIN_DS_ARGS = {
    'dataset_dir': f'{DATASET}_obj_{OBJ_ID:06d}_train',
    'num_patches': 12800
}
TRAIN_DL_ARGS = {
    'batch_size': 128,
    'num_workers': 8,
    'shuffle': True
}

VAL_DS_ARGS = {
    'dataset_dir': f'{DATASET}_obj_{OBJ_ID:06d}_test',
    'num_patches': 9600
}
VAL_DL_ARGS = {
    'batch_size': 128,
    'num_workers': 8,
    'shuffle': True
}

TEST_DS_ARGS = f'{DATASET}_obj_{OBJ_ID:06d}_target'
TEST_DL_ARGS = {
    'batch_size': 400,
    'num_workers': 8,
}


# Training configurations
TRAINER_ARGS = EasyDict({
    'num_epochs': 100,
    'scheduler_interval': 1,
    'checkpoint_epoch': 10,
    'checkpoint_path': "checkpoints/ckpt_{DATASET}_obj_{obj_id:06d}.pth"
})

OPTIMIZER_ARGS = {'lr': 1e-3}
