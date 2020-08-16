import torch
from easydict import EasyDict as edict

DATASET = 'ycbv'
OBJ_ID = 1

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEBUG = True
SEED = 309805

NUM_PTS_PER_PATCH = 2048

TRAIN_DS_ARGS = {
    'dataset_dir': 'ycbv_obj_000001_train',
    'num_patches': 6000
}

TEST_DS_ARGS = {
    'dataset_dir': 'ycbv_obj_000001_test',
    'num_patches': 2012
}

DATALOADER_ARGS = {
    'batch_size': 32,
    'num_workers': 8,
    'shuffle': True
}

TRAIN_ARGS = edict({
    'num_epochs': 100,
    'scheduler_interval': 5,
    'checkpoint_epoch': 10,
    'checkpoint_path': "checkpoints/ckpt_{DATASET}_obj_{obj_id:06d}.pth"
})


OPTIMIZER_ARGS = {
    'lr': 3e-4,
    'weight_decay': 1e-6
}

SCHEDULER_GAMMA = 0.5
