import os

import torch

ExpDir = 'path_to_dataset/'
ResDir = 'path_to_result_saved/'

multi_GPUs = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

minibatch_size = 2

DATASET_PATH = ExpDir
EPOCHS = 500
early_stop = 400

EPS_START = 0.95
EPS_END = 0.006

decay = 0.95
lr = 1e-4
n_labels = 2


WEIGHT_DECAY = 0.99
LR_DECAY = 0.8
CUDNN = True
seed = 0
epsilon = 20000

img_dir = ExpDir + 'train/'

save_path = os.path.join('./experiments', ResDir)
if not os.path.exists(save_path): os.mkdir(save_path)
