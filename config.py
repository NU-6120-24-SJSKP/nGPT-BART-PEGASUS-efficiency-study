import torch
from torch import GradScaler

# configs for nGPT
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3
PRIME_LENGTH = 128
GENERATE_LENGTH = 512
SEQ_LEN = 512
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
USE_AMP = True  # set True if GPU training
scaler = GradScaler(enabled=USE_AMP)
USE_PARAMETRIZE = True
