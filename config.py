import logging
import os

import torch
from torch import GradScaler

# configs for nGPT
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 2
LEARNING_RATE = 1e-3
PRIME_LENGTH = 128
GENERATE_LENGTH = 512
SEQ_LEN = 512
NUM_EPOCHS = 200
GENERATE_EVERY_EPOCH = 1
VALIDATE_EVERY_EPOCH = 1
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
EPOCH_MODEL = os.path.join(CHECKPOINT_DIR, "epoch_model.pt")
TRAIN_MODEL = os.path.join(CHECKPOINT_DIR, "train_model.pt")
logging.basicConfig(level=logging.ERROR)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
patience = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == torch.device("cuda:0"):
    USE_AMP = True
else:
    USE_AMP = False
scaler = GradScaler(enabled=USE_AMP)


train_loss_per_epoch = []
val_loss_per_epoch = []
perplexities = []
