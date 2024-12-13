import logging
import os

import torch
from torch import GradScaler

# Import necessary libraries for logging, file operations, and PyTorch functionalities.

# Configurations for the nGPT model
BATCH_SIZE = 4  # Number of samples processed before the model is updated
GRAD_ACCUM_EVERY = 2  # Gradient accumulation steps before performing an update
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
PRIME_LENGTH = 128  # Length of the initial sequence used for priming the model
GENERATE_LENGTH = 512  # Length of the generated sequence
SEQ_LEN = 512  # Maximum sequence length for training
NUM_EPOCHS = 200  # Total number of training epochs
GENERATE_EVERY_EPOCH = 1  # Frequency of generating samples during training
VALIDATE_EVERY_EPOCH = 1  # Frequency of validation during training
CHECKPOINT_DIR = "ngpt/checkpoints"  # Directory to save model checkpoints
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR, "best_model.pt"
)  # Path for the best model checkpoint
EPOCH_MODEL = os.path.join(
    CHECKPOINT_DIR, "epoch_model.pt"
)  # Path for the epoch model checkpoint
TRAIN_MODEL = os.path.join(
    CHECKPOINT_DIR, "train_model.pt"
)  # Path for the training model checkpoint

# Set up basic logging configuration to log errors
logging.basicConfig(level=logging.ERROR)

# Ensure the checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set patience for early stopping
patience = 3

# Determine the device for computation (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable Automatic Mixed Precision (AMP) if using GPU
if device == torch.device("cuda:0"):
    USE_AMP = True
else:
    USE_AMP = False

# Initialize the GradScaler for AMP
scaler = GradScaler(enabled=USE_AMP)

# Lists to store training and validation metrics
train_loss_per_epoch = []  # To store training loss for each epoch
val_loss_per_epoch = []  # To store validation loss for each epoch
perplexities = []  # To store perplexity scores
