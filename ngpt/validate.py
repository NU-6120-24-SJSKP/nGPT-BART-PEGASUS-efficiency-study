import os

import torch
from torch.utils.data import DataLoader

from ngpt.config import device, SEQ_LEN, BATCH_SIZE
from ngpt.data import dataset, TextSamplerDataset, collate_fn
from ngpt.helpers import cycle

# Configuration for validation
VALIDATE_EVERY_EPOCH = 1  # Frequency of validation during training

# Lists to store validation metrics
val_loss_per_epoch = []  # To store validation loss for each epoch
perplexities = []  # To store perplexity scores
training_tokens = []  # To track the number of tokens seen during training

# Initialize best validation loss and patience counter for early stopping
best_val_loss = float("inf")
CHECKPOINT_DIR = "ngpt/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure the checkpoint directory exists
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
patience_counter = 0
patience = 3  # Number of epochs to wait before stopping training if no improvement

# Prepare validation dataset and dataloader
val_dataset = dataset["validation"]
val_dataset = TextSamplerDataset(val_dataset, SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_loader = cycle(val_loader)  # Create an infinite cycle of the validation data


def validate_model(epoch):
    """
    Perform validation on the model at specified epochs and implement early stopping.

    :param epoch: Current training epoch
    :return: "stop" if training should be stopped, "continue" otherwise
    """
    global best_val_loss, patience_counter

    # Check if it's time to validate
    if (epoch + 1) == 1 or (epoch + 1) % VALIDATE_EVERY_EPOCH == 0:
        from ngpt.model import (
            model,
        )  # Import model dynamically to avoid circular dependencies

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for efficiency
            valid_data = next(val_loader)
            valid_data = valid_data.to(device)

        # Compute validation loss
        val_loss = model(valid_data, return_loss=True)
        val_loss_per_epoch.append(val_loss.item())
        perplexity = torch.exp(val_loss).item()
        perplexities.append(perplexity)
        print(f"Validation loss: {val_loss:.3f}, Perplexity: {perplexity:.3f}")

        # Early stopping logic
        if val_loss < best_val_loss - 0.01:  # Check for significant improvement
            best_val_loss = val_loss.item()
            patience_counter = 0
            print(f"New best validation loss: {best_val_loss:.3f}. Saving model...")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}"
            )

        # Stop training if patience is exhausted
        if patience_counter >= patience:
            print("Early stopping triggered. Training has converged.")
            return "stop"
    return "continue"
