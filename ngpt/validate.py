import os

import torch
from torch.utils.data import DataLoader

from ngpt.config import device, SEQ_LEN, BATCH_SIZE
from ngpt.data import dataset, TextSamplerDataset, collate_fn
from ngpt.helpers import cycle

VALIDATE_EVERY_EPOCH = 1
val_loss_per_epoch = []
perplexities = []
training_tokens = []
best_val_loss = float("inf")
CHECKPOINT_DIR = "ngpt/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
patience_counter = 0
patience = 3
val_dataset = dataset["validation"]
val_dataset = TextSamplerDataset(val_dataset, SEQ_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_loader = cycle(val_loader)


def validate_model(epoch):
    global best_val_loss, patience_counter
    if (epoch + 1) == 1 or (epoch + 1) % VALIDATE_EVERY_EPOCH == 0:
        from ngpt.model import model

        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            valid_data = valid_data.to(device)

        val_loss = model(valid_data, return_loss=True)
        val_loss_per_epoch.append(val_loss.item())
        perplexity = torch.exp(val_loss).item()
        perplexities.append(perplexity)
        print(f"Validation loss: {val_loss:.3f}, Perplexity: {perplexity:.3f}")

        # Early stopping logic
        if val_loss < best_val_loss - 0.01:
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
