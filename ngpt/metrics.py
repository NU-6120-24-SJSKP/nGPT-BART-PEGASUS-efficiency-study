import pickle

from ngpt.config import train_loss_per_epoch, val_loss_per_epoch, perplexities
from ngpt.test import (
    generation_perplexities,
    tokens_per_second,
    memory_usages,
    inference_times,
    rouge_scores,
)


def construct_metrics():
    metrics = {
        # losses
        "train_loss_per_epoch": train_loss_per_epoch,  # NUM_EPOCHS
        "val_loss_per_epoch": val_loss_per_epoch,  # validation_epochs
        # "validation_epochs": validation_epochs,
        # perplexities
        "perplexities": perplexities,  # validation_epochs
        "generation_perplexities": generation_perplexities,  # generation_epochs
        # inference and memory
        "tokens_per_second": tokens_per_second,  # generation_epochs
        "memory_usages": memory_usages,  # generation_epochs
        "inference_times": inference_times,  # generation_epochs
        "rouge_scores": rouge_scores,
    }
    with open("ngpt/results/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
