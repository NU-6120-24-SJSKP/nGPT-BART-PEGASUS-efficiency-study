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
    """
    Construct and save various metrics from the training and testing process to a pickle file.

    This function aggregates all the metrics collected during training and testing into a dictionary
    and then saves it to a file for later analysis or visualization.
    """
    metrics = {
        # Training and validation losses
        "train_loss_per_epoch": train_loss_per_epoch,  # List of training losses for each epoch
        "val_loss_per_epoch": val_loss_per_epoch,  # List of validation losses for each epoch
        # Perplexity metrics
        "perplexities": perplexities,  # List of validation perplexities
        "generation_perplexities": generation_perplexities,  # List of generation perplexities
        # Inference and memory metrics
        "tokens_per_second": tokens_per_second,  # List of tokens generated per second during inference
        "memory_usages": memory_usages,  # List of peak memory usages during inference
        "inference_times": inference_times,  # List of inference times
        # ROUGE scores for text generation evaluation
        "rouge_scores": rouge_scores,  # List of ROUGE scores for generated summaries
    }

    # Save the metrics dictionary to a pickle file
    with open("ngpt/results/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
