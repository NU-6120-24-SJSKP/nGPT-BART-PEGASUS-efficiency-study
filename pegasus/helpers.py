"""
Helper functions module containing utility functions used across the project.
Includes setup, cleanup, model inspection, and other utility functions.
"""

import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from transformers import PreTrainedModel
from pegasus.config import TrainingConfig, PathConfig

def set_seed(seed: int = TrainingConfig.SEED) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup() -> None:
    """
    Cleanup function to free GPU memory and close plots.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close('all')

def get_device() -> torch.device:
    """
    Get the appropriate device (CPU/GPU) for training.
    
    Returns:
        torch.device: Device to be used for training
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def verify_model_size(model: PreTrainedModel) -> Tuple[int, int]:
    """
    Calculate the total and trainable parameters in the model.
    
    Args:
        model: The transformer model to analyze
        
    Returns:
        Tuple containing total parameters and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def inspect_frozen_params(model: PreTrainedModel) -> Tuple[List[str], List[str]]:
    """
    Inspect which parameters are frozen and trainable in the model.
    
    Args:
        model: The transformer model to inspect
        
    Returns:
        Tuple containing lists of frozen and trainable parameter names
    """
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(name)
        else:
            trainable_params.append(name)

    print("\nFrozen parameters:")
    for name in frozen_params:
        print(f"- {name}")

    print(f"\nNumber of frozen parameters: {len(frozen_params)}")
    print(f"Number of trainable parameters: {len(trainable_params)}")
    
    return frozen_params, trainable_params

def save_checkpoint(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_loss: float = None,
    val_loss: float = None
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The scheduler state
        epoch: Current epoch number
        train_loss: Current training loss (optional)
        val_loss: Current validation loss (optional)
    """
    try:
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        
        if train_loss is not None:
            checkpoint_dict['train_loss'] = train_loss
        if val_loss is not None:
            checkpoint_dict['val_loss'] = val_loss
            
        checkpoint_path = f"{PathConfig.CHECKPOINT_PREFIX}{epoch}.pth"
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Checkpoint saved successfully at {checkpoint_path}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(
    checkpoint_path: str,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None
) -> Dict:
    """
    Load a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into (optional)
        scheduler: The scheduler to load state into (optional)
        
    Returns:
        Dictionary containing checkpoint information
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        return checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def print_gpu_utilization() -> None:
    """
    Print current GPU memory utilization.
    """
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def reset_gpu_memory() -> None:
    """
    Reset GPU memory stats and clear cache.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        print("GPU memory stats reset and cache cleared")
