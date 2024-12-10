"""
Plotting module for visualizing training metrics and model performance.
Contains functions for generating and saving various types of plots.
"""

import matplotlib.pyplot as plt
from typing import Dict, Any
import pickle
from config import PathConfig

class MetricsPlotter:
    """
    Class for creating and saving visualization plots for various metrics.
    """
    
    def __init__(self, save_dir: str = "."):
        """
        Initialize the plotter.
        
        Args:
            save_dir: Directory to save the plots
        """
        self.save_dir = save_dir
        self.metrics = self._load_metrics()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """
        Load metrics from the pickle file.
        
        Returns:
            Dictionary containing training metrics
        """
        try:
            with open(PathConfig.METRICS_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return None

    def _save_plot(self, filename: str) -> None:
        """
        Save the current plot to a file.
        
        Args:
            filename: Name of the file to save the plot
        """
        try:
            plt.savefig(f"{self.save_dir}/{filename}")
            plt.close()
        except Exception as e:
            print(f"Error saving plot: {e}")
            
    def plot_training_progress(self) -> None:
        """
        Create a combined plot showing losses and ROUGE scores.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["train_losses"], label='Train Loss')
        plt.plot(self.metrics["val_losses"], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot ROUGE scores
        plt.subplot(2, 2, 3)
        for metric, scores in self.metrics["rouge_scores"].items():
            plt.plot(scores, label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('ROUGE Scores')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        self._save_plot('training_progress.png')
        
    def plot_loss_curves(self) -> None:
        """
        Plot detailed loss curves including per-step and per-epoch losses.
        """
        # Per-step training loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["train_batch_losses"], 
                label='Training Loss per Step', 
                color='blue', 
                linewidth=0.7)
        plt.xlabel('Batch/Step')
        plt.ylabel('Loss')
        plt.title('Training Loss per Step')
        plt.legend()
        plt.grid(True)
        self._save_plot('train_loss_per_step.png')
        
        # Per-epoch validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.metrics["val_losses"]) + 1),
                self.metrics["val_losses"],
                label='Validation Loss',
                color='red',
                marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        self._save_plot('val_loss_per_epoch.png')
        
    def plot_perplexity(self) -> None:
        """
        Plot validation perplexity over epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.metrics["val_perplexities"]) + 1),
                self.metrics["val_perplexities"],
                label='Validation Perplexity',
                color='green',
                marker='o')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Validation Perplexity per Epoch')
        plt.legend()
        plt.grid(True)
        self._save_plot('val_perplexity_per_epoch.png')
        
    def plot_performance_metrics(self) -> None:
        """
        Plot various performance metrics including inference time and memory usage.
        """
        # Inference time
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.metrics["inference_times"]) + 1),
                self.metrics["inference_times"],
                label='Inference Time (s)',
                color='purple',
                marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Inference Time per Epoch')
        plt.legend()
        plt.grid(True)
        self._save_plot('inference_time_vs_epoch.png')
        
        # Tokens per second
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.metrics["tokens_per_seconds"]) + 1),
                self.metrics["tokens_per_seconds"],
                label='Tokens per Second',
                color='orange',
                marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Tokens per Second')
        plt.title('Processing Speed per Epoch')
        plt.legend()
        plt.grid(True)
        self._save_plot('tokens_per_second_vs_epoch.png')
        
        # Memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.metrics["peak_memory_usages"]) + 1),
                self.metrics["peak_memory_usages"],
                label='Peak Memory Usage (MB)',
                color='green',
                marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (MB)')
        plt.title('Peak Memory Usage per Epoch')
        plt.legend()
        plt.grid(True)
        self._save_plot('peak_memory_usage_vs_epoch.png')
        
    def plot_rouge_scores(self) -> None:
        """
        Plot detailed ROUGE scores over epochs.
        """
        plt.figure(figsize=(12, 8))
        for metric, scores in self.metrics["rouge_scores"].items():
            plt.plot(scores, label=metric, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title('ROUGE Scores Over Time')
        plt.legend()
        plt.grid(True)
        self._save_plot('rouge_scores.png')
        
    def create_all_plots(self) -> None:
        """
        Generate all available plots.
        """
        self.plot_training_progress()
        self.plot_loss_curves()
        self.plot_perplexity()
        self.plot_performance_metrics()
        self.plot_rouge_scores()
