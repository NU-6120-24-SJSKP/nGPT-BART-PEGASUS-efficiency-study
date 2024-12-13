"""
Metrics module for evaluating text summarization models.
Handles ROUGE scores calculation, timing metrics, and performance tracking.
"""

import time
import pickle
import torch
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from pegasus.config import TrainingConfig, PathConfig

class MetricsTracker:
    """
    Class for tracking and storing various training and evaluation metrics.
    """
    def __init__(self):
        self.metrics = {
            "train_losses": [],
            "train_batch_losses": [],
            "val_losses": [],
            "val_perplexities": [],
            "inference_times": [],
            "tokens_per_seconds": [],
            "peak_memory_usages": [],
            "training_tokens": [],
            "rouge_scores": {"rouge1": [], "rouge2": [], "rougeL": []}
        }
    
    def update_train_metrics(self, train_loss: float, batch_loss: float) -> None:
        """
        Update training-related metrics.
        
        Args:
            train_loss: Average training loss for the epoch
            batch_loss: Individual batch loss
        """
        self.metrics["train_losses"].append(train_loss)
        self.metrics["train_batch_losses"].append(batch_loss)
    
    def update_val_metrics(self, val_loss: float) -> None:
        """
        Update validation-related metrics.
        
        Args:
            val_loss: Validation loss
        """
        self.metrics["val_losses"].append(val_loss)
        self.metrics["val_perplexities"].append(torch.exp(torch.tensor(val_loss)).item())
    
    def update_performance_metrics(
        self, 
        inference_time: float, 
        total_tokens: int,
        peak_memory: float
    ) -> None:
        """
        Update performance-related metrics.
        
        Args:
            inference_time: Time taken for inference
            total_tokens: Number of tokens processed
            peak_memory: Peak memory usage
        """
        self.metrics["inference_times"].append(inference_time)
        self.metrics["tokens_per_seconds"].append(
            total_tokens / inference_time if inference_time > 0 else 0
        )
        self.metrics["peak_memory_usages"].append(peak_memory)
    
    def update_rouge_scores(self, rouge_scores: Dict[str, float]) -> None:
        """
        Update ROUGE scores.
        
        Args:
            rouge_scores: Dictionary containing ROUGE scores
        """
        for metric, score in rouge_scores.items():
            self.metrics["rouge_scores"][metric].append(score)
    
    def save_metrics(self) -> None:
        """Save metrics to a pickle file."""
        try:
            with open(PathConfig.METRICS_FILE, 'wb') as f:
                pickle.dump(self.metrics, f)
        except Exception as e:
            print(f"Error saving metrics: {e}")
    
    @staticmethod
    def load_metrics() -> Dict:
        """
        Load metrics from a pickle file.
        
        Returns:
            Dictionary containing saved metrics
        """
        try:
            with open(PathConfig.METRICS_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return None

class Evaluator:
    """
    Class for evaluating model performance and calculating metrics.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[Dict[str, float], float, int, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            data_loader: DataLoader containing validation data
            
        Returns:
            Tuple containing ROUGE scores, total time, total tokens, and peak memory usage
        """
        self.model.eval()
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        total_time = 0.0
        total_tokens = 0
        peak_memory = 0.0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                try:
                    batch_metrics = self._evaluate_batch(batch)
                    
                    # Update metrics
                    total_time += batch_metrics['time']
                    total_tokens += batch_metrics['tokens']
                    peak_memory = max(peak_memory, batch_metrics['memory'])
                    
                    # Update ROUGE scores
                    for metric in rouge_scores:
                        rouge_scores[metric].extend(batch_metrics['rouge_scores'][metric])
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: out of memory in evaluation. Skipping batch...")
                        torch.cuda.empty_cache()
                        continue
                    raise e
        
        # Calculate average ROUGE scores
        avg_rouge = {metric: sum(scores) / len(scores) 
                    for metric, scores in rouge_scores.items()}
        
        return avg_rouge, total_time, total_tokens, peak_memory
    
    def _evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Evaluate a single batch of data.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary containing batch evaluation metrics
        """
        start_time = time.time()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        
        # Generate summaries
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=TrainingConfig.MAX_TARGET_LENGTH
        )
        
        # Calculate batch metrics
        batch_time = time.time() - start_time
        batch_tokens = sum(len(ids) for ids in generated_ids)
        batch_memory = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                       if torch.cuda.is_available() else 0)
        
        # Calculate ROUGE scores
        batch_rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for i in range(len(input_ids)):
            reference = self.tokenizer.decode(labels[i], skip_special_tokens=True)
            generated = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            
            scores = self.scorer.score(reference, generated)
            for metric in batch_rouge_scores:
                batch_rouge_scores[metric].append(scores[metric].fmeasure)
        
        return {
            'time': batch_time,
            'tokens': batch_tokens,
            'memory': batch_memory,
            'rouge_scores': batch_rouge_scores
        }
