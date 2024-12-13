"""
Validation module for text summarization model.
Handles model validation, performance verification, and output quality assessment.
"""

import torch
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np

from model import SummarizationModel
from config import TrainingConfig

@dataclass
class ValidationMetrics:
    """Data class for storing validation metrics."""
    loss: float
    perplexity: float
    rouge_scores: Dict[str, float]
    generation_metrics: Dict[str, float]

class ValidationResult:
    """Class for storing and analyzing validation results."""
    
    def __init__(self, metrics: ValidationMetrics):
        """
        Initialize validation result.
        
        Args:
            metrics: Validation metrics
        """
        self.metrics = metrics
        
    def is_improved(self, previous_best: float) -> bool:
        """
        Check if current validation result is improved.
        
        Args:
            previous_best: Previous best validation loss
            
        Returns:
            Boolean indicating improvement
        """
        return self.metrics.loss < previous_best
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of validation results.
        
        Returns:
            Dictionary containing validation metrics summary
        """
        return {
            'loss': self.metrics.loss,
            'perplexity': self.metrics.perplexity,
            **self.metrics.rouge_scores,
            **self.metrics.generation_metrics
        }

class ModelValidator:
    """
    Class for handling model validation operations.
    """
    
    def __init__(
        self,
        model: SummarizationModel,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        Initialize the validator.
        
        Args:
            model: The summarization model
            val_loader: Validation data loader
            device: Device to run validation on
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        
    def validate_epoch(self) -> ValidationResult:
        """
        Validate model for one epoch.
        
        Returns:
            ValidationResult object containing metrics
        """
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        generation_times = []
        generated_lengths = []
        
        print("\nRunning validation...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    # Process batch
                    batch_metrics = self._validate_batch(batch)
                    
                    # Update metrics
                    total_loss += batch_metrics['loss']
                    batch_count += 1
                    
                    # Update ROUGE scores
                    for metric in rouge_scores:
                        rouge_scores[metric].extend(batch_metrics['rouge_scores'][metric])
                    
                    generation_times.append(batch_metrics['generation_time'])
                    generated_lengths.extend(batch_metrics['generated_lengths'])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("\nWARNING: Out of memory in validation. Skipping batch...")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        continue
                    raise e
        
        # Calculate average metrics
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        avg_rouge = {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
        
        generation_metrics = {
            'avg_generation_time': np.mean(generation_times),
            'avg_summary_length': np.mean(generated_lengths),
            'std_summary_length': np.std(generated_lengths)
        }
        
        # Create validation metrics
        metrics = ValidationMetrics(
            loss=avg_loss,
            perplexity=torch.exp(torch.tensor(avg_loss)).item(),
            rouge_scores=avg_rouge,
            generation_metrics=generation_metrics
        )
        
        return ValidationResult(metrics)
    
    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate a single batch.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary containing batch validation metrics
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Generate summaries
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        generated_ids = self.model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=TrainingConfig.MAX_TARGET_LENGTH
        )
        end_time.record()
        
        # Calculate generation time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            generation_time = 0
        
        # Decode generated summaries and calculate metrics
        generated_lengths = [len(ids) for ids in generated_ids]
        
        # Calculate ROUGE scores for batch
        reference_texts = [
            self.model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in labels
        ]
        generated_texts = [
            self.model.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        
        rouge_scores = self._calculate_batch_rouge(reference_texts, generated_texts)
        
        return {
            'loss': outputs.loss.item(),
            'rouge_scores': rouge_scores,
            'generation_time': generation_time,
            'generated_lengths': generated_lengths
        }
    
    def _calculate_batch_rouge(
        self,
        references: List[str],
        generated: List[str]
    ) -> Dict[str, List[float]]:
        """
        Calculate ROUGE scores for a batch.
        
        Args:
            references: List of reference summaries
            generated: List of generated summaries
            
        Returns:
            Dictionary containing ROUGE scores
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, gen in zip(references, generated):
            scores = self.model.scorer.score(ref, gen)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        return rouge_scores
    
    def validate_output_quality(self, sample_size: int = 5) -> Dict[str, Any]:
        """
        Validate quality of model outputs on sample data.
        
        Args:
            sample_size: Number of samples to validate
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_metrics = {
            'coherence_scores': [],
            'relevance_scores': [],
            'length_distribution': [],
            'diversity_scores': []
        }
        
        print("\nValidating output quality...")
        for i in range(sample_size):
            try:
                # Get random sample from validation set
                idx = torch.randint(len(self.val_loader.dataset), (1,)).item()
                sample = self.val_loader.dataset[idx]
                
                # Generate summary
                article = self.model.tokenizer.decode(
                    sample['input_ids'],
                    skip_special_tokens=True
                )
                generated = self.model.generate_summary(article)
                
                # Calculate quality metrics
                quality_scores = self._calculate_quality_metrics(
                    article,
                    generated
                )
                
                # Update metrics
                for metric, score in quality_scores.items():
                    quality_metrics[metric].append(score)
                
            except Exception as e:
                print(f"Error in quality validation for sample {i}: {e}")
                continue
        
        # Calculate average metrics
        avg_quality_metrics = {
            metric: np.mean(scores) if scores else 0
            for metric, scores in quality_metrics.items()
        }
        
        return avg_quality_metrics
    
    def _calculate_quality_metrics(
        self,
        article: str,
        summary: str
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for a single summary.
        
        Args:
            article: Original article text
            summary: Generated summary
            
        Returns:
            Dictionary containing quality metrics
        """
        # Length ratio (summary length / article length)
        length_ratio = len(summary.split()) / len(article.split())
        
        # Vocabulary diversity (unique words / total words)
        summary_words = summary.split()
        diversity = len(set(summary_words)) / len(summary_words) if summary_words else 0
        
        # Simple relevance score (keyword overlap)
        article_keywords = set(w.lower() for w in article.split())
        summary_keywords = set(w.lower() for w in summary_words)
        relevance = len(article_keywords & summary_keywords) / len(article_keywords)
        
        return {
            'coherence_scores': 1.0,  # Placeholder - would need more sophisticated metrics
            'relevance_scores': relevance,
            'length_distribution': length_ratio,
            'diversity_scores': diversity
        }
