"""
Testing and evaluation module for text summarization model.
Handles model testing, example generation, and performance evaluation.
"""

import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rouge_score import rouge_scorer
from dataclasses import dataclass

from pegasus.model import SummarizationModel

@dataclass
class TestExample:
    """Data class for storing test examples and their summaries."""
    article: str
    reference: str
    generated: str
    rouge_scores: Dict[str, float]

class ModelTester:
    """
    Class for testing and evaluating the summarization model.
    """
    
    def __init__(
        self,
        model: SummarizationModel,
        test_data: Any,
        device: torch.device
    ):
        """
        Initialize the tester.
        
        Args:
            model: The summarization model
            test_data: Test dataset
            device: Device to run testing on
        """
        self.model = model
        self.test_data = test_data
        self.device = device
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
    def generate_examples(self, num_examples: int = 3) -> List[TestExample]:
        """
        Generate example summaries from the test dataset.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of TestExample objects containing results
        """
        examples = []
        self.model.eval()
        
        print("\nGenerating example summaries...")
        for i in range(min(num_examples, len(self.test_data))):
            try:
                # Get article and reference summary
                article = self.test_data[i]["article"]
                reference = self.test_data[i]["highlights"]
                
                # Generate summary
                generated = self.model.generate_summary(article)
                
                # Calculate ROUGE scores
                scores = self.scorer.score(reference, generated)
                rouge_scores = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                }
                
                # Create and store example
                example = TestExample(
                    article=article,
                    reference=reference,
                    generated=generated,
                    rouge_scores=rouge_scores
                )
                examples.append(example)
                
                # Print results
                print(f"\nExample {i+1}:")
                print(f"Reference: {reference}")
                print(f"Generated: {generated}")
                print(f"ROUGE Scores: {rouge_scores}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error generating example {i+1}: {e}")
                continue
                
        return examples
    
    def evaluate_model(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the model on the test dataset.
        
        Returns:
            Tuple containing ROUGE scores and performance metrics
        """
        self.model.eval()
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        performance_metrics = {
            'total_time': 0.0,
            'total_tokens': 0,
            'peak_memory': 0.0
        }
        
        print("\nEvaluating model on test dataset...")
        with torch.no_grad():
            for i, item in enumerate(tqdm(self.test_data)):
                try:
                    # Generate summary
                    article = item["article"]
                    reference = item["highlights"]
                    
                    # Track memory before generation
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(self.device)
                        start_memory = torch.cuda.memory_allocated(self.device)
                    
                    # Generate and time the summary
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    generated = self.model.generate_summary(article)
                    end_time.record()
                    
                    # Synchronize CUDA events
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
                        peak_memory = (torch.cuda.max_memory_allocated(self.device) - start_memory) / (1024 ** 2)  # MB
                    else:
                        generation_time = 0
                        peak_memory = 0
                    
                    # Calculate ROUGE scores
                    scores = self.scorer.score(reference, generated)
                    for metric in rouge_scores:
                        rouge_scores[metric].append(scores[metric].fmeasure)
                    
                    # Update performance metrics
                    performance_metrics['total_time'] += generation_time
                    performance_metrics['total_tokens'] += len(self.model.tokenizer.encode(generated))
                    performance_metrics['peak_memory'] = max(
                        performance_metrics['peak_memory'],
                        peak_memory
                    )
                    
                except Exception as e:
                    print(f"Error processing test item {i}: {e}")
                    continue
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate average ROUGE scores
        avg_rouge_scores = {
            metric: sum(scores) / len(scores) if scores else 0
            for metric, scores in rouge_scores.items()
        }
        
        # Calculate final performance metrics
        final_performance_metrics = {
            'average_time_per_summary': performance_metrics['total_time'] / len(self.test_data),
            'tokens_per_second': (performance_metrics['total_tokens'] / 
                                performance_metrics['total_time'] if performance_metrics['total_time'] > 0 else 0),
            'peak_memory_usage_mb': performance_metrics['peak_memory']
        }
        
        return avg_rouge_scores, final_performance_metrics
    
    def analyze_error_cases(self, threshold: float = 0.3, num_cases: int = 5) -> List[TestExample]:
        """
        Analyze cases where the model performed poorly.
        
        Args:
            threshold: ROUGE score threshold for considering a case as error
            num_cases: Number of error cases to analyze
            
        Returns:
            List of TestExample objects containing error cases
        """
        error_cases = []
        self.model.eval()
        
        print("\nAnalyzing error cases...")
        for i, item in enumerate(self.test_data):
            if len(error_cases) >= num_cases:
                break
                
            try:
                # Generate summary
                article = item["article"]
                reference = item["highlights"]
                generated = self.model.generate_summary(article)
                
                # Calculate ROUGE scores
                scores = self.scorer.score(reference, generated)
                rouge_scores = {
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure,
                    'rougeL': scores['rougeL'].fmeasure
                }
                
                # Check if this is an error case
                if any(score < threshold for score in rouge_scores.values()):
                    error_case = TestExample(
                        article=article,
                        reference=reference,
                        generated=generated,
                        rouge_scores=rouge_scores
                    )
                    error_cases.append(error_case)
                    
                    # Print error case
                    print(f"\nError Case {len(error_cases)}:")
                    print(f"Reference: {reference}")
                    print(f"Generated: {generated}")
                    print(f"ROUGE Scores: {rouge_scores}")
                    print("-" * 50)
                    
            except Exception as e:
                print(f"Error analyzing test item {i}: {e}")
                continue
                
        return error_cases
