"""
Main execution module for text summarization project.
Orchestrates the complete pipeline from setup to evaluation.
"""

import torch
from typing import NoReturn
import sys
from pathlib import Path

from pegasus.config import TrainingConfig, PathConfig
from pegasus.model import SummarizationModel
from pegasus.data import DataManager
from pegasus.train import Trainer
from pegasus.test import ModelTester
from pegasus.plot import MetricsPlotter
from pegasus.helpers import (
    set_seed,
    get_device,
    verify_model_size,
    inspect_frozen_params,
    cleanup
)

class SummarizationPipeline:
    """
    Main class for orchestrating the summarization pipeline.
    """
    
    def __init__(self, params):
        """Initialize the pipeline with basic setup."""
        self.setup_basics()
        self.initialize_components(params)
    
    def setup_basics(self) -> None:
        """Perform basic setup operations."""
        try:
            # Set random seed for reproducibility
            set_seed()
            
            # Setup device
            self.device = get_device()
            print(f"Using device: {self.device}")
            
            # Create necessary directories
            Path(PathConfig.BEST_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            print(f"Error in basic setup: {e}")
            sys.exit(1)
    
    def initialize_components(self, params) -> None:
        """Initialize model, tokenizer, and data components."""
        try:
            # Initialize model
            self.model = SummarizationModel(self.device, params=params)
            self.model.initialize_model()
            
            # Verify model size
            total_params, trainable_params = verify_model_size(self.model.model)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            # Inspect frozen parameters
            inspect_frozen_params(self.model.model)
            
            # Load data
            self.data_manager = DataManager()
            self.train_data, self.val_data = self.data_manager.load_data(
                TrainingConfig.NUM_SAMPLES
            )
            
            # Create dataloaders
            self.train_loader, self.val_loader = self.data_manager.create_dataloaders(
                self.train_data,
                self.val_data,
                self.model.tokenizer,
                TrainingConfig.BATCH_SIZE
            )
            
        except Exception as e:
            print(f"Error in component initialization: {e}")
            sys.exit(1)
    
    def train_model(self) -> None:
        """Train the model."""
        try:
            print("\nStarting model training...")
            trainer = Trainer(
                self.model,
                self.train_loader,
                self.val_loader,
                self.device
            )
            
            # Train the model
            metrics, training_time = trainer.train()
            
            print(f"\nTraining completed in {training_time:.2f} seconds")
            
            # Generate training plots
            print("\nGenerating training plots...")
            plotter = MetricsPlotter(save_dir=PathConfig.PLOT_DIR)
            plotter.create_all_plots()
            
        except Exception as e:
            print(f"Error during training: {e}")
            cleanup()
            sys.exit(1)
    
    def evaluate_model(self) -> None:
        """Evaluate the trained model."""
        try:
            print("\nStarting model evaluation...")
            tester = ModelTester(self.model, self.val_data, self.device)
            
            # Generate example summaries
            examples = tester.generate_examples(num_examples=3)
            
            # Evaluate model performance
            rouge_scores, performance_metrics = tester.evaluate_model()
            
            print("\nModel Performance Metrics:")
            print(f"ROUGE Scores: {rouge_scores}")
            print(f"Performance Metrics: {performance_metrics}")
            
            # Analyze error cases
            error_cases = tester.analyze_error_cases(threshold=0.3, num_cases=3)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            cleanup()
            sys.exit(1)
    
    def run_pipeline(self) -> None:
        """Execute the complete pipeline."""
        try:
            self.train_model()
            self.evaluate_model()
            
        except Exception as e:
            print(f"Error in pipeline execution: {e}")
            
        finally:
            cleanup()
            print("\nPipeline execution completed")

def main() -> NoReturn:
    """Main function to execute the summarization pipeline."""
    try:
        print("Initializing Text Summarization Pipeline...")
        pipeline = SummarizationPipeline()
        pipeline.run_pipeline()
        
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user")
        cleanup()
        sys.exit(0)
        
    except Exception as e:
        print(f"Critical error in pipeline execution: {e}")
        cleanup()
        sys.exit(1)

if __name__ == '__main__':
    main()
