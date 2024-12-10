"""
Training module for text summarization model.
Handles training loop, validation, and optimization setup.
"""

import torch
import time
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import TrainingConfig, PathConfig
from metrics import MetricsTracker, Evaluator
from model import SummarizationModel
from helpers import save_checkpoint, print_gpu_utilization

class Trainer:
    """
    Class for managing the training process of the summarization model.
    """
    
    def __init__(
        self,
        model: SummarizationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The summarization model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metrics_tracker = MetricsTracker()
        self.evaluator = Evaluator(model.model, model.tokenizer, device)
        
        # Initialize best validation loss for model saving
        self.best_val_loss = float('inf')
        self.no_improve_count = 0
        
    def setup_optimization(self) -> Tuple[torch.optim.Optimizer, Any, Any]:
        """
        Setup optimizer and schedulers.
        
        Returns:
            Tuple containing optimizer and schedulers
        """
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.get_optimizer_groups(),
            lr=TrainingConfig.LEARNING_RATE
        )
        
        # Create learning rate schedulers
        num_training_steps = len(self.train_loader) * TrainingConfig.NUM_EPOCHS
        num_warmup_steps = int(num_training_steps * TrainingConfig.WARMUP_RATIO)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=4,
            verbose=True
        )
        
        return optimizer, scheduler, scheduler_plateau
        
    def train_epoch(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            optimizer: The optimizer
            scheduler: The learning rate scheduler
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_train_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{TrainingConfig.NUM_EPOCHS} [Train]")
        
        for i, batch in enumerate(progress_bar):
            try:
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
                
                # Calculate loss and backward pass
                current_loss = outputs.loss / TrainingConfig.GRADIENT_ACCUMULATION_STEPS
                current_loss.backward()
                
                # Update weights if gradient accumulation steps reached
                if (i + 1) % TrainingConfig.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update metrics
                total_train_loss += current_loss.item() * TrainingConfig.GRADIENT_ACCUMULATION_STEPS
                batch_count += 1
                self.metrics_tracker.metrics["train_batch_losses"].append(
                    current_loss.item() * TrainingConfig.GRADIENT_ACCUMULATION_STEPS
                )
                
                # Update progress bar
                if batch_count > 0:
                    avg_train_loss = total_train_loss / batch_count
                    progress_bar.set_postfix({'loss': avg_train_loss})
                
                # Clear memory
                del outputs, current_loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: out of memory in batch {i}. Skipping batch...")
                    print_gpu_utilization()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    continue
                raise e
                
        return total_train_loss / batch_count if batch_count > 0 else float('inf')
        
    def validate_epoch(self, epoch: int) -> float:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        total_val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{TrainingConfig.NUM_EPOCHS} [Val]'):
                try:
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
                    
                    # Update metrics
                    total_val_loss += outputs.loss.item()
                    val_batch_count += 1
                    
                    # Clear memory
                    del outputs
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory in validation. Skipping batch...")
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        continue
                    raise e
                    
        return total_val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
    def train(self) -> Tuple[Dict[str, Any], float]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Tuple containing metrics dictionary and total training time
        """
        optimizer, scheduler, scheduler_plateau = self.setup_optimization()
        start_time = time.time()
        
        try:
            for epoch in range(TrainingConfig.NUM_EPOCHS):
                # Training phase
                avg_train_loss = self.train_epoch(epoch, optimizer, scheduler)
                self.metrics_tracker.update_train_metrics(avg_train_loss, avg_train_loss)
                
                # Validation phase
                avg_val_loss = self.validate_epoch(epoch)
                self.metrics_tracker.update_val_metrics(avg_val_loss)
                
                # Print epoch statistics
                print(f"\nEpoch {epoch+1}/{TrainingConfig.NUM_EPOCHS}")
                print(f"Average Train Loss: {avg_train_loss:.4f}")
                print(f"Average Val Loss: {avg_val_loss:.4f}")
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
                
                # Evaluate and update metrics
                rouge_scores, total_time, total_tokens, peak_memory = self.evaluator.evaluate(
                    self.val_loader
                )
                self.metrics_tracker.update_performance_metrics(total_time, total_tokens, peak_memory)
                self.metrics_tracker.update_rouge_scores(rouge_scores)
                
                # Save metrics
                self.metrics_tracker.save_metrics()
                
                # Model checkpoint saving
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.model.save_model(PathConfig.BEST_MODEL_PATH)
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    
                # Learning rate adjustment
                scheduler_plateau.step(avg_val_loss)
                
                # Early stopping check
                if self.no_improve_count >= TrainingConfig.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
        except Exception as e:
            print(f"Error during training: {e}")
            # Save emergency checkpoint
            save_checkpoint(
                self.model.model,
                optimizer,
                scheduler,
                epoch,
                avg_train_loss,
                avg_val_loss
            )
            raise
            
        finally:
            # Clean up
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        total_time = time.time() - start_time
        return self.metrics_tracker.metrics, total_time
