import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup,BartTokenizer
from tqdm import tqdm
import pickle
import time
from bart.validate import Validator
import os
from bart.config import TrainingConfig, create_small_bart_config
from bart.data import load_data, create_dataloaders
from bart.helper import set_seed, verify_model_size, cleanup, inspect_frozen_params
from bart.model import SummarizationModel
from bart.metrics import evaluate
from bart.plot import (plot_rouge_scores, plot_train_loss_per_step, 
                 plot_val_loss_per_epoch, plot_train_val_loss, 
                 plot_val_perplexity, plot_inference_time, 
                 plot_tokens_per_second, plot_peak_memory_usage, 
                 plot_training_tokens_loss)

class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, device, config, training_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.training_config = training_config
        self.validator = Validator(model, val_loader, tokenizer, device)
        
        self.metrics = {
            "train_losses": [],
            "train_batch_losses": [],
            "val_perplexities": [],
            "inference_times": [],
            "tokens_per_seconds": [],
            "training_tokens": [],
            "peak_memory_usages": [],
            "val_losses": [],
            "rouge_scores": {"rouge1": [], "rouge2": [], "rougeL": []}
        }

    def setup_optimization(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.training_config.WEIGHT_DECAY,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, 
                             lr=self.training_config.LEARNING_RATE)
        
        num_training_steps = len(self.train_loader) * self.training_config.NUM_EPOCHS
        num_warmup_steps = int(num_training_steps * self.training_config.WARMUP_RATIO)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=4,
            verbose=True
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_train_loss = 0
        batch_count = 0
        tokens_seen_so_far = 0
        
        progress_bar = tqdm(self.train_loader, 
                          desc=f"Epoch {epoch+1}/{self.training_config.NUM_EPOCHS} [Train]")
        
        for i, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   labels=labels)
                
                current_loss = outputs.loss / self.training_config.GRADIENT_ACCUMULATION_STEPS
                current_loss.backward()
                
                if (i + 1) % self.training_config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                self.metrics["train_batch_losses"].append(
                    current_loss.item() * self.training_config.GRADIENT_ACCUMULATION_STEPS)
                total_train_loss += current_loss.item() * self.training_config.GRADIENT_ACCUMULATION_STEPS
                batch_count += 1
                tokens_seen_so_far += batch['input_ids'].numel()
                
                if batch_count > 0:
                    avg_train_loss = total_train_loss / batch_count
                    progress_bar.set_postfix({'loss': avg_train_loss})
                
                del outputs, current_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"WARNING: out of memory in batch {i}. Skipping batch...")
                    continue
                else:
                    raise e
                    
        return total_train_loss / batch_count if batch_count > 0 else float('inf'), tokens_seen_so_far

    def save_checkpoint(self, model_path):
        try:
            torch.save({
                'model': self.model,
                'tokenizer': self.tokenizer,
                'config': self.config
            }, model_path)
            print(f"Model saved successfully at {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def save_metrics(self, pickle_file="training_metrics.pkl"):
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.metrics, f)

    def train(self):
        self.setup_optimization()
        best_val_loss = float('inf')
        no_improve = 0
        start_time = time.time()
        
        try:
            for epoch in range(self.training_config.NUM_EPOCHS):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                avg_train_loss, tokens_seen = self.train_epoch(epoch)
                
                # Validation phase
                val_loss, val_perplexity = self.validator.validate_epoch()
                
                # Update metrics
                self.metrics["train_losses"].append(avg_train_loss)
                self.metrics["val_losses"].append(val_loss)
                self.metrics["val_perplexities"].append(val_perplexity)
                self.metrics["training_tokens"].append(tokens_seen)
                
                print(f"Epoch {epoch+1}/{self.training_config.NUM_EPOCHS}")
                print(f"Average Train Loss: {avg_train_loss:.4f}")
                print(f"Average Val Loss: {val_loss:.4f}")
                print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
                
                self.scheduler_plateau.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("./best_model.pth")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.training_config.EARLY_STOPPING_PATIENCE:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        break
                
                self.save_metrics()
                
        except Exception as e:
            print(f"Training error: {e}")
            self.save_metrics()
            self._save_emergency_checkpoint(epoch)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return time.time() - start_time

    def _save_emergency_checkpoint(self, epoch):
        try:
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }
            torch.save(checkpoint_dict, f'emergency_checkpoint_epoch_{epoch}.pth')
        except Exception as e:
            print(f"Failed to save emergency checkpoint: {e}")

def run_train(params):
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # Setup
        set_seed()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        config = create_small_bart_config(params)
        summarization_model = SummarizationModel(config, device)
        model = summarization_model.model
        
        # Verify model parameters
        total_params, trainable_params = verify_model_size(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        inspect_frozen_params(model)
        
        # Load data
        train_data, val_data = load_data(num_samples=15000)
        train_loader, val_loader = create_dataloaders(
            train_data, 
            val_data, 
            tokenizer, 
            TrainingConfig.BATCH_SIZE, 
            TrainingConfig.MAX_INPUT_LENGTH, 
            TrainingConfig.MAX_TARGET_LENGTH
        )
        
        # Initialize trainer and start training
        trainer = Trainer(
            model, 
            train_loader, 
            val_loader, 
            tokenizer, 
            device, 
            config, 
            TrainingConfig
        )
        training_time = trainer.train()
        
        # Save final model
        model_save_path = "./final_bart_model.pth"
        summarization_model.save_model(model_save_path, config)
        print(f"Final model saved at {model_save_path}")
        
        # Generate plots
        plot_rouge_scores()
        plot_train_loss_per_step()
        plot_val_loss_per_epoch()
        plot_train_val_loss()
        plot_val_perplexity()
        plot_inference_time()
        plot_tokens_per_second()
        plot_peak_memory_usage()
        plot_training_tokens_loss()
        
        # Final evaluation
        print("Evaluating fine-tuned model...")
        rouge_scores, inference_time, total_tokens, peak_memory = evaluate(
            model=model,
            data_loader=val_loader,
            tokenizer=tokenizer,
            device=device
        )
        print("Fine-tuned Model Performance:")
        print(f"ROUGE Scores: {rouge_scores}")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Tokens processed: {total_tokens}")
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        
        print("\nGenerating example summaries...")
        for i in range(3):
            article = val_data[i]["article"]
            reference = val_data[i]["highlights"]
            generated = summarization_model.generate_summary(article)
            print(f"\nArticle {i+1}:")
            print(f"Reference: {reference}")
            print(f"Generated: {generated}")
            print("-" * 50)
            
    finally:
        cleanup()
