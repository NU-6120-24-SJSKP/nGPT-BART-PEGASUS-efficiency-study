import os
import torch
from transformers import BartTokenizer,BartForConditionalGeneration, get_linear_schedule_with_warmup,BartConfig
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def verify_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close('all')

def create_small_bart_config():
    config = BartConfig(
        vocab_size=50265,  # Standard BART vocabulary size
        max_position_embeddings=512,
        d_model=512,       # Reduced from 768
        encoder_layers=6,  # Reduced from 12
        decoder_layers=6,  # Reduced from 12
        encoder_attention_heads=8,  # Reduced from 16
        decoder_attention_heads=8,  # Reduced from 16
        encoder_ffn_dim=2048,  # Reduced from 3072
        decoder_ffn_dim=2048,  # Reduced from 3072
        activation_function="gelu"
    )
    return config

TRAINING_PARAMS = {
    'MAX_INPUT_LENGTH': 512,
    'MAX_TARGET_LENGTH': 128,
    'BATCH_SIZE': 4,
    'GRADIENT_ACCUMULATION_STEPS': 2,
    'NUM_EPOCHS': 10,
    'LEARNING_RATE': 2e-5,
    'WARMUP_RATIO': 0.1,
    'WEIGHT_DECAY': 0.01,
    'EARLY_STOPPING_PATIENCE': 3
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(num_samples=1000):
    # Load the dataset
    dataset = load_dataset('cnn_dailymail')#, '3.0.0')

    # Calculate how many samples we want for each split
    train_samples = int(0.9 * num_samples)  # 90% of samples for training
    val_samples = num_samples - train_samples  # 10% of samples for validation

    # Randomly select indices for training and validation
    train_indices = range(train_samples)
    val_indices = range(len(dataset['validation']))[:val_samples]

    # Select the samples from the respective splits
    train_data = dataset['train'].select(train_indices)
    val_data = dataset['validation'].select(val_indices)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    return train_data, val_data



class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]['article']
        summary = self.data[idx]['highlights']
        inputs = self.tokenizer(article, max_length=self.max_input_length, truncation=True, padding='max_length', return_tensors='pt')
        targets = self.tokenizer(summary, max_length=self.max_target_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze()
        }

def create_dataloaders(train_data, val_data, tokenizer, batch_size):
    train_dataset = SummarizationDataset(train_data, tokenizer, TRAINING_PARAMS['MAX_INPUT_LENGTH'], TRAINING_PARAMS['MAX_TARGET_LENGTH'])
    val_dataset = SummarizationDataset(val_data, tokenizer, TRAINING_PARAMS['MAX_INPUT_LENGTH'], TRAINING_PARAMS['MAX_TARGET_LENGTH'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    total_time = 0.0
    total_tokens = 0
    peak_memory = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=TRAINING_PARAMS['MAX_TARGET_LENGTH'])
            step_time = time.time() - start_time
            total_time += step_time

            if torch.cuda.is_available():
                current_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_memory = max(peak_memory, current_peak_memory)
            
            total_tokens += sum(len(ids) for ids in generated_ids)
            
            for i in range(len(input_ids)):
                reference = tokenizer.decode(labels[i], skip_special_tokens=True)
                generated_summary = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

                rouge_result = scorer.score(reference, generated_summary)
                for metric in rouge_scores:
                    rouge_scores[metric].append(rouge_result[metric].fmeasure)

    avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    return avg_rouge, total_time, total_tokens, peak_memory



def train_model(model, train_loader, val_loader, tokenizer, device, num_epochs, config):
    try:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': TRAINING_PARAMS['WEIGHT_DECAY'],
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=TRAINING_PARAMS['LEARNING_RATE'])
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(num_training_steps * TRAINING_PARAMS['WARMUP_RATIO'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # Scheduler for reducing learning rate when validation loss stagnates
        scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=4,
            verbose=True
        )
        best_val_loss = float('inf')
        patience = TRAINING_PARAMS['EARLY_STOPPING_PATIENCE']
        no_improve = 0
        model_save_path = "./trained_bart_model.pth"

        train_losses = []
        train_batch_losses = []
        val_perplexities = []
        inference_times = []
        tokens_per_seconds = []
        tokens_seen_so_far = 0
        training_tokens = []
        peak_memory_usages = []
        val_losses = []  
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        start_time = time.time()

        for epoch in range(num_epochs):
            # Initialize these at the start of each epoch
            total_train_loss = 0
            batch_count = 0
            avg_train_loss = float('inf')  # Default value

            try:
                # Clear GPU cache before each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU Memory before epoch {epoch + 1}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                model.train()
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

                for i, batch in enumerate(progress_bar):
                    try:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        current_loss = outputs.loss / TRAINING_PARAMS['GRADIENT_ACCUMULATION_STEPS']
                        current_loss.backward()

                        if (i + 1) % TRAINING_PARAMS['GRADIENT_ACCUMULATION_STEPS'] == 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                        # Update loss tracking
                        train_batch_losses.append(current_loss.item() * TRAINING_PARAMS['GRADIENT_ACCUMULATION_STEPS'])
                        total_train_loss += current_loss.item() * TRAINING_PARAMS['GRADIENT_ACCUMULATION_STEPS']
                        batch_count += 1
                        tokens_seen_so_far += batch['input_ids'].numel()

                        # Update progress bar
                        if batch_count > 0:
                            avg_train_loss = total_train_loss / batch_count
                            progress_bar.set_postfix({'loss': avg_train_loss})

                        # Clear memory after optimization step
                        del outputs, current_loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(f"WARNING: out of memory in batch {i}. Skipping batch...")
                            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                            continue
                        else:
                            raise e

                if batch_count > 0:
                    avg_train_loss = total_train_loss / batch_count
                    train_losses.append(avg_train_loss)

                # Validation phase
                model.eval()
                total_val_loss = 0
                val_batch_count = 0

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                        try:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)

                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            total_val_loss += outputs.loss.item()
                            val_batch_count += 1

                            # Clear memory after each validation batch
                            del outputs
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print("WARNING: out of memory in validation. Skipping batch...")
                                continue
                            else:
                                raise e
                if val_batch_count > 0:
                    avg_val_loss = total_val_loss / val_batch_count
                    val_losses.append(avg_val_loss)
                    val_perplexities.append(torch.exp(torch.tensor(avg_val_loss)).item())
                    training_tokens.append(tokens_seen_so_far)
                    print(f"Epoch {epoch+1}/{num_epochs}")
                    print(f"Average Train Loss: {avg_train_loss:.4f}")
                    print(f"Average Val Loss: {avg_val_loss:.4f}")
                    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")


                    scheduler_plateau.step(avg_val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Updated learning rate after plateau adjustment: {current_lr}")

                    # # Evaluate ROUGE scores
                    # if (epoch + 1) % 2 == 0:
                    print(f"Evaluating after epoch {epoch+1}...")
                    current_rouge, total_time, total_tokens, peak_memory = evaluate(model, val_loader, tokenizer, device)
                    inference_times.append(total_time)
                    tokens_per_seconds.append(total_tokens / total_time if total_time > 0 else 0)
                    peak_memory_usages.append(peak_memory)
                    for metric in rouge_scores:
                        rouge_scores[metric].append(current_rouge[metric])
                        
                    print(f"Current ROUGE Scores: {current_rouge}")
                    
                    # Model saving with error handling
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        try:
                            # torch.save(model.state_dict(), best_model_path)
                            torch.save({
                                'model': model,  # Save the full model
                                'tokenizer': tokenizer,
                                'config': config
                            }, model_save_path)
                            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                        except Exception as e:
                            print(f"Error saving model: {e}")
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                            break

            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}")
                # Attempt to save checkpoint even if epoch fails
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }

                if batch_count > 0:
                    checkpoint_dict['train_loss'] = avg_train_loss
                if 'avg_val_loss' in locals():
                    checkpoint_dict['val_loss'] = avg_val_loss

                torch.save(checkpoint_dict, f'emergency_checkpoint_epoch_{epoch}.pth')
                continue

        end_time = time.time()
        training_time = end_time - start_time

    except Exception as e:
        print(f"Critical training error: {e}")
        raise
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_losses, val_losses, rouge_scores, training_time, train_batch_losses, val_perplexities, inference_times, tokens_per_seconds, peak_memory_usages, training_tokens


def plot_training_progress(train_losses, val_losses, rouge_scores):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 3)
    for metric, scores in rouge_scores.items():
        plt.plot(scores, label=metric)
    plt.legend()
    plt.title('ROUGE Scores')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    # files.download('training_progress.png')

def generate_summary(model, article, tokenizer, device, max_length=None):
    """
    Generate a summary for the given article using the trained model.

    Args:
        model: The trained Pegasus model
        article: Input article text
        tokenizer: Pegasus tokenizer
        device: Device to run generation on
        max_length: Optional override for maximum length. If None, uses TRAINING_PARAMS value
    """
    # Use the same max length as training if not specified
    if max_length is None:
        max_length = TRAINING_PARAMS['MAX_TARGET_LENGTH']

    inputs = tokenizer(
        article,
        max_length=TRAINING_PARAMS['MAX_INPUT_LENGTH'],  # Use consistent input length
        truncation=True,
        return_tensors="pt"
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=max_length,
        early_stopping=True,
        length_penalty=2.0,  # Added for better length control
        min_length=int(max_length/4),  # Added reasonable minimum length
        no_repeat_ngram_size=4  # Prevent repetition
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def inspect_frozen_params(model):
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

    print("\nNumber of frozen parameters:", len(frozen_params))
    print("Number of trainable parameters:", len(trainable_params))

def load_saved_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    config = checkpoint['config']
    return model.to(device), tokenizer, config

def plot_train_loss_per_step(batch_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_losses, label='Training Loss per Step', color='blue', linewidth=0.7)
    plt.xlabel('Batch/Step')
    plt.ylabel('Loss')
    plt.title('Training Loss per Step')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_per_step.png')
    plt.close()

def plot_val_loss_per_epoch(val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_loss_per_epoch.png')
    plt.close()

def plot_train_val_loss(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_val_loss_per_epoch.png')
    plt.close()

def plot_val_perplexity(val_perplexities):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(val_perplexities) + 1), val_perplexities, label='Validation Perplexity', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_perplexity_per_epoch.png')
    plt.close()

def plot_inference_time(inference_times):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inference_times) + 1), inference_times, label='Inference Time (s)', color='purple', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Inference Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('inference_time_vs_epoch.png')
    plt.close()

def plot_tokens_per_second(tokens_per_seconds):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tokens_per_seconds) + 1), tokens_per_seconds, label='Tokens per Second', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Tokens per Second')
    plt.title('Tokens per Second per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('tokens_per_second_vs_epoch.png')
    plt.close()

def plot_peak_memory_usage(peak_memory_usages):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(peak_memory_usages) + 1), peak_memory_usages, label='Peak Memory Usage (MB)', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title('Peak Memory Usage per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('peak_memory_usage_vs_epoch.png')
    plt.close()

def plot_training_tokens_loss(training_tokens,val_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(training_tokens, val_loss)
    plt.title("Validation Loss vs. Training Tokens")
    plt.xlabel("Training Tokens")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.savefig('plot_training_tokens_loss.png')
    plt.close()

def main():
    try:
        set_seed()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        config = create_small_bart_config()
        model = BartForConditionalGeneration(config).to(device)
        total_params,trainable_params = verify_model_size(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        inspect_frozen_params(model)


        train_data, val_data = load_data(num_samples=1000)
        train_loader, val_loader = create_dataloaders(train_data, val_data, tokenizer, TRAINING_PARAMS['BATCH_SIZE'])
        train_losses, val_losses, rouge_scores, training_time, train_batch_losses, val_perplexities, inference_times, tokens_per_seconds, peak_memory_usages, training_tokens = train_model(model, train_loader, val_loader, tokenizer, device, TRAINING_PARAMS['NUM_EPOCHS'], config)
        model_save_path = "./final_bart_model.pth"
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
            'config': config
        }, model_save_path)
        print(f"Final model saved at {model_save_path}")
        plot_training_progress(train_losses, val_losses,rouge_scores)
        plot_train_loss_per_step(train_batch_losses)
        plot_val_loss_per_epoch(val_losses)
        plot_train_val_loss(train_losses, val_losses)
        plot_val_perplexity(val_perplexities)

        plot_inference_time(inference_times)
        plot_tokens_per_second(tokens_per_seconds)
        plot_peak_memory_usage(peak_memory_usages)
        plot_training_tokens_loss(training_tokens,val_losses)

        print("Evaluating fine-tuned model...")
        fine_tuned_rouge = evaluate(model, val_loader, tokenizer, device)
        print("Fine-tuned Model Performance:")
        print(f"ROUGE Scores: {fine_tuned_rouge}")
        print(f"Total training time: {training_time:.2f} seconds")

        print("\nGenerating example summaries...")
        for i in range(3):
            article = val_data[i]["article"]
            reference = val_data[i]["highlights"]
            generated = generate_summary(model, article, tokenizer, device)
            print(f"\nArticle {i+1}:")
            print(f"Reference: {reference}")
            print(f"Generated: {generated}")
            print("-" * 50)
    finally:
        # Add cleanup at the end
        cleanup()

if __name__ == '__main__':
    main()