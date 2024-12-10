#!pip install transformers datasets rouge_score nltk tqdm matplotlib

import os
import pickle
import torch
from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from google.colab import drive, files

# Mount Google Drive
drive.mount('/content/drive')

def verify_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close('all')

def create_small_pegasus_config():
    '''Create a smaller Pegasus configuration suitable for training on 3000 samples.'''
    config = PegasusConfig(
        vocab_size=96103,  # Original vocab size for tokenizer compatibility
        encoder_layers=8,  # Reduced from 16
        decoder_layers=8,  # Reduced from 16
        encoder_attention_heads=16,  # Reduced from 16
        decoder_attention_heads=16,  # Reduced from 16
        encoder_ffn_dim=2048,  # Reduced from 4096
        decoder_ffn_dim=2048,  # Reduced from 4096
        d_model=512,  # Reduced from 1024
        max_position_embeddings=512,  # Reduced context length
        pad_token_id=0,
        eos_token_id=1,
        forced_eos_token_id=1,
        activation_function='gelu',
        dropout=0.2,  # Increased dropout for smaller dataset
        attention_dropout=0.2,
        activation_dropout=0.2,
        num_beams=4,
        encoder_layerdrop=0.1,  # Added layerdrop for regularization
        decoder_layerdrop=0.1,
        scale_embedding=True,
        use_cache=True,
        is_encoder_decoder=True
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
    'EARLY_STOPPING_PATIENCE': 3,
    'num_samples': 15000
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# def load_data(num_samples=3000):
#     dataset = load_dataset('cnn_dailymail', '3.0.0')
#     full_train_data = dataset['train'].select(range(num_samples))
#     train_size = int(0.9 * len(full_train_data))
#     train_data = full_train_data.select(range(train_size))
#     val_data = full_train_data.select(range(train_size, len(full_train_data)))
#     return train_data, val_data

def load_data(num_samples=1000):
    # Load the dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')

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
            total_tokens += sum(len(ids) for ids in generated_ids)

            if torch.cuda.is_available():
                current_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_memory = max(peak_memory, current_peak_memory)

            for i in range(len(input_ids)):
                reference = tokenizer.decode(labels[i], skip_special_tokens=True)
                generated_summary = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

                rouge_result = scorer.score(reference, generated_summary)
                for metric in rouge_scores:
                    rouge_scores[metric].append(rouge_result[metric].fmeasure)

    avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    return avg_rouge, total_time, total_tokens, peak_memory



def train_model(model, train_loader, val_loader, tokenizer, device, num_epochs):

    #Add metrics dictionary here
    metrics = {
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
        best_model_path = "/content/drive/My Drive/NLP-Project/best_custom_pegasus-TEST.pt"
        patience = TRAINING_PARAMS['EARLY_STOPPING_PATIENCE']
        no_improve = 0

        # train_losses = []
        # val_losses = []
        # rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        # train_batch_losses = []  # For per-step losses
        # val_perplexities = []   # For validation perplexity
        # inference_times = []    # For inference timing
        # tokens_per_seconds = [] # For throughput
        # peak_memory_usages = [] # For memory usage

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
                        total_train_loss += current_loss.item() * TRAINING_PARAMS['GRADIENT_ACCUMULATION_STEPS']
                        batch_count += 1
                        metrics["train_batch_losses"].append(current_loss.item() * TRAINING_PARAMS['GRADIENT_ACCUMULATION_STEPS'])

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

                # if batch_count > 0:
                #     avg_train_loss = total_train_loss / batch_count
                #     train_losses.append(avg_train_loss)


                if batch_count > 0:
                    avg_train_loss = total_train_loss / batch_count
                    metrics["train_losses"].append(avg_train_loss)

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
                    metrics["val_losses"].append(avg_val_loss)
                    metrics["val_perplexities"].append(torch.exp(torch.tensor(avg_val_loss)).item())

                # if val_batch_count > 0:
                #     avg_val_loss = total_val_loss / val_batch_count
                #     val_losses.append(avg_val_loss)

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
                    # inference_times.append(total_time)
                    # tokens_per_seconds.append(total_tokens / total_time if total_time > 0 else 0)
                    # peak_memory_usages.append(peak_memory)

                    # for metric in rouge_scores:
                    #     rouge_scores[metric].append(current_rouge[metric])

                    metrics["inference_times"].append(total_time)
                    metrics["tokens_per_seconds"].append(total_tokens / total_time if total_time > 0 else 0)
                    metrics["peak_memory_usages"].append(peak_memory)

                    for metric in metrics["rouge_scores"]:
                        metrics["rouge_scores"][metric].append(current_rouge[metric])

                    # print(f"Current ROUGE Scores: {current_rouge}")
                    with open("training_metrics.pkl", 'wb') as f:
                        pickle.dump(metrics, f)


                    


                    # Model saving with error handling
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        try:
                            torch.save(model.state_dict(), best_model_path)
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

    # return train_losses, val_losses, rouge_scores, training_time, train_batch_losses, val_perplexities, inference_times, tokens_per_seconds, peak_memory_usages
    return metrics, training_time


def plot_training_progress():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(saved_metrics["train_losses"], label='Train Loss')
    plt.plot(saved_metrics["val_losses"], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot ROUGE scores
    plt.subplot(2, 2, 3)
    for metric, scores in saved_metrics["rouge_scores"].items():
        plt.plot(scores, label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('ROUGE Scores')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def plot_train_loss_per_step():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(saved_metrics["train_batch_losses"], label='Training Loss per Step', color='blue', linewidth=0.7)
    plt.xlabel('Batch/Step')
    plt.ylabel('Loss')
    plt.title('Training Loss per Step')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_per_step.png')
    plt.close()

def plot_val_loss_per_epoch():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["val_losses"]) + 1), saved_metrics["val_losses"], 
             label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_loss_per_epoch.png')
    plt.close()

def plot_train_val_loss():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    epochs = range(1, len(saved_metrics["train_losses"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, saved_metrics["train_losses"], label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, saved_metrics["val_losses"], label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_val_loss_per_epoch.png')
    plt.close()

def plot_val_perplexity():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["val_perplexities"]) + 1), 
             saved_metrics["val_perplexities"], label='Validation Perplexity', 
             color='green', marker='o')
    plt.yscale('log')  # Using log scale for perplexity
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_perplexity_per_epoch.png')
    plt.close()

def plot_inference_time():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["inference_times"]) + 1), 
             saved_metrics["inference_times"], label='Inference Time (s)', 
             color='purple', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Inference Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('inference_time_vs_epoch.png')
    plt.close()

def plot_tokens_per_second():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["tokens_per_seconds"]) + 1), 
             saved_metrics["tokens_per_seconds"], label='Tokens per Second', 
             color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Tokens per Second')
    plt.title('Tokens per Second per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('tokens_per_second_vs_epoch.png')
    plt.close()

def plot_peak_memory_usage():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["peak_memory_usages"]) + 1), 
             saved_metrics["peak_memory_usages"], label='Peak Memory Usage (MB)', 
             color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title('Peak Memory Usage per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('peak_memory_usage_vs_epoch.png')
    plt.close()

def plot_rouge_scores():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(12, 8))
    for metric, scores in saved_metrics["rouge_scores"].items():
        plt.plot(scores, label=metric)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('ROUGE Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('rouge_scores.png')
    plt.close()


# def generate_summary(model, article, tokenizer, device, max_length=128):
#     inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt").to(device)
#     summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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

def main():
    try:
        set_seed()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
        config = create_small_pegasus_config()
        model = PegasusForConditionalGeneration(config).to(device)
        total_params,trainable_params = verify_model_size(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        inspect_frozen_params(model)


        train_data, val_data = load_data(TRAINING_PARAMS['num_samples'])
        train_loader, val_loader = create_dataloaders(train_data, val_data, tokenizer, TRAINING_PARAMS['BATCH_SIZE'])
        #train_losses, val_losses, rouge_scores, training_time, train_batch_losses, val_perplexities, inference_times, tokens_per_seconds, peak_memory_usages = train_model(model, train_loader, val_loader, tokenizer, device, TRAINING_PARAMS['NUM_EPOCHS'])
        metrics, training_time = train_model(model, train_loader, val_loader, tokenizer, device, TRAINING_PARAMS['NUM_EPOCHS'])

        print("Generating training plots...")
        plot_train_loss_per_step()
        plot_val_loss_per_epoch()
        plot_train_val_loss()
        plot_val_perplexity()
        plot_inference_time()
        plot_tokens_per_second()
        plot_peak_memory_usage()
        plot_rouge_scores()    





        print("Evaluating fine-tuned model...")
        fine_tuned_rouge, total_time, total_tokens, peak_memory = evaluate(model, val_loader, tokenizer, device)
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