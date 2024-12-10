
# # Install required libraries
# #!pip install transformers datasets rouge_score nltk tqdm matplotlib
# import random
# import time
# import os
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, Dataset
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer, get_linear_schedule_with_warmup
# from datasets import load_dataset
# from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu
# from google.colab import drive, files

# # Mount Google Drive
# drive.mount('/content/drive')


# # Constants
# MAX_INPUT_LENGTH = 1024
# MAX_TARGET_LENGTH = 128
# BATCH_SIZE = 4
# GRADIENT_ACCUMULATION_STEPS = 4
# NUM_EPOCHS = 10
# LEARNING_RATE = 5e-5

# def set_seed(seed=42):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def load_data(num_samples=1000):
#     dataset = load_dataset("cnn_dailymail", "3.0.0")
#     full_train_data = dataset["train"].select(range(num_samples))
#     train_size = int(0.9 * len(full_train_data))
#     train_data = full_train_data.select(range(train_size))
#     val_data = full_train_data.select(range(train_size, len(full_train_data)))
#     return train_data, val_data

# class SummarizationDataset(Dataset):
#     def __init__(self, data, tokenizer, max_input_length=MAX_INPUT_LENGTH, max_target_length=MAX_TARGET_LENGTH):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_input_length = max_input_length
#         self.max_target_length = max_target_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         article = self.data[idx]["article"]
#         summary = self.data[idx]["highlights"]
#         inputs = self.tokenizer(article, max_length=self.max_input_length, truncation=True, padding="max_length", return_tensors="pt")
#         targets = self.tokenizer(summary, max_length=self.max_target_length, truncation=True, padding="max_length", return_tensors="pt")
#         return {
#             "input_ids": inputs.input_ids.squeeze(),
#             "attention_mask": inputs.attention_mask.squeeze(),
#             "labels": targets.input_ids.squeeze()
#         }

# def create_dataloaders(train_data, val_data, tokenizer):
#     train_dataset = SummarizationDataset(train_data, tokenizer)
#     val_dataset = SummarizationDataset(val_data, tokenizer)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
#     return train_loader, val_loader

# def evaluate(model, data_loader, tokenizer, device):
#     model.eval()
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     bleu_scores = []
#     rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Evaluating"):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_TARGET_LENGTH)
            
#             for i in range(len(input_ids)):
#                 reference = tokenizer.decode(labels[i], skip_special_tokens=True)
#                 generated_summary = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

#                 bleu_score = sentence_bleu([reference.split()], generated_summary.split())
#                 bleu_scores.append(bleu_score)

#                 rouge_result = scorer.score(reference, generated_summary)
#                 for metric in rouge_scores:
#                     rouge_scores[metric].append(rouge_result[metric].fmeasure)

#     avg_bleu = sum(bleu_scores) / len(bleu_scores)
#     avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

#     return avg_bleu, avg_rouge

# def train_model(model, train_loader, val_loader, tokenizer, device, num_epochs):
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
#     num_training_steps = len(train_loader) * num_epochs
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

#     best_val_loss = float('inf')
#     best_model_path = '/content/drive/My Drive/NLP-Project/best_pegasus_model_modular_script_test.pth'
    
#     train_losses = []
#     val_losses = []
#     bleu_scores = []
#     rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

#     start_time = time.time()

#     for epoch in range(num_epochs):
#         model.train()
#         total_train_loss = 0
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
#         for i, batch in enumerate(progress_bar):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)

#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             total_train_loss += loss.item()
            
#             loss = loss / GRADIENT_ACCUMULATION_STEPS
#             loss.backward()
            
#             if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             progress_bar.set_postfix({"train_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

#         avg_train_loss = total_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
        
#         # Validation
#         model.eval()
#         total_val_loss = 0
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["labels"].to(device)

#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#                 loss = outputs.loss
#                 total_val_loss += loss.item()

#         avg_val_loss = total_val_loss / len(val_loader)
#         val_losses.append(avg_val_loss)
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#         # Evaluate every 2 epochs
#         if (epoch + 1) % 2 == 0:
#             print(f"Evaluating after epoch {epoch+1}...")
#             current_bleu, current_rouge = evaluate(model, val_loader, tokenizer, device)
#             bleu_scores.append(current_bleu)
#             for metric in rouge_scores:
#                 rouge_scores[metric].append(current_rouge[metric])
#             print(f"Current BLEU Score: {current_bleu}")
#             print(f"Current ROUGE Scores: {current_rouge}")

#         # Save the model if it's the best so far based on validation loss
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), best_model_path)
#             print(f"New best model saved with validation loss: {best_val_loss:.4f}")

#     end_time = time.time()
#     training_time = end_time - start_time
#     print(f"Total training time: {training_time:.2f} seconds")

#     return train_losses, val_losses, bleu_scores, rouge_scores, training_time

# def plot_training_progress(train_losses, val_losses, bleu_scores, rouge_scores):
#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.legend()
#     plt.title('Training and Validation Loss')

#     plt.subplot(2, 2, 2)
#     plt.plot(bleu_scores)
#     plt.title('BLEU Score')

#     plt.subplot(2, 2, 3)
#     for metric, scores in rouge_scores.items():
#         plt.plot(scores, label=metric)
#     plt.legend()
#     plt.title('ROUGE Scores')

#     plt.tight_layout()
#     plt.savefig('training_progress.png')
#     plt.close()
    
#     # Download the plot
#     files.download('training_progress.png')

# def generate_summary(model, article, tokenizer, device, max_length=128):
#     inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt").to(device)
#     summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# def main():
#     set_seed()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
#     tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
#     model.to(device)

#     train_data, val_data = load_data(num_samples=5000)  
#     train_loader, val_loader = create_dataloaders(train_data, val_data, tokenizer)

#     print("Evaluating base model...")
#     base_bleu, base_rouge = evaluate(model, val_loader, tokenizer, device)
#     print("Base Model Performance:")
#     print(f"BLEU Score: {base_bleu}")
#     print(f"ROUGE Scores: {base_rouge}")

#     train_losses, val_losses, bleu_scores, rouge_scores, training_time = train_model(model, train_loader, val_loader, tokenizer, device, NUM_EPOCHS)

#     plot_training_progress(train_losses, val_losses, bleu_scores, rouge_scores)

#     print("Evaluating fine-tuned model...")
#     fine_tuned_bleu, fine_tuned_rouge = evaluate(model, val_loader, tokenizer, device)
#     print("Fine-tuned Model Performance:")
#     print(f"BLEU Score: {fine_tuned_bleu}")
#     print(f"ROUGE Scores: {fine_tuned_rouge}")

#     print("Performance Improvement:")
#     print(f"BLEU: {fine_tuned_bleu - base_bleu}")
#     print(f"ROUGE-1: {fine_tuned_rouge['rouge1'] - base_rouge['rouge1']}")
#     print(f"ROUGE-2: {fine_tuned_rouge['rouge2'] - base_rouge['rouge2']}")
#     print(f"ROUGE-L: {fine_tuned_rouge['rougeL'] - base_rouge['rougeL']}")

#     print(f"Total training time: {training_time:.2f} seconds")

#     # Generate example summaries
#     print("\nGenerating example summaries...")
#     for i in range(3):
#         article = val_data[i]["article"]
#         reference = val_data[i]["highlights"]
#         generated = generate_summary(model, article, tokenizer, device)
#         print(f"\nArticle {i+1}:")
#         print(f"Reference: {reference}")
#         print(f"Generated: {generated}")
#         print("-" * 50)

# if __name__ == "__main__":
#     main()



! pip install transformers datasets rouge_score nltk tqdm matplotlib
import random
import time
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from google.colab import drive, files

# Mount Google Drive
drive.mount('/content/drive')

# Install required libraries
!pip install transformers datasets rouge_score nltk tqdm matplotlib

# Modified constants
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5  # Reduced from 5e-5
WARMUP_RATIO = 0.1  # Added warmup
WEIGHT_DECAY = 0.01  # Added weight decay

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(num_samples=1000):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    full_train_data = dataset["train"].select(range(num_samples))
    train_size = int(0.9 * len(full_train_data))
    train_data = full_train_data.select(range(train_size))
    val_data = full_train_data.select(range(train_size, len(full_train_data)))
    return train_data, val_data

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=MAX_INPUT_LENGTH, max_target_length=MAX_TARGET_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        summary = self.data[idx]["highlights"]
        inputs = self.tokenizer(article, max_length=self.max_input_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(summary, max_length=self.max_target_length, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

def create_dataloaders(train_data, val_data, tokenizer):
    train_dataset = SummarizationDataset(train_data, tokenizer)
    val_dataset = SummarizationDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_TARGET_LENGTH)

            for i in range(len(input_ids)):
                reference = tokenizer.decode(labels[i], skip_special_tokens=True)
                generated_summary = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

                bleu_score = sentence_bleu([reference.split()], generated_summary.split())
                bleu_scores.append(bleu_score)

                rouge_result = scorer.score(reference, generated_summary)
                for metric in rouge_scores:
                    rouge_scores[metric].append(rouge_result[metric].fmeasure)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    return avg_bleu, avg_rouge

def train_model(model, train_loader, val_loader, tokenizer, device, num_epochs):
    # Modified optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    # Modified scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    best_model_path = '/content/drive/My Drive/NLP-Project/best_pegasus_model_20000_data.pth'
    patience = 3  # Early stopping patience
    no_improve = 0

    train_losses = []
    val_losses = []
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        # Training loop with gradient clipping
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"train_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation with early stopping
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Average Val Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")


        # Evaluate every 2 epochs
        if (epoch + 1) % 2 == 0:
            print(f"Evaluating after epoch {epoch+1}...")
            current_bleu, current_rouge = evaluate(model, val_loader, tokenizer, device)
            bleu_scores.append(current_bleu)
            for metric in rouge_scores:
                rouge_scores[metric].append(current_rouge[metric])
            print(f"Current BLEU Score: {current_bleu}")
            print(f"Current ROUGE Scores: {current_rouge}")

        # Save best model and check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break




    end_time = time.time()
    training_time = end_time - start_time

    return train_losses, val_losses, bleu_scores, rouge_scores, training_time

def plot_training_progress(train_losses, val_losses, bleu_scores, rouge_scores):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(2, 2, 2)
    plt.plot(bleu_scores)
    plt.title('BLEU Score')

    plt.subplot(2, 2, 3)
    for metric, scores in rouge_scores.items():
        plt.plot(scores, label=metric)
    plt.legend()
    plt.title('ROUGE Scores')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

    # Download the plot
    files.download('training_progress.png')

def generate_summary(model, article, tokenizer, device, max_length=128):
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
    model.to(device)

    train_data, val_data = load_data(num_samples=10000)
    train_loader, val_loader = create_dataloaders(train_data, val_data, tokenizer)

    print("Evaluating base model...")
    base_bleu, base_rouge = evaluate(model, val_loader, tokenizer, device)
    print("Base Model Performance:")
    print(f"BLEU Score: {base_bleu}")
    print(f"ROUGE Scores: {base_rouge}")

    train_losses, val_losses, bleu_scores, rouge_scores, training_time = train_model(model, train_loader, val_loader, tokenizer, device, NUM_EPOCHS)

    plot_training_progress(train_losses, val_losses, bleu_scores, rouge_scores)

    print("Evaluating fine-tuned model...")
    fine_tuned_bleu, fine_tuned_rouge = evaluate(model, val_loader, tokenizer, device)
    print("Fine-tuned Model Performance:")
    print(f"BLEU Score: {fine_tuned_bleu}")
    print(f"ROUGE Scores: {fine_tuned_rouge}")

    print("Performance Improvement:")
    print(f"BLEU: {fine_tuned_bleu - base_bleu}")
    print(f"ROUGE-1: {fine_tuned_rouge['rouge1'] - base_rouge['rouge1']}")
    print(f"ROUGE-2: {fine_tuned_rouge['rouge2'] - base_rouge['rouge2']}")
    print(f"ROUGE-L: {fine_tuned_rouge['rougeL'] - base_rouge['rougeL']}")

    print(f"Total training time: {training_time:.2f} seconds")

    # Generate example summaries
    print("\nGenerating example summaries...")
    for i in range(3):
        article = val_data[i]["article"]
        reference = val_data[i]["highlights"]
        generated = generate_summary(model, article, tokenizer, device)
        print(f"\nArticle {i+1}:")
        print(f"Reference: {reference}")
        print(f"Generated: {generated}")
        print("-" * 50)

if __name__ == "__main__":
    main()