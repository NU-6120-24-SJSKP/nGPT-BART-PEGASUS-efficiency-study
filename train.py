import os
import signal
import time

from fvcore.nn import FlopCountAnalysis
from rouge_score import rouge_scorer
import random
import re

import tqdm
import tracemalloc

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
import matplotlib.pyplot as plt
import logging

from config import *

# ignore some flops warnings
logging.basicConfig(level=logging.ERROR)

# gpu specific
assert not (USE_AMP and not torch.cuda.is_available())

# sampling helpers
from helpers import *

# nGPT char language model
from model import model

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
# 33719952

# sample flops
example_input = torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN)).long().to(device)  # Create example input tensor
flops = FlopCountAnalysis(model, example_input)
total_flops = flops.total()
print(f"Total FLOPs: {total_flops}")


# 68979554513.0

# using a very basic tokenizer
def tokenize(examples):
    tokenized_text = []
    for example in examples['article']:
        # ignore non ascii
        ascii_text = re.sub(r'[^\x00-\x7F]+', ' ', example)
        tokenized_text.append([ord(char) for char in ascii_text])
    return {"text": tokenized_text}


# data preparation
dataset = load_dataset("cnn_dailymail", "3.0.0", split={'train': 'train[:5000]', 'validation': 'validation[:4500]'})
dataset = dataset.map(tokenize, batched=True, num_proc=4,
                      remove_columns=['article', 'highlights', 'id'])
dataset.set_format(type='torch')

train_dataset = dataset['train']
val_dataset = dataset['validation']


# make dataset ngpt compatible to read
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['text']
        text = text.to(device)

        if len(text) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - len(text), dtype=torch.long).to(device)
            full_seq = torch.cat([text, padding])
        else:
            rand_start = torch.randint(0, len(text) - self.seq_len - 1, (1,))
            full_seq = text[rand_start: rand_start + self.seq_len + 1].long()

        return full_seq


# prepare datasets
train_dataset = TextSamplerDataset(train_dataset, SEQ_LEN)
val_dataset = TextSamplerDataset(val_dataset, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# optimizer
optim = Adam(model.parameters(), lr=LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# if not using parametrize, register normalizing on optimizer step
if not USE_PARAMETRIZE:
    model.register_step_post_hook(optim)

# training
NUM_EPOCHS = 20
BATCHES_PER_EPOCH = len(train_dataset) // BATCH_SIZE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
best_val_loss = float('inf')
patience = 3
epochs_without_improvement = 0

train_losses = []
val_losses = []
validation_epochs = []
generation_epochs = []
perplexities = []
inference_times = []
tokens_per_second = []
memory_usages = []
training_tokens = []
tokens_seen_so_far = 0
train_losses_per_epoch = []
train_losses_per_batch = []
GENERATE_EVERY_EPOCH = 4  # not final value
VALIDATE_EVERY_EPOCH = 4  # not final value
generation_perplexities = []
# contexts
validation_epochs_1k = []
validation_epochs_4k = []
validation_epochs_8k = []
val_losses_1k = []
val_losses_4k = []
val_losses_8k = []
training_tokens_1k = []
training_tokens_4k = []
training_tokens_8k = []
# learning_rates = []

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    print(f"Best model saved at {BEST_MODEL_PATH}")
    exit(0)


signal.signal(signal.SIGINT, signal_handler)

CONTEXT_LENGTHS = [1024, 4096, 8192]

# start training
train_start_time = time.time()
STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE
# scheduler = CosineAnnealingLR(optim, T_max=NUM_EPOCHS * STEPS_PER_EPOCH, eta_min=0)
epoch_iterator = tqdm.tqdm(range(NUM_EPOCHS), mininterval=10.0, desc="training")
for epoch in epoch_iterator:
    epoch_start_time = time.time()
    running_loss = 0.0
    batch_iterator = tqdm.tqdm(enumerate(train_loader), total=STEPS_PER_EPOCH, desc=f"Epoch {epoch + 1}", leave=False)
    for batch_idx, data in batch_iterator:
        model.train()
        data = data.to(device)
        for _ in range(GRAD_ACCUM_EVERY):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                loss = model(data, return_loss=True)

            scaler.scale(loss / GRAD_ACCUM_EVERY).backward()
            running_loss += loss.item()

        train_losses_per_batch.append(loss.item())
        print(f"training loss: {loss.item():.3f}")

        scaler.step(optim)
        # scheduler.step()  # Update learning rate *after* optimizer step
        # current_lr = scheduler.get_last_lr()[0]  # Get current learning rate
        # learning_rates.append(current_lr)

        scaler.update()
        optim.zero_grad()

        tokens_seen_so_far += data.numel()

        # validation state
        if (epoch + 1) % VALIDATE_EVERY_EPOCH == 0 and batch_idx + 1 == STEPS_PER_EPOCH:
            validation_epochs.append(epoch + 1)
            model.eval()
            with torch.no_grad():
                for context_length in CONTEXT_LENGTHS:
                    valid_data = next(val_loader)
                    valid_data = valid_data.to(device)
                    valid_data_truncated = valid_data[:, :context_length].to(device)
                    loss = model(valid_data_truncated, return_loss=True)
                    perplexity = torch.exp(loss)
                    if context_length == 1024:
                        val_losses_1k.append(loss.item())
                        validation_epochs_1k.append(epoch + (batch_idx + 1) / STEPS_PER_EPOCH)
                        training_tokens_1k.append(tokens_seen_so_far)
                    elif context_length == 4096:
                        val_losses_4k.append(loss.item())
                        validation_epochs_4k.append(epoch + (batch_idx + 1) / STEPS_PER_EPOCH)
                        training_tokens_4k.append(tokens_seen_so_far)
                    elif context_length == 8192:
                        val_losses_8k.append(loss.item())
                        validation_epochs_8k.append(epoch + (batch_idx + 1) / STEPS_PER_EPOCH)
                        training_tokens_8k.append(tokens_seen_so_far)
                perplexities.append(perplexity.item())
                training_tokens.append(tokens_seen_so_far)
                print(f"validation loss: {loss.item():.3f}, perplexity: {perplexity.item():.3f}")
                val_losses.append(loss.item())
                if loss < best_val_loss:
                    best_val_loss = loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

        # Generate state
        if (epoch + 1) % GENERATE_EVERY_EPOCH == 0 and batch_idx + 1 == STEPS_PER_EPOCH:
            generation_epochs.append(epoch + 1)
            model.eval()
            with torch.no_grad():
                inp = random.choice(val_dataset)[:PRIME_LENGTH]
                prime = decode_tokens(inp)

                prompt = inp[None, ...].to(device)
                start_time = time.time()
                tracemalloc.start()
                sampled = base_decoding(model, prompt, GENERATE_LENGTH)
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_usages.append(peak / 10 ** 6)
                generated_text = decode_tokens(sampled[0])

                inference_time = end_time - start_time
                inference_times.append(inference_time)

                num_tokens_generated = len(generated_text)
                tokens_per_sec = num_tokens_generated / inference_time
                tokens_per_second.append(tokens_per_sec)

                # Get a random target summary from the validation set for ROUGE calculation
                random_val_index = random.randint(0, len(val_dataset) - 1)
                target_summary = decode_tokens(val_dataset[random_val_index])  # Decode target

                scores = scorer.score(target_summary, generated_text)

                generation_loss = model(sampled, return_loss=True)
                generation_perplexity = torch.exp(generation_loss)
                generation_perplexities.append(generation_perplexity.item())
                print(f"Generation Perplexity: {generation_perplexity.item():.3f}\n")

                print(f"Prime (Input):\n{prime} \n\n {'*' * 100}\n")
                print(f"Generated Continuation:\n{generated_text}\n{'*' * 100}\n")
                print(f"ROUGE Scores:\n{scores}\n")
                print(f"Inference Time: {inference_time:.3f} seconds")
                print(f"Tokens per Second: {tokens_per_sec:.3f}")
                print(f"Peak Memory Usage: {peak / 10 ** 6:.6f} MB")

        if batch_idx + 1 >= STEPS_PER_EPOCH:  # To ensure only one epoch of data is processed
            break

    epoch_loss = running_loss / (STEPS_PER_EPOCH * GRAD_ACCUM_EVERY)
    train_losses_per_epoch.append(epoch_loss)  # Append average training loss for the epoch
    print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss:.4f}")
    epoch_end_time = time.time()
    time_per_epoch = (epoch_end_time - epoch_start_time) / (epoch + 1)
    print(f"Average Time per Epoch: {time_per_epoch:.4f} seconds")

train_end_time = time.time()
time_training = train_end_time - train_start_time
print(f"Training Time: {time_training:.3f} seconds")

model.load_state_dict(torch.load(BEST_MODEL_PATH))
print(f"Loaded best model from {BEST_MODEL_PATH}")

# plt.figure(figsize=(10, 8))

# Loss, perplexity graphs
# Batch Training Loss vs. Epochs
plt.figure(figsize=(8, 6))
# plt.subplot(2, 2, 1)
plt.plot(range(len(train_losses_per_batch)),
         train_losses_per_batch)
plt.title("Training Loss per Batch/Step")
plt.xlabel("Training Steps / Batches")
plt.ylabel("Loss")
# plt.show()
plt.savefig("train_loss_vs_batch.png")

# Validation Loss vs. Epochs
# plt.subplot(2, 2, 2)
plt.figure(figsize=(8, 6))
plt.plot(validation_epochs, val_losses)
plt.title("Validation Loss")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Loss")
plt.savefig("validation_loss_vs_epoch.png")

# Training and Validation Loss vs. Epochs (combined)
# plt.subplot(2, 2, 3)
from scipy.interpolate import interp1d

train_loss_at_validation_epochs = interp1d(range(1, NUM_EPOCHS + 1), train_losses_per_epoch, kind='linear')(
    validation_epochs)

plt.plot(validation_epochs, train_loss_at_validation_epochs, label='Average Train Loss')  # Interpolated training loss
plt.plot(validation_epochs, val_losses, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs (Validation Steps)")  # The unit would be epoch fractions now
plt.ylabel("Loss")
plt.legend()
plt.savefig("train_loss_validation_loss_vs_epoch.png")
# plt.show()

# Perplexity vs. Epochs
# plt.subplot(2, 2, 4)
plt.figure(figsize=(8, 6))
plt.plot(validation_epochs, perplexities)
plt.title("Validation Perplexity (Combined)")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Perplexity")
plt.tight_layout()
plt.savefig("loss_vs_perplexity.png")
# plt.show()

# Inference plots
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.plot(generation_epochs, inference_times)
plt.title("Inference Time")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Time (seconds)")

plt.subplot(1, 3, 2)
plt.plot(generation_epochs, tokens_per_second)
plt.title("Tokens per Second")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Tokens/sec")

plt.subplot(1, 3, 3)
plt.plot(generation_epochs, memory_usages)
plt.title("Peak Memory Usage")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Memory Usage (MB)")
plt.savefig("inference_graphs.png")

# plt.show()

# Generation perplexity
plt.figure(figsize=(8, 6))
plt.plot(generation_epochs, generation_perplexities)
plt.title("Generation epochs vs Generation Perplexity")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Generation Perplexity")
plt.savefig("generation_perplexities.png")
# plt.show()

# Loss vs training tokens like in the paper
plt.figure(figsize=(8, 6))
plt.plot(training_tokens, val_losses)
plt.title("Validation Loss vs. Training Tokens")
plt.xlabel("Training Tokens")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.savefig("tokens_per_loss.png")

# plt.show()

# Plotting
plt.figure(figsize=(12, 8))

# 1k Context vs. Training Tokens
plt.subplot(2, 2, 1)
plt.plot(training_tokens_1k, val_losses_1k)
plt.title("Validation Loss vs. Training Tokens (1k Context)")
plt.xlabel("Training Tokens")
plt.ylabel("Validation Loss")
plt.grid(True)

# 4k Context vs. Training Tokens
plt.subplot(2, 2, 2)
plt.plot(training_tokens_4k, val_losses_4k)
plt.title("Validation Loss vs. Training Tokens (4k Context)")
plt.xlabel("Training Tokens")
plt.ylabel("Validation Loss")
plt.grid(True)

# 8k Context vs. Training Tokens
plt.subplot(2, 2, 3)
plt.plot(training_tokens_8k, val_losses_8k)  # Corrected data
plt.title("Validation Loss vs. Training Tokens (8k Context)")
plt.xlabel("Training Tokens")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_tokens_all.png")
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(range(len(learning_rates)), learning_rates)
# plt.title("Learning Rate Schedule")
# plt.xlabel("Training Steps / Batches")
# plt.ylabel("Learning Rate")
# plt.grid(True)
# plt.savefig("learning_rates.png")
# # plt.show()
