import logging
import os
import pickle
import re

import matplotlib.pyplot as plt
import tqdm
from datasets import load_dataset
from fvcore.nn import FlopCountAnalysis
from torch._C._nn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

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
example_input = (
    torch.randint(0, 256, (BATCH_SIZE, SEQ_LEN)).long().to(device)
)  # Create example input tensor
flops = FlopCountAnalysis(model, example_input)
total_flops = flops.total()
print(f"Total FLOPs: {total_flops}")
# 68979554513.0


def construct_metrics():
    metrics = {
        # losses
        "train_loss_per_epoch": train_loss_per_epoch,  # NUM_EPOCHS
        "val_loss_per_epoch": val_loss_per_epoch,  # validation_epochs
        # "validation_epochs": validation_epochs,
        # perplexities
        "perplexities": perplexities,  # validation_epochs
        "generation_perplexities": generation_perplexities,  # generation_epochs
        # inference and memory
        "tokens_per_second": tokens_per_second,  # generation_epochs
        "memory_usages": memory_usages,  # generation_epochs
        "inference_times": inference_times,  # generation_epochs
        "rouge_scores": rouge_scores,
        "training_tokens_4k": training_tokens_4k,
        "training_tokens_1k": training_tokens_1k,
        "training_tokens_8k": training_tokens_8k,
        "val_losses_1k": val_losses_1k,
        "val_losses_8k": val_losses_8k,
        "val_losses_4k": val_losses_4k,
    }
    with open("metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)


# using a very basic tokenizer
def tokenize(examples):
    tokenized_text = []
    for example in examples["article"]:
        # ignore non ascii
        ascii_text = re.sub(r"[^\x00-\x7F]+", " ", example)
        tokenized_text.append([ord(char) for char in ascii_text])
    return {"text": tokenized_text}


# data preparation
dataset = load_dataset(
    "cnn_dailymail",
    "3.0.0",
    split={"train": "train[:100]", "validation": "validation[:10]"},
)
# dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset = dataset.map(
    tokenize, batched=True, num_proc=1, remove_columns=["article", "highlights", "id"]
)
dataset.set_format(type="torch")
# split_index = int(0.8 * len(dataset["train"]))
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
# train_dataset = dataset["train"].select(range(split_index))
# val_dataset = dataset["train"].select(range(split_index, len(dataset["train"])))


# make dataset ngpt compatible to read
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]["text"]

        # Calculate maximum possible sequence length for this text
        max_seq_len = min(self.seq_len, len(text) - 1)

        if max_seq_len < 1:
            # Text is too short, pad the entire sequence
            padding = torch.zeros(self.seq_len + 1, dtype=torch.long)
            full_seq = padding.to(device)
        else:
            # Adjust sequence length and get random start position
            rand_start = torch.randint(0, len(text) - max_seq_len, (1,))
            sequence = text[rand_start : rand_start + max_seq_len + 1]

            # Pad if necessary to reach desired sequence length
            if len(sequence) < self.seq_len + 1:
                padding = torch.zeros(
                    self.seq_len + 1 - len(sequence), dtype=torch.long
                )
                full_seq = torch.cat([sequence, padding]).to(device)
            else:
                full_seq = sequence.to(device)

        return full_seq


# prepare datasets
train_dataset = TextSamplerDataset(train_dataset, SEQ_LEN)
val_dataset = TextSamplerDataset(val_dataset, SEQ_LEN)


def collate_fn(batch):
    # Add padding to create tensors of equal length within each batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0).to(
        device
    )  # Pad and move to device
    return padded_batch


train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# optimizer
optim = Adam(model.parameters(), lr=LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# if not using parametrize, register normalizing on optimizer step
if not USE_PARAMETRIZE:
    model.register_step_post_hook(optim)

# training
NUM_EPOCHS = 200
GENERATE_EVERY_EPOCH = 1
VALIDATE_EVERY_EPOCH = 1
BATCHES_PER_EPOCH = len(train_dataset) // BATCH_SIZE
patience = 3

# loss lists
val_loss_per_epoch = []
train_loss_per_epoch = []

# inference stuff
training_tokens = []
tokens_seen_so_far = 0

# perplexities
perplexities = []

# learning_rates = []
metrics = {}

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
EPOCH_MODEL = os.path.join(CHECKPOINT_DIR, "epoch_model.pt")
TRAIN_MODEL = os.path.join(CHECKPOINT_DIR, "train_model.pt")

CONTEXT_LENGTHS = [1024, 4096, 8192]

# start training
train_start_time = time.time()
STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE

epoch_iterator = tqdm.tqdm(range(NUM_EPOCHS), mininterval=10.0, desc="training")
# for epoch in epoch_iterator:
#     epoch_start_time = time.time()
#     batch_iterator = tqdm.tqdm(
#         enumerate(train_loader),
#         total=STEPS_PER_EPOCH,
#         desc=f"Epoch {epoch + 1}",
#         leave=False,
#     )
#     train_loss_per_batch = []
#     val_loss = float("-inf")
#     for batch_idx, data in batch_iterator:
#         if batch_idx >= STEPS_PER_EPOCH:
#             break
#         model.train()
#         data = data.to(device)
#         running_loss = 0.0
#         for _ in range(GRAD_ACCUM_EVERY):
#             with torch.autocast(
#                 device_type="cuda", dtype=torch.float16, enabled=USE_AMP
#             ):
#                 loss = model(data, return_loss=True)
#
#             scaler.scale(loss / GRAD_ACCUM_EVERY).backward()
#             running_loss = loss.item()
#
#         curr_loss = running_loss / GRAD_ACCUM_EVERY
#         train_loss_per_batch.append(curr_loss)
#         print(f"training loss: {curr_loss:.3f}")
#
#         scaler.step(optim)
#         scaler.update()
#         optim.zero_grad()
#
#         tokens_seen_so_far += data.numel()
#
#     # validation state
#     if (epoch + 1) == 1 or (epoch + 1) % VALIDATE_EVERY_EPOCH == 0:
#         model.eval()
#         with torch.no_grad():
#             valid_data = next(val_loader)
#             valid_data = valid_data.to(device)
#             for context_length in CONTEXT_LENGTHS:
#                 valid_data_truncated = valid_data[:, :context_length].to(device)
#                 trunc_loss = model(valid_data_truncated, return_loss=True)
#                 context_helper(context_length, trunc_loss, epoch, tokens_seen_so_far)
#         val_loss = model(valid_data, return_loss=True)
#         val_loss_per_epoch.append(val_loss.item())
#         validation_epochs.append(epoch + 1)
#         perplexity = torch.exp(val_loss).item()
#         perplexities.append(perplexity)
#         training_tokens.append(tokens_seen_so_far)
#         print(f"validation loss: {val_loss:.3f}, " f"perplexity: {perplexity:.3f}")
#         if val_loss < best_val_loss:
#             best_val_loss = loss
#
#     # Generate state
#     if (epoch + 1 == 1) or (epoch + 1) % GENERATE_EVERY_EPOCH == 0:
#         model.eval()
#         with torch.no_grad():
#             inp = random.choice(val_dataset)[:PRIME_LENGTH]
#             prime = decode_tokens(inp)
#             prompt = inp[None, ...].to(device)
#             prompt_cumulative_inference(prompt, val_dataset, prime)
#     epoch_loss = sum(train_loss_per_batch) / STEPS_PER_EPOCH
#     train_loss_per_epoch.append(
#         epoch_loss
#     )  # Append average training loss for the epoch
#     print(f"Epoch {epoch + 1}, Average Loss over the epoch: {epoch_loss:.4f}")
#     epoch_end_time = time.time()
#     time_per_epoch = (epoch_end_time - epoch_start_time) / (epoch + 1)
#     print(f"Time per Epoch: {time_per_epoch:.4f} seconds")
#     construct_metrics()
#     if val_loss >= epoch_loss:
#         if patience == 0:
#             print(f"Model has converged, saving metrics and model")
#             torch.save(model.state_dict(), BEST_MODEL_PATH)
#             break
#         else:
#             patience -= 1
#     print(f"Saving model at epoch {epoch + 1}")
#     torch.save(model.state_dict(), EPOCH_MODEL)

# Initialize variables for early stopping
best_val_loss = float("inf")  # Best validation loss starts as infinity
patience_counter = 0  # Counter for early stopping

for epoch in epoch_iterator:
    epoch_start_time = time.time()
    batch_iterator = tqdm.tqdm(
        enumerate(train_loader),
        total=STEPS_PER_EPOCH,
        desc=f"Epoch {epoch + 1}",
        leave=False,
    )
    train_loss_per_batch = []
    val_loss = float("-inf")

    # Training loop (unchanged)
    running_loss = 0.0
    for batch_idx, data in batch_iterator:
        if batch_idx >= STEPS_PER_EPOCH:
            break

        model.train()
        data = data.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            loss = model(data, return_loss=True)

        loss = loss / GRAD_ACCUM_EVERY
        loss.backward()

        running_loss += loss.item()

        if (batch_idx + 1) % GRAD_ACCUM_EVERY == 0 or (
            batch_idx + 1
        ) == STEPS_PER_EPOCH:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            curr_loss = running_loss / GRAD_ACCUM_EVERY
            train_loss_per_batch.append(curr_loss)
            print(f"Training loss: {curr_loss:.3f}")

            running_loss = 0.0

        tokens_seen_so_far += data.numel()

    # Validation step
    if (epoch + 1) == 1 or (epoch + 1) % VALIDATE_EVERY_EPOCH == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            valid_data = valid_data.to(device)
            for context_length in CONTEXT_LENGTHS:
                valid_data_truncated = valid_data[:, :context_length].to(device)
                trunc_loss = model(valid_data_truncated, return_loss=True)
                context_helper(context_length, trunc_loss, epoch, tokens_seen_so_far)

        val_loss = model(valid_data, return_loss=True)
        val_loss_per_epoch.append(val_loss.item())
        perplexity = torch.exp(val_loss).item()
        perplexities.append(perplexity)
        training_tokens.append(tokens_seen_so_far)
        print(f"Validation loss: {val_loss:.3f}, Perplexity: {perplexity:.3f}")

        # Early stopping logic
        if val_loss < best_val_loss - 0.01:
            best_val_loss = val_loss.item()
            patience_counter = 0
            print(f"New best validation loss: {best_val_loss:.3f}. Saving model...")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}"
            )

        # Stop training if patience is exhausted
        if patience_counter >= patience:
            print("Early stopping triggered. Training has converged.")
            break

    # Generate step (unchanged)
    if (epoch + 1 == 1) or (epoch + 1) % GENERATE_EVERY_EPOCH == 0:
        model.eval()
        with torch.no_grad():
            inp = random.choice(val_dataset)[:PRIME_LENGTH]
            prime = decode_tokens(inp)
            prompt = inp[None, ...].to(device)
            prompt_cumulative_inference(prompt, val_dataset, prime)

    # Calculate and log average training loss for the epoch
    epoch_loss = sum(train_loss_per_batch) / len(train_loss_per_batch)
    train_loss_per_epoch.append(epoch_loss)
    print(f"Epoch {epoch + 1}, Average Loss over the epoch: {epoch_loss:.4f}")

    epoch_end_time = time.time()
    time_per_epoch = (epoch_end_time - epoch_start_time) / (epoch + 1)
    print(f"Time per Epoch: {time_per_epoch:.4f} seconds")

    construct_metrics()

    # Save checkpoint at each epoch
    print(f"Saving model at epoch {epoch + 1}")
    torch.save(model.state_dict(), EPOCH_MODEL)

# for epoch in epoch_iterator:
#     epoch_start_time = time.time()
#     batch_iterator = tqdm.tqdm(
#         enumerate(train_loader),
#         total=STEPS_PER_EPOCH,
#         desc=f"Epoch {epoch + 1}",
#         leave=False,
#     )
#     train_loss_per_batch = []
#     val_loss = float("-inf")
#
#     # Reset running loss for the epoch
#     running_loss = 0.0
#
#     for batch_idx, data in batch_iterator:
#         if batch_idx >= STEPS_PER_EPOCH:
#             break
#
#         model.train()
#         data = data.to(device)
#
#         # Forward pass with mixed precision (if enabled)
#         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
#             loss = model(data, return_loss=True)
#
#         # Normalize loss by accumulation steps and backpropagate
#         loss = loss / GRAD_ACCUM_EVERY
#         loss.backward()
#
#         # Accumulate running loss for logging
#         running_loss += loss.item()
#
#         # Perform optimizer step and reset gradients after every GRAD_ACCUM_EVERY steps
#         if (batch_idx + 1) % GRAD_ACCUM_EVERY == 0 or (
#             batch_idx + 1
#         ) == STEPS_PER_EPOCH:
#             scaler.step(optim)  # Apply scaled gradients
#             scaler.update()  # Update scaler for mixed precision
#             optim.zero_grad()  # Reset gradients
#
#             # Log the average loss over the accumulated steps
#             curr_loss = running_loss / GRAD_ACCUM_EVERY
#             train_loss_per_batch.append(curr_loss)
#             print(f"Training loss: {curr_loss:.3f}")
#
#             # Reset running loss for the next accumulation cycle
#             running_loss = 0.0
#
#         tokens_seen_so_far += data.numel()
#
#     # Validation step
#     if (epoch + 1) == 1 or (epoch + 1) % VALIDATE_EVERY_EPOCH == 0:
#         model.eval()
#         with torch.no_grad():
#             valid_data = next(val_loader)
#             valid_data = valid_data.to(device)
#             for context_length in CONTEXT_LENGTHS:
#                 valid_data_truncated = valid_data[:, :context_length].to(device)
#                 trunc_loss = model(valid_data_truncated, return_loss=True)
#                 context_helper(context_length, trunc_loss, epoch, tokens_seen_so_far)
#         val_loss = model(valid_data, return_loss=True)
#         val_loss_per_epoch.append(val_loss.item())
#         validation_epochs.append(epoch + 1)
#         perplexity = torch.exp(val_loss).item()
#         perplexities.append(perplexity)
#         training_tokens.append(tokens_seen_so_far)
#         print(f"Validation loss: {val_loss:.3f}, Perplexity: {perplexity:.3f}")
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#
#     # Generate step
#     if (epoch + 1 == 1) or (epoch + 1) % GENERATE_EVERY_EPOCH == 0:
#         model.eval()
#         with torch.no_grad():
#             inp = random.choice(val_dataset)[:PRIME_LENGTH]
#             prime = decode_tokens(inp)
#             prompt = inp[None, ...].to(device)
#             prompt_cumulative_inference(prompt, val_dataset, prime)
#
#     # Calculate and log average training loss for the epoch
#     epoch_loss = sum(train_loss_per_batch) / len(train_loss_per_batch)
#     train_loss_per_epoch.append(
#         epoch_loss
#     )  # Append average training loss for the epoch
#     print(f"Epoch {epoch + 1}, Average Loss over the epoch: {epoch_loss:.4f}")
#
#     epoch_end_time = time.time()
#     time_per_epoch = (epoch_end_time - epoch_start_time) / (epoch + 1)
#     print(f"Time per Epoch: {time_per_epoch:.4f} seconds")
#
#     construct_metrics()
#
#     # Early stopping or saving best model logic
#     if val_loss >= epoch_loss:
#         if patience == 0:
#             print(f"Model has converged, saving metrics and model")
#             torch.save(model.state_dict(), BEST_MODEL_PATH)
#             break
#         else:
#             patience -= 1
#
#     print(f"Saving model at epoch {epoch + 1}")
#     torch.save(model.state_dict(), EPOCH_MODEL)

construct_metrics()
torch.save(model.state_dict(), TRAIN_MODEL)
train_end_time = time.time()
time_training = train_end_time - train_start_time
print(f"Training Time: {time_training:.3f} seconds")

# model.load_state_dict(torch.load(BEST_MODEL_PATH))
# print(f"Loaded best model from {BEST_MODEL_PATH}")

# plt.figure(figsize=(10, 8))
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss_per_epoch) + 1), train_loss_per_epoch)
plt.title("Training Loss vs Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.savefig("train_loss_vs_epoch.png")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_loss_per_epoch) + 1), val_loss_per_epoch)
plt.title("Validation Loss vs Epochs")
plt.xlabel("Validation Epochs")
plt.ylabel("Validation Loss")
plt.savefig("validation_loss_vs_epoch.png")

plt.plot(
    range(1, len(train_loss_per_epoch) + 1),
    train_loss_per_epoch,
    color="blue",
    label="Training",
)
plt.plot(
    range(1, len(val_loss_per_epoch) + 1),
    val_loss_per_epoch,
    color="red",
    label="Validation",
)
plt.title("Training and Validation Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("train_loss_validation_loss_vs_epoch.png")
# plt.show()

# Perplexity vs. Epochs
# plt.subplot(2, 2, 4)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_loss_per_epoch) + 1), perplexities)
plt.title("Validation Perplexity (Combined)")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Perplexity")
plt.tight_layout()
plt.savefig("loss_vs_perplexity.png")
# plt.show()

# Inference plots
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), inference_times)
plt.title("Inference Time")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Time (seconds)")
plt.savefig("inference_time.png")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), tokens_per_second)
plt.title("Tokens per Second")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Tokens/sec")
plt.savefig("tokens_per_second.png")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), memory_usages)
plt.title("Peak Memory Usage")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Memory Usage (MB)")
plt.savefig("inference_graphs.png")

# Generation perplexity
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), generation_perplexities)
plt.title("Generation epochs vs Generation Perplexity")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Generation Perplexity")
plt.savefig("generation_perplexities.png")
# plt.show()

# Loss vs training tokens like in the paper
plt.figure(figsize=(8, 6))
plt.plot(training_tokens, val_loss_per_epoch)
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
