import time

from fvcore.nn import FlopCountAnalysis
from rouge_score import rouge_scorer
import random
import re

import tqdm
import tracemalloc

import torch
from torch.optim import Adam
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

# Using lucidrain's nGPT implementation
from nGPT_pytorch import nGPT
from datasets import load_dataset
import matplotlib.pyplot as plt
import logging

# ignore some flops warnings
logging.basicConfig(level=logging.ERROR)

# common constants
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3
PRIME_LENGTH = 128
GENERATE_LENGTH = 512
SEQ_LEN = 512

# gpu specific
USE_AMP = False  # set True if GPU training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
assert not (USE_AMP and not torch.cuda.is_available())
scaler = GradScaler(enabled=USE_AMP)

# ngpt specific
USE_PARAMETRIZE = True


# helpers
def exists(v):
    return v is not None


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# sampling helpers

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1, keepdim=True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim, keepdim=keepdim)


# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)


def base_decoding(
        net,
        prompt: Tensor,
        seq_len: int,
        temperature=1.5,
        min_p=1e-1,
        filter_thres=0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]

        logits = min_p_filter(logits, min_p=min_p)
        sample = gumbel_sample(logits, temperature=temperature, dim=-1)

        out = torch.cat((out, sample), dim=-1)

    return out[..., prompt_seq_len:]


# nGPT char language model

model = nGPT(
    num_tokens=256,
    dim=512,
    depth=8,
    dim_head=128,
    tied_embedding=True,
    add_value_residual=True,
    attn_norm_qk=False,
    manual_norm_weights=not USE_PARAMETRIZE
).to(device)

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
dataset = load_dataset("cnn_dailymail", "3.0.0", split={'train': 'train[:40]', 'validation': 'validation[:40]'})
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
NUM_EPOCHS = 3
BATCHES_PER_EPOCH = len(train_dataset) // BATCH_SIZE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

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
GENERATE_EVERY_EPOCH = 1  # not final value
VALIDATE_EVERY_EPOCH = 1  # not final value
generation_perplexities = []

# start training
train_start_time = time.time()
STEPS_PER_EPOCH = len(train_dataset) // BATCH_SIZE
epoch_iterator = tqdm.tqdm(range(NUM_EPOCHS), mininterval=10.0, desc="training")
for epoch in epoch_iterator:
    epoch_start_time = time.time()
    running_loss = 0.0
    batch_iterator = tqdm.tqdm(enumerate(train_loader), total=STEPS_PER_EPOCH, desc=f"Epoch {epoch + 1}", leave=False)
    for batch_idx, data in batch_iterator:
        model.train()
        for _ in range(GRAD_ACCUM_EVERY):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                loss = model(data, return_loss=True)

            scaler.scale(loss / GRAD_ACCUM_EVERY).backward()
            running_loss += loss.item()

        scaler.step(optim)
        scaler.update()

        optim.zero_grad()
        train_losses_per_batch.append(loss.item())
        print(f"training loss: {loss.item():.3f}")

        tokens_seen_so_far += data.numel()

        # validation state
        if (epoch + 1) % VALIDATE_EVERY_EPOCH == 0 and batch_idx + 1 == STEPS_PER_EPOCH:
            validation_epochs.append(epoch + 1)
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)

                loss = model(valid_data, return_loss=True)
                val_losses.append(loss.item())
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
                training_tokens.append(tokens_seen_so_far)
                print(f"validation loss: {loss.item():.3f}, perplexity: {perplexity.item():.3f}")

        # Generate state
        if (epoch + 1) % GENERATE_EVERY_EPOCH == 0 and batch_idx + 1 == STEPS_PER_EPOCH:
            generation_epochs.append(epoch + 1)
            model.eval()
            with torch.no_grad():
                inp = random.choice(val_dataset)[:PRIME_LENGTH]
                prime = decode_tokens(inp)

                prompt = inp[None, ...].to(device)  # ensure prompt is on device
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
                print(f"Peak Memory Usage: {peak / 10 ** 6:.2f} MB")
        if batch_idx + 1 >= STEPS_PER_EPOCH:  # To ensure only one epoch of data is processed
            break

    epoch_loss = running_loss / (STEPS_PER_EPOCH * GRAD_ACCUM_EVERY)
    train_losses_per_epoch.append(epoch_loss)  # Append average training loss for the epoch
    print(f"Epoch {epoch + 1}, Average Loss: {epoch_loss:.4f}")
    epoch_end_time = time.time()
    time_per_epoch = (epoch_end_time - epoch_start_time) / (epoch + 1)  # add 1 to i to avoid division by zero error
    print(f"Average Time per Epoch: {time_per_epoch:.4f} seconds")

train_end_time = time.time()
time_training = train_end_time - train_start_time
print(f"Training Time: {time_training:.3f} seconds")

plt.figure(figsize=(10, 8))

# Loss, perplexity graphs
# Batch Training Loss vs. Epochs
plt.subplot(2, 2, 1)
plt.plot(range(len(train_losses_per_batch)),
         train_losses_per_batch)
plt.title("Training Loss per Batch/Step")
plt.xlabel("Training Steps / Batches")
plt.ylabel("Loss")

# Validation Loss vs. Epochs
plt.subplot(2, 2, 2)
plt.plot(validation_epochs, val_losses)
plt.title("Validation Loss")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Loss")

# Training and Validation Loss vs. Epochs (combined)
plt.subplot(2, 2, 3)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses_per_epoch, label='Average Train Loss')
plt.plot(validation_epochs, val_losses, label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Loss")
plt.legend()

# Perplexity vs. Epochs
plt.subplot(2, 2, 4)
plt.plot(validation_epochs, perplexities)
plt.title("Validation Perplexity")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Perplexity")

plt.tight_layout()
plt.show()

# Inference plots
plt.figure(figsize=(10, 6))

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

plt.show()

# Generation perplexity
plt.figure(figsize=(8, 6))
plt.plot(generation_epochs, generation_perplexities)
plt.title("Generation epochs vs Generation Perplexity")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Generation Perplexity")

# Loss vs training tokens like in the paper
plt.figure(figsize=(8, 6))
plt.plot(training_tokens, val_losses)
plt.title("Validation Loss vs. Training Tokens")
plt.xlabel("Training Tokens")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.show()
