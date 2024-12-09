# helpers
import random
import time
import tracemalloc

import torch
from rouge_score import rouge_scorer
from torch import Tensor

from config import GENERATE_LENGTH
from model import model

validation_epochs_1k = []
validation_epochs_4k = []
validation_epochs_8k = []
val_losses_1k = []
val_losses_4k = []
val_losses_8k = []
training_tokens_1k = []
training_tokens_4k = []
training_tokens_8k = []

memory_usages = []
inference_times = []
tokens_per_second = []
generation_perplexities = []
rouge_scores = []

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


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


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, keepdim=True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(
        dim=dim, keepdim=keepdim
    )


# min_p
# https://arxiv.org/abs/2407.01082


def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float("-inf"), logits)


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


def context_helper(context_length, loss, context_epoch, tokens_seen_so_far):
    if context_length == 1024:
        val_losses_1k.append(loss.item())
        validation_epochs_1k.append(context_epoch)
        training_tokens_1k.append(tokens_seen_so_far)
    elif context_length == 4096:
        val_losses_4k.append(loss.item())
        validation_epochs_4k.append(context_epoch)
        training_tokens_4k.append(tokens_seen_so_far)
    elif context_length == 8192:
        val_losses_8k.append(loss.item())
        validation_epochs_8k.append(context_epoch)
        training_tokens_8k.append(tokens_seen_so_far)


def average_validation_loss():
    return (
        sum(val_losses_1k) / len(val_losses_1k)
        + sum(val_losses_4k) / len(val_losses_4k)
        + sum(val_losses_8k) / len(val_losses_8k)
    )


def prompt_cumulative_inference(prompt, val_dataset, prime):
    start_time = time.time()
    tracemalloc.start()
    sampled = base_decoding(model, prompt, GENERATE_LENGTH)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usages.append(peak / 10**6)
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    generated_text = decode_tokens(sampled[0])
    num_tokens_generated = len(generated_text)
    tokens_per_sec = num_tokens_generated / inference_time
    tokens_per_second.append(tokens_per_sec)

    # Get a random target summary from the validation set for ROUGE calculation
    random_val_index = random.randint(0, len(val_dataset) - 1)
    target_summary = decode_tokens(val_dataset[random_val_index])  # Decode target

    scores = scorer.score(target_summary, generated_text)
    rouge_scores.append(scores)
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


def return_metrics():
    return {}
