import random
import time
import tracemalloc

import torch
from rouge_score import rouge_scorer

from ngpt.config import PRIME_LENGTH, GENERATE_EVERY_EPOCH, device, GENERATE_LENGTH
from ngpt.helpers import (
    decode_tokens,
    base_decoding,
)
from ngpt.validate import val_dataset

memory_usages = []
inference_times = []
tokens_per_second = []
generation_perplexities = []
rouge_scores = []

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
model = None


def prompt_cumulative_inference(prompt, val_dataset, prime):
    global model
    start_time = time.time()
    tracemalloc.start()
    from ngpt.model import model

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


def generate_summary(epoch):
    if (epoch + 1 == 1) or (epoch + 1) % GENERATE_EVERY_EPOCH == 0:
        from ngpt.model import model

        model.eval()
        with torch.no_grad():
            inp = random.choice(val_dataset)[:PRIME_LENGTH]
            prime = decode_tokens(inp)
            prompt = inp[None, ...].to(device)
            prompt_cumulative_inference(prompt, val_dataset, prime)
