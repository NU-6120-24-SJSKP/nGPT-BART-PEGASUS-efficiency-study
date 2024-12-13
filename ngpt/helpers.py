# Helper functions for various operations in the nGPT model

import torch
from torch import Tensor


def exists(v):
    """
    Check if a variable exists (is not None).

    :param v: Variable to check
    :return: True if the variable exists, False otherwise
    """
    return v is not None


def cycle(loader):
    """
    Create an infinite cycle of the data loader.

    :param loader: DataLoader to cycle through
    :yield: Next batch of data from the loader
    """
    while True:
        for data in loader:
            yield data


def decode_token(token):
    """
    Convert a token (ASCII value) to its corresponding character, ensuring it's printable.

    :param token: ASCII value of the token
    :return: Corresponding character as a string
    """
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    """
    Decode a sequence of tokens into a string.

    :param tokens: List or tensor of tokens
    :return: Decoded string
    """
    return "".join(list(map(decode_token, tokens)))


def log(t, eps=1e-20):
    """
    Compute the log of a tensor with a small epsilon to avoid log(0).

    :param t: Input tensor
    :param eps: Small value to clamp the tensor to avoid log(0)
    :return: Log of the tensor
    """
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    """
    Generate Gumbel noise for sampling.

    :param t: Tensor to match the shape of the noise
    :return: Gumbel noise tensor
    """
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, keepdim=True):
    """
    Sample from a categorical distribution using the Gumbel-Softmax trick.

    :param t: Logits tensor
    :param temperature: Temperature for sampling, controls randomness
    :param dim: Dimension along which to sample
    :param keepdim: Whether to keep the dimension after sampling
    :return: Sampled tensor
    """
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(
        dim=dim, keepdim=keepdim
    )


# min_p
# Reference: https://arxiv.org/abs/2407.01082


def min_p_filter(logits, min_p=0.1):
    """
    Filter logits based on a minimum probability threshold.

    :param logits: Logits tensor
    :param min_p: Minimum probability threshold
    :return: Filtered logits tensor
    """
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
    """
    Generate text based on a given prompt using the model.

    :param net: The neural network model
    :param prompt: Initial prompt tensor
    :param seq_len: Desired length of the generated sequence
    :param temperature: Temperature for sampling, controls randomness
    :param min_p: Minimum probability threshold for filtering
    :param filter_thres: Threshold for filtering logits (not used in this function)
    :return: Tensor of generated tokens
    """
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        # Get logits for the next token
        logits = net(out)
        logits = logits[:, -1]

        # Apply min_p filtering to logits
        logits = min_p_filter(logits, min_p=min_p)

        # Sample the next token using Gumbel-Softmax
        sample = gumbel_sample(logits, temperature=temperature, dim=-1)

        # Append the sampled token to the output sequence
        out = torch.cat((out, sample), dim=-1)

    # Return only the generated part of the sequence
    return out[..., prompt_seq_len:]
