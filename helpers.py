# helpers
import torch
from torch import Tensor


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
