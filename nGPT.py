import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

"""
__author__ = "sanjiv joshi"
__email__ = "joshi.sanj@northeastern.edu"
__version__ = "base+generate"
To contribute, fork this file, make suggestions (comment under the code with proposed code),
black format, commit, pull.
"""


class nGPT(nn.Module):
    """
    Normalized GPT model that performs representation learning on hypersphere.
    All vectors (embeddings, MLP, attention matrices, hidden states) are unit norm normalized.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model embeddings (normalized to unit norm)
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_mlp: Dimension of MLP hidden layer
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_mlp):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [nGPTLayer(d_model, n_heads, d_mlp) for _ in range(n_layers)]
        )
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        self.s_z = nn.Parameter(torch.ones(vocab_size))
        self.s_z_init = 1
        self.s_z_scale = 1 / math.sqrt(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        A5: Experimental setup: Initialize weights with normal distribution
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Equation 3: Forward pass normalizing embeddings and hidden states throughout.
        Also scale.

        Args:
            x: Input token ids [batch_size, seq_len]
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        token_emb = self.token_embedding(x)
        x = token_emb
        x = F.normalize(x, dim=-1)

        for layer in self.layers:
            x = layer(x)

        logits = self.output(x)
        logits = logits * (self.s_z * self.s_z_init / self.s_z_scale)
        return logits

    def normalize_embeddings(self):
        """
        Table 1, last point: Normalize embedding matrices along embedding dimension
        """
        with torch.no_grad():
            self.token_embedding.weight.data = F.normalize(
                self.token_embedding.weight.data, dim=-1
            )
            self.output.weight.data = F.normalize(self.output.weight.data, dim=0)

    def normalize_all_weights(self):
        """
        Done after each epoch
        """
        self.normalize_embeddings()
        for layer in self.layers:
            layer.normalize_weights()

    def generate(
        self,
        input_ids,
        max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_beams=4,
        length_penalty=0.8,
        no_repeat_ngram_size=3,
        early_stopping=True,
    ):
        """
        Generation function with beam search
        """
        self.eval()
        with torch.no_grad():
            if num_beams > 1 and not do_sample:
                batch_size = input_ids.shape[0]
                beam_scores = torch.zeros(
                    (batch_size, num_beams), device=input_ids.device
                )
                beam_tokens = input_ids.repeat_interleave(num_beams, dim=0)
                done_sequences = []

                for _ in range(max_length - input_ids.size(1)):
                    outputs = self(beam_tokens)
                    next_token_logits = outputs[:, -1, :] / temperature

                    if length_penalty != 1.0:
                        next_token_logits = next_token_logits / (
                            len(done_sequences) ** length_penalty
                        )

                    if no_repeat_ngram_size > 0:
                        for i in range(batch_size * num_beams):
                            ngram_blacklist = set()
                            for j in range(
                                beam_tokens.size(1) - no_repeat_ngram_size + 1
                            ):
                                ngram = tuple(
                                    beam_tokens[
                                        i, j : j + no_repeat_ngram_size
                                    ].tolist()
                                )
                                ngram_blacklist.add(ngram)
                            for ngram in ngram_blacklist:
                                next_token_logits[i, ngram[-1]] = float("-inf")

                    next_scores = F.log_softmax(
                        next_token_logits, dim=-1
                    ) + beam_scores.unsqueeze(-1)
                    next_scores = next_scores.view(batch_size, -1)

                    next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=1)
                    beam_scores = next_scores

                    beam_tokens = torch.cat(
                        [beam_tokens, next_tokens.unsqueeze(-1)], dim=-1
                    )

                    if (
                        early_stopping
                        and (next_tokens == self.config.eos_token_id).any()
                    ):
                        done_sequences.append(beam_tokens[0])
                        if len(done_sequences) >= num_beams:
                            break

                return (
                    beam_tokens[:1]
                    if not done_sequences
                    else done_sequences[0].unsqueeze(0)
                )
            else:
                """
                Standard sampling, I initially had only this.
                """
                for _ in range(max_length - input_ids.size(1)):
                    outputs = self(input_ids)
                    next_token_logits = outputs[:, -1, :] / temperature

                    if do_sample:
                        next_token_logits = top_k_top_p_filtering(
                            next_token_logits, top_k=top_k, top_p=top_p
                        )
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                            -1
                        )

                    input_ids = torch.cat([input_ids, next_token], dim=-1)

                    if (next_token == self.config.eos_token_id).all():
                        break

                return input_ids


class nGPTLayer(nn.Module):
    """
    Normalized transformer layer with attention and MLP blocks.
    Updates are controlled by learnable eigen learning rates.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_mlp: MLP hidden dimension
    """

    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.attention = nGPTAttention(d_model, n_heads)
        self.mlp = nGPTMLP(d_model, d_mlp)

        self.alpha_A = nn.Parameter(torch.full((d_model,), 0.05))
        self.alpha_M = nn.Parameter(torch.full((d_model,), 0.05))
        self.alpha_A_init = 0.05
        self.alpha_A_scale = 1 / math.sqrt(d_model)
        self.alpha_M_init = 0.05
        self.alpha_M_scale = 1 / math.sqrt(d_model)

    def forward(self, x):
        """
        Equations 10 and 11:
        Forward pass with normalized updates using eigen learning rates
        """
        h_A = self.attention(x)
        x = F.normalize(
            x + (self.alpha_A * self.alpha_A_init / self.alpha_A_scale) * (h_A - x),
            dim=-1,
        )

        h_M = self.mlp(x)
        x = F.normalize(
            x + (self.alpha_M * self.alpha_M_init / self.alpha_M_scale) * (h_M - x),
            dim=-1,
        )
        return x

    def normalize_weights(self):
        self.attention.normalize_weights()
        self.mlp.normalize_weights()


class nGPTAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        Normalized multi-head attention mechanism where all vectors are unit normalized.
        Query-key dot products represent cosine similarities bounded in [-1,1]. I didn't figure out
        how to modify the number of hyperspheres and I am not sure if it helps in anyway.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.s_qk = nn.Parameter(torch.ones(self.n_heads, self.d_head))
        self.s_qk_init = 1
        self.s_qk_scale = 1 / math.sqrt(d_model)
        self.rotary_embedding = RotaryEmbedding(self.d_head)

    def forward(self, x):
        """
        Forward pass with normalized attention computations

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            y: Output tensor [batch_size, seq_len, d_model]
        """
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = self.rotary_embedding.rotate_queries_or_keys(q)
        k = self.rotary_embedding.rotate_queries_or_keys(k)

        q = F.normalize(q, dim=-1) * (
            self.s_qk * self.s_qk_init / self.s_qk_scale
        ).unsqueeze(1)
        k = F.normalize(k, dim=-1) * (
            self.s_qk * self.s_qk_init / self.s_qk_scale
        ).unsqueeze(1)

        att = (q @ k.transpose(-2, -1)) * math.sqrt(self.d_head)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.W_o(y)
        return y

    def normalize_weights(self):
        with torch.no_grad():
            self.W_q.weight.data = F.normalize(self.W_q.weight.data, dim=0)
            self.W_k.weight.data = F.normalize(self.W_k.weight.data, dim=0)
            self.W_v.weight.data = F.normalize(self.W_v.weight.data, dim=0)
            self.W_o.weight.data = F.normalize(self.W_o.weight.data, dim=1)


class nGPTMLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        """
        Normalized MLP block with SwiGLU activation.
        All weight matrices are normalized along embedding dimension.

        Args:
            d_model: Model dimension
            d_mlp: Hidden dimension of MLP
        """
        super().__init__()
        self.W_u = nn.Linear(d_model, d_mlp, bias=False)
        self.W_v = nn.Linear(d_model, d_mlp, bias=False)
        self.W_o = nn.Linear(d_mlp, d_model, bias=False)

        self.s_u = nn.Parameter(torch.ones(d_mlp))
        self.s_v = nn.Parameter(torch.ones(d_mlp))
        self.s_u_init = 1
        self.s_u_scale = 1
        self.s_v_init = 1
        self.s_v_scale = 1

    def forward(self, x):
        """
        Forward pass with SwiGLU activation and normalized weights

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            x: Output tensor [batch_size, seq_len, d_model]
        """

        """
        Equations 18 and 19:
        SwiGLU.
        """
        u = self.W_u(x) * (self.s_u * self.s_u_init / self.s_u_scale)
        v = (
            self.W_v(x)
            * (self.s_v * self.s_v_init / self.s_v_scale)
            * math.sqrt(self.W_v.in_features)
        )
        x = u * F.silu(v)
        x = self.W_o(x)
        return x

    def normalize_weights(self):
        with torch.no_grad():
            self.W_u.weight.data = F.normalize(self.W_u.weight.data, dim=0)
            self.W_v.weight.data = F.normalize(self.W_v.weight.data, dim=0)
            self.W_o.weight.data = F.normalize(self.W_o.weight.data, dim=1)


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    Yoinked this code from huggingface transformers and adapted to this model. 
    Original function: 
    huggingface/transformers@3b00b623b7cad9e1b7c71c97fff24a0286b37045
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits
