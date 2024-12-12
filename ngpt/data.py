import re

import torch
from datasets import load_dataset
from torch._C._nn import pad_sequence
from torch.utils.data import Dataset

from ngpt.config import device


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
    split={"train": "train[:10]", "validation": "validation[:1]"},
)
# dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset = dataset.map(
    tokenize, batched=True, num_proc=1, remove_columns=["article", "highlights", "id"]
)
dataset.set_format(type="torch")


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


def collate_fn(batch):
    # Add padding to create tensors of equal length within each batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0).to(
        device
    )  # Pad and move to device
    return padded_batch
