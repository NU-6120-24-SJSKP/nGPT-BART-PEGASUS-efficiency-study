import re

import torch
from datasets import load_dataset
from torch._C._nn import pad_sequence
from torch.utils.data import Dataset

from ngpt.config import device


# Self-made tokenizer function
def tokenize(examples):
    """
    Tokenize the input text by converting each character to its ASCII value and removing non-ASCII characters.

    :param examples: Dictionary containing the 'article' key with text to tokenize
    :return: Dictionary with 'text' key containing tokenized text
    """
    tokenized_text = []
    for example in examples["article"]:
        # Remove non-ASCII characters from the text
        ascii_text = re.sub(r"[^\x00-\x7F]+", " ", example)
        # Convert each character to its ASCII value
        tokenized_text.append([ord(char) for char in ascii_text])
    return {"text": tokenized_text}


# Data preparation
dataset = load_dataset(
    "cnn_dailymail",
    "3.0.0",
    split={"train": "train[:15000]", "validation": "validation[:1500]"},
)
# This line is commented out, suggesting it's for loading the full dataset
# dataset = load_dataset("cnn_dailymail", "3.0.0")

# Apply tokenization to the dataset
dataset = dataset.map(
    tokenize,
    batched=True,  # Process data in batches for efficiency
    num_proc=1,  # Use 1 process for mapping
    remove_columns=["article", "highlights", "id"],  # Remove unnecessary columns
)
# Set the dataset format to PyTorch tensors
dataset.set_format(type="torch")


class TextSamplerDataset(Dataset):
    """
    Custom dataset class for sampling text sequences of a specified length.
    """

    def __init__(self, data, seq_len):
        """
        Initialize the dataset.

        :param data: The dataset to sample from
        :param seq_len: The desired sequence length for each sample
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset.

        :param index: The index of the sample to retrieve
        :return: A tensor of the sampled sequence, padded if necessary
        """
        text = self.data[index]["text"]

        # Calculate maximum possible sequence length for this text
        max_seq_len = min(self.seq_len, len(text) - 1)

        if max_seq_len < 1:
            # If the text is too short, pad the entire sequence
            padding = torch.zeros(self.seq_len + 1, dtype=torch.long)
            full_seq = padding.to(device)
        else:
            # Adjust sequence length and get a random start position
            rand_start = torch.randint(0, len(text) - max_seq_len, (1,))
            sequence = text[rand_start: rand_start + max_seq_len + 1]

            # Pad if necessary to reach the desired sequence length
            if len(sequence) < self.seq_len + 1:
                padding = torch.zeros(
                    self.seq_len + 1 - len(sequence), dtype=torch.long
                )
                full_seq = torch.cat([sequence, padding]).to(device)
            else:
                full_seq = sequence.to(device)

        return full_seq


def collate_fn(batch):
    """
    Collate function to pad sequences in a batch to the same length.

    :param batch: List of tensors representing sequences
    :return: Padded tensor batch
    """
    # Add padding to create tensors of equal length within each batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0).to(device)
    return padded_batch
