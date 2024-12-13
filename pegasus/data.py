"""
Data processing module for text summarization.
Handles dataset loading, processing, and DataLoader creation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Tuple, Dict, Any
from pegasus.config import TrainingConfig

class SummarizationDataset(Dataset,TrainingConfig):
    """
    Custom Dataset class for text summarization task.
    Handles tokenization and preparation of input-target pairs.
    """
    
    def __init__(self, 
                 data: Any, 
                 tokenizer: PreTrainedTokenizer, 
                 max_input_length: int, 
                 max_target_length: int):
        """
        Initialize the dataset.
        
        Args:
            data: Raw dataset containing articles and summaries
            tokenizer: Tokenizer for processing text
            max_input_length: Maximum length for input sequences
            max_target_length: Maximum length for target sequences
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        article = self.data[idx]['article']
        summary = self.data[idx]['highlights']
        
        # Tokenize input article
        inputs = self.tokenizer(
            article,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize target summary
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze()
        }

class DataManager:
    """
    Manages data loading and preparation for training and validation.
    """
    
    @staticmethod
    def load_data(num_samples: int = TrainingConfig.NUM_SAMPLES) -> Tuple[Any, Any]:
        """
        Load and split the CNN/DailyMail dataset.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            Tuple containing training and validation datasets
        """
        # Load the dataset
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        
        # Calculate split sizes
        train_samples = int(0.9 * num_samples)  # 90% for training
        val_samples = num_samples - train_samples  # 10% for validation
        
        # Select indices for training and validation
        train_indices = range(train_samples)
        val_indices = range(len(dataset['validation']))[:val_samples]
        
        # Create the splits
        train_data = dataset['train'].select(train_indices)
        val_data = dataset['validation'].select(val_indices)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        return train_data, val_data

    @staticmethod
    def create_dataloaders(
        train_data: Any,
        val_data: Any,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = TrainingConfig.BATCH_SIZE
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoader objects for training and validation.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            tokenizer: Tokenizer for processing text
            batch_size: Batch size for DataLoaders
            
        Returns:
            Tuple containing training and validation DataLoaders
        """
        # Create dataset objects
        train_dataset = SummarizationDataset(
            train_data,
            tokenizer,
            TrainingConfig.MAX_INPUT_LENGTH,
            TrainingConfig.MAX_TARGET_LENGTH
        )
        
        val_dataset = SummarizationDataset(
            val_data,
            tokenizer,
            TrainingConfig.MAX_INPUT_LENGTH,
            TrainingConfig.MAX_TARGET_LENGTH
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
