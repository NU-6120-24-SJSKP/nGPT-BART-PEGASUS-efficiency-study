"""
Model management module for text summarization.
Handles model initialization, configuration, and text generation.
"""

import torch
from typing import Optional, Dict, Any
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from config import ModelConfig, TrainingConfig, GenerationConfig

class SummarizationModel:
    """
    Wrapper class for managing the summarization model.
    Handles model initialization, configuration, and generation.
    """
    
    def __init__(
        self,
        device: torch.device,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the summarization model.
        
        Args:
            device: Device to place the model on
            model_config: Optional custom model configuration
        """
        self.device = device
        self.config = (model_config if model_config is not None 
                      else ModelConfig.create_small_pegasus_config())
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        
    def initialize_model(self) -> None:
        """
        Initialize the Pegasus model and tokenizer.
        """
        try:
            self.model = PegasusForConditionalGeneration(self.config).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
            print("Model and tokenizer initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def get_optimizer_groups(self) -> list:
        """
        Create parameter groups for optimization with weight decay handling.
        
        Returns:
            List of parameter groups for optimizer
        """
        if not self.model:
            raise ValueError("Model not initialized. Call initialize_model first.")
            
        no_decay = ['bias', 'LayerNorm.weight']
        return [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': TrainingConfig.WEIGHT_DECAY,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
    
    def generate_summary(
        self,
        article: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> str:
        """
        Generate a summary for the given article.
        
        Args:
            article: Input article text
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            
        Returns:
            Generated summary text
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not initialized. Call initialize_model first.")
            
        try:
            # Set generation lengths
            max_length = max_length or TrainingConfig.MAX_TARGET_LENGTH
            min_length = min_length or int(max_length * GenerationConfig.MIN_LENGTH_RATIO)
            
            # Tokenize input
            inputs = self.tokenizer(
                article,
                max_length=TrainingConfig.MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=GenerationConfig.NUM_BEAMS,
                max_length=max_length,
                min_length=min_length,
                length_penalty=GenerationConfig.LENGTH_PENALTY,
                no_repeat_ngram_size=GenerationConfig.NO_REPEAT_NGRAM_SIZE,
                early_stopping=True
            )
            
            # Decode and return summary
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
        
    def save_model(self, path: str) -> None:
        """
        Save the model state.
        
        Args:
            path: Path to save the model
        """
        if not self.model:
            raise ValueError("Model not initialized. Call initialize_model first.")
            
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, path: str) -> None:
        """
        Load the model state.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = PegasusForConditionalGeneration.from_pretrained(path).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(path)
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if model and tokenizer are initialized."""
        return self.model is not None and self.tokenizer is not None
    
    def to(self, device: torch.device) -> None:
        """
        Move the model to specified device.
        
        Args:
            device: Device to move the model to
        """
        if self.model:
            self.model.to(device)
            self.device = device
            
    def train(self) -> None:
        """Set the model to training mode."""
        if self.model:
            self.model.train()
            
    def eval(self) -> None:
        """Set the model to evaluation mode."""
        if self.model:
            self.model.eval()
