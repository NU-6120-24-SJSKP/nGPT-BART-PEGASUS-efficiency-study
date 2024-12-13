"""
Configuration file for the text summarization model.
Contains all hyperparameters and settings used throughout the project.
"""

from transformers import PegasusConfig

class TrainingConfig:
    """
    Configuration class containing all training-related parameters.
    """
    # Model input/output parameters
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    BATCH_SIZE = 4
    
    # Training hyperparameters
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3
    
    # Dataset parameters
    NUM_SAMPLES = 5000
    
    # Random seed for reproducibility
    SEED = 42

class ModelConfig:
    """
    Configuration class for the Pegasus model architecture.
    """
    @staticmethod
    def create_small_pegasus_config():
        """
        Create a smaller Pegasus configuration suitable for training on limited data.
        
        Returns:
            PegasusConfig: Configuration object for the Pegasus model
        """
        return PegasusConfig(
            vocab_size=96103,  # Original vocab size for tokenizer compatibility
            encoder_layers=8,   # Reduced from 16
            decoder_layers=8,   # Reduced from 16
            encoder_attention_heads=16,
            decoder_attention_heads=16,
            encoder_ffn_dim=2048,  # Reduced from 4096
            decoder_ffn_dim=2048,  # Reduced from 4096
            d_model=512,          # Reduced from 1024
            max_position_embeddings=512,
            pad_token_id=0,
            eos_token_id=1,
            forced_eos_token_id=1,
            activation_function='gelu',
            dropout=0.2,           # Increased dropout for smaller dataset
            attention_dropout=0.2,
            activation_dropout=0.2,
            num_beams=4,
            encoder_layerdrop=0.1, # Added layerdrop for regularization
            decoder_layerdrop=0.1,
            scale_embedding=True,
            use_cache=True,
            is_encoder_decoder=True
        )

class GenerationConfig:
    """
    Configuration class for text generation parameters.
    """
    NUM_BEAMS = 4
    LENGTH_PENALTY = 2.0
    NO_REPEAT_NGRAM_SIZE = 4
    MIN_LENGTH_RATIO = 0.25  # Minimum length will be max_length * MIN_LENGTH_RATIO

class PathConfig:
    """
    Configuration class for file paths and locations.
    """
    BEST_MODEL_PATH = "/content/drive/My Drive/NLP-Project/TEST.pt"
    METRICS_FILE = "training_metrics.pkl"
    CHECKPOINT_PREFIX = "emergency_checkpoint_epoch_"
