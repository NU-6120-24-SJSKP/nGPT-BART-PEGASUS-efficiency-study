from transformers import BartConfig

class TrainingConfig:
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 2
    NUM_EPOCHS = 2
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    EARLY_STOPPING_PATIENCE = 3

def create_small_bart_config():
    config = BartConfig(
        vocab_size=50265,
        max_position_embeddings=512,
        d_model=600,
        encoder_layers=7,
        decoder_layers=7,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2304,
        decoder_ffn_dim=2304,
        activation_function="gelu"
    )
    return config