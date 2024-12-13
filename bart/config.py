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

def create_small_bart_config(params=None):
    try:
        config = BartConfig(
            vocab_size=params['vocab_size'],
            max_position_embeddings=params['pos_embedding'],
            d_model=params['d_model'],
            encoder_layers=params['encoder_layers'],
            decoder_layers=params['decoder_layers'],
            encoder_attention_heads=params['enc_attn_head'],
            decoder_attention_heads=params['dec_attn_head'],
            encoder_ffn_dim=params['enc_ffnn_dim'],
            decoder_ffn_dim=params['decc_ffnn_dim'],
            activation_function="gelu"
        )
    except AttributeError:
        print("Complete set of parameters were not provided for BART \n"
              "Look at parameters help using 'python main.py -h'. Using default values")
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
