from nGPT_pytorch import nGPT

from ngpt.config import device

# Flag to determine whether to use parameterization in the model
USE_PARAMETRIZE = True
model = None  # Global variable to store the model instance


def init_model(params):
    """
    Initialize an nGPT model with the given parameters or default values if parameters are missing.

    :param params: Dictionary containing model parameters
    :return: Initialized nGPT model instance
    """
    global model

    try:
        # Attempt to initialize the model with provided parameters
        model = nGPT(
            num_tokens=params["num_tokens"],  # Number of tokens in the vocabulary
            dim=params["dim"],  # Dimensionality of the model
            depth=params["depth"],  # Number of transformer layers
            dim_head=params["dim_head"],  # Dimensionality of each attention head
            tied_embedding=params[
                "tied_embedding"
            ],  # Whether to tie input and output embeddings
            add_value_residual=params[
                "add_value_residual"
            ],  # Add residual connection to value in attention
            attn_norm_qk=params["attn_norm_qk"],  # Normalize query and key in attention
            manual_norm_weights=params[
                "manual_norm_weights"
            ],  # Use manual normalization weights
        ).to(device)
    except KeyError as e:
        # If any parameter is missing, print a message and use default values
        print(
            "Complete set of parameters were not provided for nGPT \n"
            "Look at parameters help using 'python main.py -h'. Using default values"
        )
        model = nGPT(
            num_tokens=256,  # Default vocabulary size
            dim=1024,  # Default model dimensionality
            depth=8,  # Default number of layers
            dim_head=128,  # Default attention head dimensionality
            tied_embedding=True,  # Default to tie embeddings
            add_value_residual=True,  # Default to add residual to value
            attn_norm_qk=False,  # Default not to normalize query and key
            manual_norm_weights=not USE_PARAMETRIZE,  # Use manual normalization if not using parameterization
        ).to(device)
    return model


"""
Varied the dim, depth, dim_head in several experiments to determine 
the right values for 15k samples. 
There were times where model was converging within 2 epochs with 
validation loss still plenty to reduce due to mismatches.

model = nGPT(
            num_tokens=256, 
            dim=1024,
            depth=8,
            dim_head=128,
            tied_embedding=True,
            add_value_residual=True,
            attn_norm_qk=False, # they say the query/key normalization is optional
            manual_norm_weights=not USE_PARAMETRIZE,
        ).to(device)
"""
