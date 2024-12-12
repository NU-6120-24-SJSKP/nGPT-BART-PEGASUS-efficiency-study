from nGPT_pytorch import nGPT

from ngpt.config import device

USE_PARAMETRIZE = True
model = None


def init_model(params):
    global model
    try:
        model = nGPT(
            num_tokens=params["num_tokens"],
            dim=params["dim"],
            depth=params["depth"],
            dim_head=params["dim_head"],
            tied_embedding=params["tied_embedding"],
            add_value_residual=params["add_value_residual"],
            attn_norm_qk=params["attn_norm_qk"],
            manual_norm_weights=params["manual_norm_weights"],
        ).to(device)
    except AttributeError as e:
        print("Complete set of parameters were not provided for BART \n"
              "Look at parameters help using 'python main.py -h'. Using default values")
        model = nGPT(
            num_tokens=256,
            dim=1024,
            depth=8,
            dim_head=128,
            tied_embedding=True,
            add_value_residual=True,
            attn_norm_qk=False,
            manual_norm_weights=not USE_PARAMETRIZE,
        ).to(device)
    return model
