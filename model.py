from nGPT_pytorch import nGPT

from config import USE_PARAMETRIZE, device

model = nGPT(
    num_tokens=256,
    dim=1024,
    depth=8,
    dim_head=128,
    tied_embedding=True,
    add_value_residual=True,
    attn_norm_qk=False,
    manual_norm_weights=not USE_PARAMETRIZE
).to(device)
