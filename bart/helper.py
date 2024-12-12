import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def verify_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close('all')

def inspect_frozen_params(model):
    frozen_params = []
    trainable_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(name)
        else:
            trainable_params.append(name)
    
    print("\nFrozen parameters:")
    for name in frozen_params:
        print(f"- {name}")
    
    print("\nNumber of frozen parameters:", len(frozen_params))
    print("Number of trainable parameters:", len(trainable_params))

def load_saved_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    config = checkpoint['config']
    return model.to(device), tokenizer, config