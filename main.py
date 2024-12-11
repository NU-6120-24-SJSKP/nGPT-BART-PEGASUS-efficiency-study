import argparse
import json

from ngpt.train import NGPTTrainer

parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
parser.add_argument(
    "--MODEL", type=str, required=True, help="Model name (e.g., ngpt, bARt, pegasus)"
)
parser.add_argument(
    "--PARAMS",
    type=str,
    required=False,
    help="Path to the JSON file containing model parameters.",
)
parser.add_argument(
    "--TRAIN", type=int, required=False, help="Number of training samples."
)
parser.add_argument(
    "--VALIDATE", type=int, required=False, help="Number of validation samples."
)
args = parser.parse_args()
param_file = args.PARAMS

params = {}
if param_file:
    try:
        with open(param_file, "r") as file:
            params.update(json.load(file))
            print("Loaded parameters from JSON file:")
    except FileNotFoundError:
        print(f"Error: File {param_file} not found. Using default parameters.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {param_file}. Using default parameters.")

if str(args.MODEL).lower() == "ngpt":
    nGPT = NGPTTrainer(params)
    nGPT.train()
elif str(args.MODEL).lower() == "bart":
    pass
elif str(args.MODEL).lower() == "pegasus":
    pass