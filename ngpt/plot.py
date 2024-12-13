import pickle

import matplotlib.pyplot as plt

# Load metrics from a pickle file
with open("results/metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# Extract various metrics from the loaded data
train_loss_per_epoch = metrics["train_loss_per_epoch"]
val_loss_per_epoch = metrics["val_loss_per_epoch"]
perplexities = metrics["perplexities"]
generation_perplexities = metrics["generation_perplexities"]
tokens_per_second = metrics["tokens_per_second"]
memory_usages = metrics["memory_usages"]
inference_times = metrics["inference_times"]
training_tokens = metrics["training_tokens"]
rouge_scores = metrics["rouge_scores"]

# Plot Training Loss vs Epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_loss_per_epoch) + 1), train_loss_per_epoch)
plt.title("Training Loss vs Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.savefig("train_loss_vs_epoch.png")

# Plot Validation Loss vs Epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_loss_per_epoch) + 1), val_loss_per_epoch)
plt.title("Validation Loss vs Epochs")
plt.xlabel("Validation Epochs")
plt.ylabel("Validation Loss")
plt.savefig("validation_loss_vs_epoch.png")

# Plot both Training and Validation Loss vs Epochs
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(train_loss_per_epoch) + 1),
    train_loss_per_epoch,
    color="blue",
    label="Training",
)
plt.plot(
    range(1, len(val_loss_per_epoch) + 1),
    val_loss_per_epoch,
    color="red",
    label="Validation",
)
plt.title("Training and Validation Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("train_loss_validation_loss_vs_epoch.png")

# Plot Validation Perplexity vs Epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(val_loss_per_epoch) + 1), perplexities)
plt.title("Validation Perplexity (Combined)")
plt.xlabel("Epochs (Validation Steps)")
plt.ylabel("Perplexity")
plt.tight_layout()
plt.savefig("loss_vs_perplexity.png")

# Inference plots
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), inference_times)
plt.title("Inference Time")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Time (seconds)")
plt.savefig("inference_time.png")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), tokens_per_second)
plt.title("Tokens per Second")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Tokens/sec")
plt.savefig("tokens_per_second.png")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), memory_usages)
plt.title("Peak Memory Usage")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Memory Usage (MB)")
plt.savefig("inference_graphs.png")

# Plot Generation Perplexity vs Epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(generation_perplexities) + 1), generation_perplexities)
plt.title("Generation epochs vs Generation Perplexity")
plt.xlabel("Epochs (Generation Steps)")
plt.ylabel("Generation Perplexity")
plt.savefig("generation_perplexities.png")
# plt.show()  # Commented out, likely for non-interactive plotting

# Plot Validation Loss vs Training Tokens
plt.figure(figsize=(8, 6))
plt.plot(training_tokens, val_loss_per_epoch)
plt.title("Validation Loss vs. Training Tokens")
plt.xlabel("Training Tokens")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.savefig("tokens_per_loss.png")

# Plot ROUGE Scores
plt.figure(figsize=(8, 6))
rouge1_fmeasures = [score["rouge1"].fmeasure for score in metrics["rouge_scores"]]
rouge2_fmeasures = [score["rouge2"].fmeasure for score in metrics["rouge_scores"]]
rougeL_fmeasures = [score["rougeL"].fmeasure for score in metrics["rouge_scores"]]
plt.plot(range(1, len(rouge1_fmeasures) + 1), rouge1_fmeasures, label="R1F-measure")
plt.plot(range(1, len(rouge2_fmeasures) + 1), rouge2_fmeasures, label="R2F-measure")
plt.plot(range(1, len(rougeL_fmeasures) + 1), rougeL_fmeasures, label="RLF-measure")
plt.legend()
plt.title("Rouge1 vs. Rouge2 vs. RougeL")
plt.savefig("rouge1_vs_rouge2_vs_rougeL.png")
