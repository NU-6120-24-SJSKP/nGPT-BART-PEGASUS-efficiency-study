import matplotlib.pyplot as plt
import pickle

def plot_rouge_scores():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(12, 8))

    for metric, scores in saved_metrics["rouge_scores"].items():
        plt.plot(scores, label=metric)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('ROUGE Scores')
    plt.savefig('rouge_scores.png')
    plt.close()
    # files.download('training_progress.png')

def plot_train_loss_per_step():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(saved_metrics["train_batch_losses"], label='Training Loss per Step', color='blue', linewidth=0.7)
    plt.xlabel('Batch/Step')
    plt.ylabel('Loss')
    plt.title('Training Loss per Step')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_per_step.png')
    plt.close()

def plot_val_loss_per_epoch():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["val_losses"]) + 1), saved_metrics["val_losses"], label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_loss_per_epoch.png')
    plt.close()

def plot_train_val_loss():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    epochs = range(1, len(saved_metrics["train_losses"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, saved_metrics["train_losses"], label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, saved_metrics["val_losses"], label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_val_loss_per_epoch.png')
    plt.close()

def plot_val_perplexity():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["val_perplexities"]) + 1), saved_metrics["val_perplexities"], label='Validation Perplexity', color='green', marker='o')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('val_perplexity_per_epoch.png')
    plt.close()

def plot_inference_time():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["inference_times"]) + 1), saved_metrics["inference_times"], label='Inference Time (s)', color='purple', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Inference Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('inference_time_vs_epoch.png')
    plt.close()

def plot_tokens_per_second():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["tokens_per_seconds"]) + 1), saved_metrics["tokens_per_seconds"], label='Tokens per Second', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Tokens per Second')
    plt.title('Tokens per Second per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('tokens_per_second_vs_epoch.png')
    plt.close()

def plot_peak_memory_usage():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(saved_metrics["peak_memory_usages"]) + 1), saved_metrics["peak_memory_usages"], label='Peak Memory Usage (MB)', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Memory (MB)')
    plt.title('Peak Memory Usage per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('peak_memory_usage_vs_epoch.png')
    plt.close()

def plot_training_tokens_loss():
    with open("training_metrics.pkl", 'rb') as f:
        saved_metrics = pickle.load(f)
    plt.figure(figsize=(8, 6))
    plt.plot(saved_metrics["training_tokens"], saved_metrics["val_losses"])
    plt.title("Validation Loss vs. Training Tokens")
    plt.xlabel("Training Tokens")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.savefig('plot_training_tokens_loss.png')
    plt.close()


# plot_rouge_scores()
# plot_train_loss_per_step()
# plot_val_loss_per_epoch()
# plot_train_val_loss()
# plot_val_perplexity()
# plot_inference_time()
# plot_tokens_per_second()
# plot_peak_memory_usage()
# plot_training_tokens_loss()
