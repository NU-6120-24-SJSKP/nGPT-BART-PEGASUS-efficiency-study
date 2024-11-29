import torch
import torch.nn.functional as F
from datasets import load_dataset
from rouge_score import rouge_scorer

# Cos they mention in the paper to use this for optimization
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from nGPT import nGPT

"""
__author__ = "sanjiv joshi"
__email__ = "joshi.sanj@northeastern.edu"
__version__ = "train+rouge"
To contribute, fork this file, make suggestions (comment under the code with proposed code),
black format, commit, pull.
"""

train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
val_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")

# I am using a GPT2 Tokenizer, thought this would be a good base if everyone follows, just to be fair.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    return tokenizer(
        examples["article"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )


train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch]),
    }


train_dataloader = DataLoader(
    train_tokenized, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=4
)
val_dataloader = DataLoader(
    val_tokenized, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4
)

# My GPU died a gruesome death
# model = nGPT(
#     vocab_size=tokenizer.vocab_size, d_model=1024, n_heads=14, n_layers=8, d_mlp=4096
# )
model = nGPT(
    vocab_size=tokenizer.vocab_size, d_model=768, n_heads=12, n_layers=8, d_mlp=3072
)

# I do not have LoRA for this to work, this code was pulled from chatGPT
from torch.nn.utils import parameters_to_vector

total_params = parameters_to_vector(model.parameters()).numel()
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# According to the paper, they don't want us to have weight_decay so we should agree on these
# parameters. I still have the weight_decay here to discuss.
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01
)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Our model does not have a loss function per se, but cross_entropy should be a good measure
def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.1,
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Just for the "fun" of it for number crunching.
def evaluate_rouge(model, dataset, num_samples=100):
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    with torch.no_grad():
        for i in range(num_samples):
            input_ids = torch.tensor(dataset[i]["input_ids"]).unsqueeze(0).to(device)
            generated = model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True,
                do_sample=False,
                temperature=1.0,
            )
            generated_summary = tokenizer.decode(generated[0], skip_special_tokens=True)
            reference_summary = val_dataset[i]["highlights"]
            scores = scorer.score(reference_summary, generated_summary)

            for metric in rouge_scores:
                rouge_scores[metric] += scores[metric].fmeasure

    for metric in rouge_scores:
        rouge_scores[metric] /= num_samples

    return rouge_scores


# Train loop.
num_epochs = 20
best_rouge_l = 0
patience = 3
no_improvement = 0
accumulation_steps = 4

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids)
        loss = (
            F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.1,
            )
            / accumulation_steps
        )

        loss.backward()
        total_train_loss += loss.item() * accumulation_steps

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.normalize_all_weights()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}"
            )

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    val_loss = evaluate_loss(model, val_dataloader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    rouge_scores = evaluate_rouge(model, val_tokenized)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.6f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.6f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.6f}")

    if rouge_scores["rougeL"] > best_rouge_l:
        best_rouge_l = rouge_scores["rougeL"]
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "rouge_scores": rouge_scores,
            },
            f"model_rouge_{best_rouge_l:.6f}.pt",
        )
        no_improvement = 0
    else:
        no_improvement += 1

    # Stop early.
    if no_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

    torch.cuda.empty_cache()
