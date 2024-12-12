from rouge_score import rouge_scorer
import torch
import time
from tqdm import tqdm

def evaluate(model, data_loader, tokenizer, device):
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    total_time = 0.0
    total_tokens = 0
    peak_memory = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            generated_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                max_length=128
            )

            step_time = time.time() - start_time
            total_time += step_time

            if torch.cuda.is_available():
                current_peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                peak_memory = max(peak_memory, current_peak_memory)

            total_tokens += sum(len(ids) for ids in generated_ids)

            for i in range(len(input_ids)):
                reference = tokenizer.decode(labels[i], skip_special_tokens=True)
                generated_summary = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                rouge_result = scorer.score(reference, generated_summary)
                
                for metric in rouge_scores:
                    rouge_scores[metric].append(rouge_result[metric].fmeasure)

    avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}
    return avg_rouge, total_time, total_tokens, peak_memory

def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()

def calculate_tokens_per_second(total_tokens, total_time):
    return total_tokens / total_time if total_time > 0 else 0