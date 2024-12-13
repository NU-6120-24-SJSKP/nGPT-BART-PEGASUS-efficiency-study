import torch
from tqdm import tqdm

class Validator:
    def __init__(self, model, val_loader, tokenizer, device):
        self.model = model
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device

    def validate_epoch(self):
        self.model.eval()
        total_val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )

                    total_val_loss += outputs.loss.item()
                    val_batch_count += 1

                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("WARNING: out of memory in validation. Skipping batch...")
                        continue
                    else:
                        raise e

        if val_batch_count > 0:
            avg_val_loss = total_val_loss / val_batch_count
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            return avg_val_loss, val_perplexity
        return float('inf'), float('inf')

    def validate_with_generation(self, num_examples=3):
        self.model.eval()
        examples = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_examples:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=2.0,
                    min_length=32,
                    no_repeat_ngram_size=4
                )

                reference = self.tokenizer.decode(labels[0], skip_special_tokens=True)
                generated = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                examples.append({
                    'reference': reference,
                    'generated': generated
                })

        return examples