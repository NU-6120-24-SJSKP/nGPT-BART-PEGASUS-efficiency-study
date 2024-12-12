import torch
from tqdm import tqdm

class Tester:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_summary(self, article, max_length=128):
        inputs = self.tokenizer(
            article,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_length,
            early_stopping=True,
            length_penalty=2.0,
            min_length=int(max_length/4),
            no_repeat_ngram_size=4
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def test_model(self, test_data, num_examples=3):
        self.model.eval()
        results = []
        
        print("\nGenerating example summaries...")
        for i in tqdm(range(min(num_examples, len(test_data))), desc="Testing"):
            article = test_data[i]["article"]
            reference = test_data[i]["highlights"]
            generated = self.generate_summary(article)
            
            results.append({
                "article": article,
                "reference": reference,
                "generated": generated
            })
            
            print(f"\nArticle {i+1}:")
            print(f"Reference: {reference}")
            print(f"Generated: {generated}")
            print("-" * 50)
            
        return results

    def batch_generate_summaries(self, articles):
        self.model.eval()
        summaries = []
        
        with torch.no_grad():
            for article in tqdm(articles, desc="Generating Summaries"):
                summary = self.generate_summary(article)
                summaries.append(summary)
                
        return summaries