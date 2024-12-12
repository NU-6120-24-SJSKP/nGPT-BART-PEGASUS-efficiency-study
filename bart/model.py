from transformers import BartForConditionalGeneration, BartTokenizer
import torch

class SummarizationModel:
    def __init__(self, config, device):
        self.model = BartForConditionalGeneration(config).to(device)
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
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

    def save_model(self, path, config):
        torch.save({
            'model': self.model,
            'tokenizer': self.tokenizer,
            'config': config
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model = checkpoint['model'].to(self.device)
        self.tokenizer = checkpoint['tokenizer']
        return checkpoint['config']

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total_params, trainable_params

    def freeze_params(self):
        frozen_params = []
        trainable_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                frozen_params.append(name)
            else:
                trainable_params.append(name)
        
        return frozen_params, trainable_params