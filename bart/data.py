from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length, max_target_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]['article']
        summary = self.data[idx]['highlights']
        
        inputs = self.tokenizer(article, max_length=self.max_input_length, truncation=True, padding='max_length', return_tensors='pt')
        targets = self.tokenizer(summary, max_length=self.max_target_length, truncation=True, padding='max_length', return_tensors='pt')
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze()
        }

def load_data(num_samples=1000):
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    
    train_samples = int(0.9 * num_samples)
    val_samples = num_samples - train_samples
    
    train_indices = range(train_samples)
    val_indices = range(len(dataset['validation']))[:val_samples]
    
    train_data = dataset['train'].select(train_indices)
    val_data = dataset['validation'].select(val_indices)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def create_dataloaders(train_data, val_data, tokenizer, batch_size, max_input_length, max_target_length):
    train_dataset = SummarizationDataset(train_data, tokenizer, max_input_length, max_target_length)
    val_dataset = SummarizationDataset(val_data, tokenizer, max_input_length, max_target_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader