import os
import torch
from transformers import BartTokenizer
from config import TrainingConfig, create_small_bart_config
from data import load_data, create_dataloaders
from helper import set_seed, verify_model_size, cleanup, inspect_frozen_params
from model import SummarizationModel
from train import Trainer
from validate import Validator
from metrics import evaluate
from plot import (plot_rouge_scores, plot_train_loss_per_step, 
                 plot_val_loss_per_epoch, plot_train_val_loss, 
                 plot_val_perplexity, plot_inference_time, 
                 plot_tokens_per_second, plot_peak_memory_usage, 
                 plot_training_tokens_loss)

def main():
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # Setup
        set_seed()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and tokenizer
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        config = create_small_bart_config()
        summarization_model = SummarizationModel(config, device)
        model = summarization_model.model
        
        # Verify model parameters
        total_params, trainable_params = verify_model_size(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        inspect_frozen_params(model)
        
        # Load data
        train_data, val_data = load_data(num_samples=100)
        train_loader, val_loader = create_dataloaders(
            train_data, 
            val_data, 
            tokenizer, 
            TrainingConfig.BATCH_SIZE, 
            TrainingConfig.MAX_INPUT_LENGTH, 
            TrainingConfig.MAX_TARGET_LENGTH
        )
        
        # Initialize trainer and start training
        trainer = Trainer(
            model, 
            train_loader, 
            val_loader, 
            tokenizer, 
            device, 
            config, 
            TrainingConfig
        )
        training_time = trainer.train()
        
        # Save final model
        model_save_path = "./final_bart_model.pth"
        summarization_model.save_model(model_save_path, config)
        print(f"Final model saved at {model_save_path}")
        
        # Generate plots
        plot_rouge_scores()
        plot_train_loss_per_step()
        plot_val_loss_per_epoch()
        plot_train_val_loss()
        plot_val_perplexity()
        plot_inference_time()
        plot_tokens_per_second()
        plot_peak_memory_usage()
        plot_training_tokens_loss()
        
        # Final evaluation
        print("Evaluating fine-tuned model...")
        rouge_scores, inference_time, total_tokens, peak_memory = evaluate(
            model=model,
            data_loader=val_loader,
            tokenizer=tokenizer,
            device=device
        )
        # validator = Validator(model, val_loader, tokenizer, device)
        # fine_tuned_rouge, _, _, _ = validator.validate_epoch()
        
        # print("Fine-tuned Model Performance:")
        # print(f"ROUGE Scores: {fine_tuned_rouge}")
        # print(f"Total training time: {training_time:.2f} seconds")
        print("Fine-tuned Model Performance:")
        print(f"ROUGE Scores: {rouge_scores}")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Inference time: {inference_time:.2f} seconds")
        print(f"Tokens processed: {total_tokens}")
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        
        # Generate example summaries
        # print("\nGenerating example summaries...")
        # examples = validator.validate_with_generation(num_examples=3)
        # for i, example in enumerate(examples, 1):
        #     print(f"\nArticle {i}:")
        #     print(f"Reference: {example['reference']}")
        #     print(f"Generated: {example['generated']}")
        #     print("-" * 50)
        print("\nGenerating example summaries...")
        for i in range(3):
            article = val_data[i]["article"]
            reference = val_data[i]["highlights"]
            generated = summarization_model.generate_summary(article)
            print(f"\nArticle {i+1}:")
            print(f"Reference: {reference}")
            print(f"Generated: {generated}")
            print("-" * 50)
            
    finally:
        cleanup()

if __name__ == '__main__':
    main()