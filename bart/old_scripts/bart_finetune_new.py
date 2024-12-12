import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments,TrainerCallback
import bitsandbytes as bnb
import numpy as np
import time
from evaluate import load

rouge_metric = load("rouge")

class EpochMetricsCallback(TrainerCallback):
    def __init__(self,steps_per_epoch):
        self.current_step = 0
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_epoch = None
        self.train_start_time = None
        self.current_epoch = 0
        self.last_metrics = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        print(f"Steps per epoch: {self.steps_per_epoch}")
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.last_metrics = metrics
        
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step += 1
        if self.steps_per_epoch and self.current_step % self.steps_per_epoch == 0:
            self.current_epoch += 1
            print(f"\n{'='*50}")
            print(f"End of Epoch {self.current_epoch}")
            if self.last_metrics:
                print(f"ROUGE-1: {self.last_metrics.get('eval_rouge1', 0):.3f}")
                print(f"ROUGE-2: {self.last_metrics.get('eval_rouge2', 0):.3f}")
                print(f"ROUGE-L: {self.last_metrics.get('eval_rougeL', 0):.3f}")
                print(f"Perplexity: {self.last_metrics.get('eval_perplexity', 0):.3f}")
            print(f"{'='*50}\n")
            
    def on_train_end(self, args, state, control, **kwargs):
        train_end_time = time.time()
        total_time = train_end_time - self.train_start_time
        print(f"Training completed in {total_time:.2f} seconds")
        print(f"Total steps completed: {self.current_step}")

class SummarizationTrainer:
    def __init__(self, model_name="facebook/bart-base", load_in_8bit=True, save_path="./summarization_model"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            # load_in_8bit=load_in_8bit,
            # device_map="auto"
        )
        self.save_path = save_path
        
    def prepare_model(self):
        # self.model = prepare_model_for_kbit_training(self.model)
        
        config = LoraConfig(
            r=1,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        
        self.model = get_peft_model(self.model, config)
        
    def preprocess_function(self, examples):
        inputs = self.tokenizer(
            examples["article"],
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        labels = self.tokenizer(
            examples["highlights"],
            max_length=128,
            padding="max_length",
            truncation=True
        )
        
        inputs["labels"] = labels["input_ids"]
        return inputs
    
    def compute_metrics(self, pred):
        """Compute ROUGE and perplexity metrics."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        labels_ids = np.where(labels_ids != -100, labels_ids, self.tokenizer.pad_token_id)
     
        decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        rouge_1 = result["rouge1"]
        rouge_2 = result["rouge2"]
        rouge_l = result["rougeL"]

        log_probs = pred.predictions
        perplexity = np.exp(np.mean(np.log(log_probs)))
        
        return {
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "perplexity": perplexity,
        }
    
    def save_model(self):
        """Save the model and tokenizer to the specified save path."""
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print(f"Model and tokenizer saved to {self.save_path}")
    
    def train(self, num_epochs=3, batch_size=2):
        dataset = load_dataset("cnn_dailymail",split={'train': 'train[:50]', 
                                            'validation': 'validation[:10]'})
        tokenized_dataset = {
        'train': dataset['train'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        ),
        'validation': dataset['validation'].map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['validation'].column_names
        )
        }
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.save_path,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            eval_steps=len(tokenized_dataset['train']) // (batch_size * 4),
            # eval_steps=500,
            save_strategy="epoch",
            logging_steps=100,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=True,
            predict_with_generate=True
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=self.compute_metrics,
            callbacks=[EpochMetricsCallback(len(tokenized_dataset['train']) // (batch_size * 4))]
        )
        
        trainer.train()
        self.save_model()  


def count_trainable_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_and_generate(text, model_path="./summarization_model"):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    
    summary_ids = model.generate(
        input_ids,
        max_length=128,
        min_length=30,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    trainer = SummarizationTrainer()
    trainer.prepare_model()
    total_params = count_trainable_parameters(trainer.model)
    print(f"Total trainable parameters after LoRA: {total_params}")
    trainer.train()
    # article = "When I first started listening to Rogan during 2017, I thought Joey Diaz was one of his best guests. He had crazy stories and had that loud and old school New Yorker type of humor. But then as I listened to him more and listened to his old podcast, I started to get a little drained. A lot of his stories seemed a bit far fetched and they are very depressing. It almost felt like his podcast talks were a form of talk therapy for him. Which is fine but it got to be too much for my tastes.There was a part in their latest episode together where Joey pulls out a metallic bag with weed (I think) and he was crumpling it up in front of the mic and Joe looked visibly uncomfortable. He was trying even kind of dropping hints for Joey to put the bag a away.I think the main problem with the Joe Rogan and Joey Diaz relationship is that they are on way different life trajectories at this point. Rogan was always the more successful but he was still one of the guys. And now he has one of the biggest entertainment platforms in the world. Meanwhile Joey Diaz still seems kind of lost in life despite enjoying moderate success in life in recent years. He was talking about selling weed again and just didn't seem like he ascended as much as we hoped.Meanwhile, comedian guests like Bert Kresicher will now enjoy more time and favor with Rogan because they are more media friendly. Bert is goofy but he is very cunning because he knows the business and knows how to navigate it for increased success. He can promote the shit out of himself and is clearly very ambitious. Whereas Joey Diaz is kind of stuck talking about cocaine, his mom and being a thief in New York and Jersey.And props to Rogan for still having him on out of loyalty and friendship. But it's sad. Because I have noticed (even before this episode) that Rogan almost seems a bit uncomfortable around Diaz at times and even subtly talks down to him. Joey Diaz kind of became that old friend that is stuck in the past and stagnant and is best to hangout with for a quick lunch every now and then for old time's sake.Do you guys agree?"
    # summary = load_and_generate(article)
    # print(f"Generated Summary: {summary}")