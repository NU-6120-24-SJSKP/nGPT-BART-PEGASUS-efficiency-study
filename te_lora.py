import pandas as pd
from datasets import Dataset,load_dataset
import evaluate
import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments,AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import os
from bitsandbytes.optim import AdamW

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
                                                    #  load_in_4bit = True,
                                                    #  device_map = "auto",)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

total_params = count_trainable_parameters(model)
print(f"Total trainable parameters before: {total_params}")

lora_config = LoraConfig(
    r=1,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
rouge = evaluate.load("rouge")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device is {device}" )
model.to(device) 

total_params = count_trainable_parameters(model)
print(f"Total trainable parameters after LoRA: {total_params}")



def compute_metrics(eval_pred):
   
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "rougeLsum": result.get("rougeLsum", None)
    }

def preprocess_data(examples):
    inputs = [article for article in examples['article']]
    targets = [summary for summary in examples['highlights']]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = [label for label in labels["input_ids"]]
    return model_inputs

def train_model():
    train_dataset = load_dataset("cnn_dailymail", split='train[:1000]')
    val_dataset = load_dataset("cnn_dailymail", split='validation[:1000]')
    
    # train_df = pd.read_csv("./cnn_dailymail/train.csv").head(1000)
    # val_df = pd.read_csv("./cnn_dailymail/validation.csv").head(1000)
    # train_dataset = Dataset.from_pandas(train_df)
    # val_dataset = Dataset.from_pandas(val_df)
    train_dataset = train_dataset.map(preprocess_data, batched=True)
    val_dataset = val_dataset.map(preprocess_data, batched=True)

    # optimizer = AdamW(
    # params=model.parameters(),  # Pass model parameters
    # lr=2e-4,  # Learning rate
    # weight_decay=0.1,  # Weight decay
    # )
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.1,
        save_total_limit=3,
        fp16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model("./fine_tuned_lora_bart")
    tokenizer.save_pretrained("./fine_tuned_lora_bart")
    model.save_pretrained("./fine_tuned_lora_bart")

def inference():
    model = BartForConditionalGeneration.from_pretrained("./fine_tuned_lora_bart")
    tokenizer = BartTokenizer.from_pretrained("./fine_tuned_lora_bart")
    article = "When I first started listening to Rogan during 2017, I thought Joey Diaz was one of his best guests. He had crazy stories and had that loud and old school New Yorker type of humor. But then as I listened to him more and listened to his old podcast, I started to get a little drained. A lot of his stories seemed a bit far fetched and they are very depressing. It almost felt like his podcast talks were a form of talk therapy for him. Which is fine but it got to be too much for my tastes.There was a part in their latest episode together where Joey pulls out a metallic bag with weed (I think) and he was crumpling it up in front of the mic and Joe looked visibly uncomfortable. He was trying even kind of dropping hints for Joey to put the bag a away.I think the main problem with the Joe Rogan and Joey Diaz relationship is that they are on way different life trajectories at this point. Rogan was always the more successful but he was still one of the guys. And now he has one of the biggest entertainment platforms in the world. Meanwhile Joey Diaz still seems kind of lost in life despite enjoying moderate success in life in recent years. He was talking about selling weed again and just didn't seem like he ascended as much as we hoped.Meanwhile, comedian guests like Bert Kresicher will now enjoy more time and favor with Rogan because they are more media friendly. Bert is goofy but he is very cunning because he knows the business and knows how to navigate it for increased success. He can promote the shit out of himself and is clearly very ambitious. Whereas Joey Diaz is kind of stuck talking about cocaine, his mom and being a thief in New York and Jersey.And props to Rogan for still having him on out of loyalty and friendship. But it's sad. Because I have noticed (even before this episode) that Rogan almost seems a bit uncomfortable around Diaz at times and even subtly talks down to him. Joey Diaz kind of became that old friend that is stuck in the past and stagnant and is best to hangout with for a quick lunch every now and then for old time's sake.Do you guys agree?"
    inputs = tokenizer([article], max_length=256, return_tensors="pt", truncation=True, padding="max_length")
    summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
    print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

def main():
    torch.cuda.empty_cache()
    train_model()
    # inference()

if __name__ == "__main__":
    main()
