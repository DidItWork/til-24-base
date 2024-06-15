# Standard imports
import logging
from typing import Dict, List, Any
import json
import os

# Third-party imports
import torch
from sklearn.metrics import accuracy_score
from transformers import (AutoTokenizer, EncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from datasets import Dataset
from torch.utils.data import random_split

from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(pred) -> Dict[str, float]:
    """
    Function to compute token-wise accuracy.
    Args:
    - pred: Predicted outputs
    
    Returns:
    - A dictionary containing the accuracy metric.
    """
    
    preds = pred.predictions
    labels = pred.label_ids

    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    accuracy = accuracy_score(labels_flat, preds_flat)
    return {"accuracy": accuracy}

# Initialize the logger
logger = logging.getLogger(__name__)

def prepare_data(json_file:str = "", data_root:Any = None, tokenizer=None) -> Dataset:

    #Returns a list of data sample dictionaries with extracted audio features and tokenized transcripts

    # input_text = ["Turret, prepare to deploy electromagnetic pulse. Heading zero six five, target is grey and white fighter jet. Engage when ready."]*100
    # output_text = ["electromagnetic pulse | 065 | grey and white fighter jet"]*100

    input_text = []
    output_text = []

    with open(data_root / json_file, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            input_text.append(instance["transcript"])
            output_text.append(" | ".join([instance["tool"],instance["heading"],instance["target"]]))

            
    input_tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    output_tokenized = tokenizer(output_text, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    
    return Dataset.from_dict({
        "input_ids": input_tokenized.input_ids,
        "attention_mask": input_tokenized.attention_mask,
        "labels": output_tokenized.input_ids,
    })


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")


model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.config.min_length = 0
model.config.max_length = 64
model.config.length_penalty = 0.0

all_data = prepare_data(json_file="nlp.jsonl", data_root=Path(f"data/"), tokenizer=tokenizer)

train_data, test_data = random_split(all_data, [0.8, 0.2])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/benluo/til-24-base/nlp/weights/nlp-ft",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    dataloader_num_workers=4,
    # gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    # generation_max_length=64,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    # report_to=["tensorboard"],
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Log the start of training
logger.info("Starting training...")

# Train (This should overfit the model to the single example)
trainer.train()

# Log the end of training
logger.info("Training completed.")

# Evaluate the model
model.eval()

print(model.config)

with torch.no_grad():
    example_input = "Turret, prepare to deploy electromagnetic pulse. Heading zero six five, target is grey and white fighter jet. Engage when ready."
    input_tokenized = tokenizer(example_input, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
    outputs = model.generate(input_tokenized.input_ids, attention_mask=input_tokenized.attention_mask)
    # outputs = model(**input_tokenized)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Input String: {example_input}")
print(f"Predicted Output: {output_str}")