# Standard imports
import logging
from typing import Dict, List, Any
import json

# Third-party imports
import torch
from sklearn.metrics import accuracy_score
from transformers import (BertTokenizer, T5ForConditionalGeneration,
                          AutoTokenizer, EncoderDecoderModel, BertModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          BertGenerationEncoder, BertGenerationDecoder,
                          DataCollatorWithPadding)
from datasets import Dataset
from torch.utils.data import random_split

from pathlib import Path
from dataclasses import dataclass

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(pred) -> Dict[str, float]:
    """
    Function to compute token-wise accuracy.
    Args:
    - pred: Predicted outputs
    
    Returns:
    - A dictionary containing the accuracy metric.
    """
    
    preds = pred.predictions.detach()
    labels = pred.label_ids.detach()
    print(preds.shape, labels.shape)
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

    # print(input_text[0])
    # print(output_text[0])

    # print(input_tokenized.input_ids[0])
    # print(input_tokenized.attention_mask[0])
    # print(output_tokenized.input_ids[0])
    
    return Dataset.from_dict({
        "input_ids": input_tokenized.input_ids,
        "attention_mask": input_tokenized.attention_mask,
        "labels": output_tokenized.input_ids,
    })

# @dataclass
# class DataCollator:

#     def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:

#         batch = dict()

#         batch["input_ids"] = torch.cat([feature["input_ids"] for feature in features],dim=0)
#         batch["labels"] = torch.cat([feature["labels"] for feature in features],dim=0)
#         batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#         batch["attention_mask"] = torch.cat([feature["attention_mask"] for feature in features],dim=0)
        

#         return batch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
# tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

# use BERT's cls token as BOS token and sep token as EOS token
# encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-base-cased", bos_token_id=101, eos_token_id=102)
# # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
# decoder = BertGenerationDecoder.from_pretrained(
#     "google-bert/bert-base-cased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
# )
# bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)
# model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')

# bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "google-bert/bert-base-cased")

# bert2bert = BertModel.from_pretrained("bert-base-cased")
model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.config.min_length = 0
model.config.max_length = 64

# text1 = "La economía circular se ha convertido en una tendencia prometedora y vital. Alberto Rodríguez ha estado trabajando con baterías eléctricas desde el año 2010"
# output1 = "La economía circular@@@se ha convertido en@@@una tendencia prometedora y vital |||  Alberto Rodríguez@@ha estado trabajando con@@baterías eléctricas desde el año 2010"

# text2= """se reciclan baterías usadas desde el año 2013. planta de Nissan reciclan baterías usadas
# Carlos Ghosn habló la visión de la empresa de reducir el impacto negativo de sus productos en el medio ambiente. Carlos Ghosn Destacó el uso de baterías eléctricas es fundamental para reducir significativamente las emisiones de CO2
# su equipo profesional se esfuerza en desarrollar tecnologías más avanzadas. María Fernández Nacida en 1987 en Barcelona
# La economía circular de las baterías eléctricas es beneficiosa para el medio ambiente. Otra empresa trabaja en la economía circular de las baterías eléctricas
# Umicore es especialista en la fabricación y reciclaje de baterías. Umicore abrió una planta de reciclaje de baterías de iones de litio en Bélgica en 2019"""

# output2="""se reciclan@@@baterías usadas@@@desde el año 2013. planta de Nissan@@@reciclan@@@baterías usadas
# Carlos Ghosn@@@habló@@@la visión de la empresa de reducir el impacto negativo de sus productos en el medio ambiente. Carlos Ghosn@@@Destacó@@@el uso de baterías eléctricas es fundamental para reducir significativamente las emisiones de CO2
# su equipo profesional@@@se esfuerza en desarrollar@@@tecnologías más avanzadas. María Fernández@@@Nacida en@@@1987 en Barcelona
# La economía circular de las baterías eléctricas@@@es@@@beneficiosa para el medio ambiente. Otra empresa@@@trabaja@@@en la economía circular de las baterías eléctricas
# Umicore@@@es especialista en@@@la fabricación y reciclaje de baterías. Umicore@@@abrió@@@una planta de reciclaje de baterías de iones de litio en Bélgica en 2019"""

all_data = prepare_data(json_file="nlp.jsonl", data_root=Path(f"data/"), tokenizer=tokenizer)

train_data, test_data = random_split(all_data, [0.8, 0.2])

print(all_data)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/benluo/til-24-base/nlp/weights/nlp-ft",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=1000,
    # dataloader_num_workers=4,
    # gradient_checkpointing=True,
    # fp16=True,
    # evaluation_strategy="steps",
    # per_device_eval_batch_size=1,
    predict_with_generate=True,
    # generation_max_length=64,
    save_steps=1000,
    # eval_steps=10,
    logging_steps=25,
    # report_to=["tensorboard"],
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    # eval_dataset=test_data,
    # compute_metrics=compute_metrics,
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