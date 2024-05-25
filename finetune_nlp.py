# Standard imports
import logging
from typing import Dict, List, Any
import json

# Third-party imports
import torch
from sklearn.metrics import accuracy_score
from transformers import (BertTokenizer, EncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from torch.utils.data import Dataset, random_split

from pathlib import Path
from dataclasses import dataclass

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(pred) -> Dict[str, float]:
    """
    Function to compute token-wise accuracy.
    Args:
    - pred: Predicted outputs
    
    Returns:
    - A dictionary containing the accuracy metric.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    accuracy = accuracy_score(labels_flat, preds_flat)
    return {"accuracy": accuracy}

# Initialize the logger
logger = logging.getLogger(__name__)

class NLPDataset(Dataset):

    def __init__(self, json_file:str = "", data_root:Any = None, tokenizer=None):

        self.data = self.prepare_data(json_file=json_file, data_root=data_root, tokenizer=tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]

    def prepare_data(self, json_file:str = "", data_root:Any = None, tokenizer=None) -> List[Dict]:

        #Returns a list of data sample dictionaries with extracted audio features and tokenized transcripts

        instances = []

        with open(data_root / json_file, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                instance = json.loads(line.strip())
                output_text = " | ".join([instance["tool"],instance["heading"],instance["target"]])
                input_tokenized = tokenizer(instance["transcript"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                output_tokenized = tokenizer(output_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

                instances.append(
                    {
                        "transcript": instance["transcript"],
                        "output_text": output_text,
                        "input_ids": input_tokenized.input_ids,
                        "attention_mask": input_tokenized.attention_mask,
                        "labels": output_tokenized.input_ids,
                    }
                )
        
        return instances

@dataclass
class DataCollator:

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:

        batch = dict()

        batch["input_ids"] = torch.cat([feature["input_ids"] for feature in features],dim=0)
        batch["attention_mask"] = torch.cat([feature["attention_mask"] for feature in features],dim=0)
        batch["labels"] = torch.cat([feature["labels"] for feature in features],dim=0)

        return batch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-cased")
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "google-bert/bert-base-cased")
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

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

all_data = NLPDataset(json_file="nlp.jsonl", data_root=Path(f"data/"), tokenizer=tokenizer)

train_data, test_data = random_split(all_data, [0.8, 0.2])

data_collator = DataCollator()

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/benluo/til-24-base/nlp/weights/nlp-ft",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=100,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Log the start of training
logger.info("Starting training...")

# Train (This should overfit the model to the single example)
# trainer.train()

# Log the end of training
logger.info("Training completed.")

# Evaluate the model
bert2bert.eval()

print(bert2bert.config)

with torch.no_grad():
    example_input = test_data[0]["transcript"]
    input_tokenized = tokenizer(example_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    outputs = bert2bert.generate(input_tokenized.input_ids.to("cuda"), attention_mask=input_tokenized.attention_mask.to("cuda"))
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(f"Predicted Output: {output_str}")