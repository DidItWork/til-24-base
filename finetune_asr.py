"""
Referenced from https://huggingface.co/blog/fine-tune-whisper

"""

import base64
import json
from typing import Any, Dict, List, Union
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import torch
import os
import io
import soundfile as sf
from transformers import AutoProcessor, WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset, random_split
import evaluate
from dataclasses import dataclass


load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

device = "cuda" if torch.cuda.is_available() else "cpu"

class ASRManager:
    def __init__(self):
        # initialize the model here
        # Load the model and processor
        model_name = "openai/whisper-tiny"
        
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # self.model.save_pretrained("../models/whisper_large_v3")
        # self.processor.save_pretrained("../models/wav2vec2prrocessor")

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription

        # Function to decode audio bytes
        def decode_audio_bytes(audio_bytes):
            # Use soundfile to read the bytes
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
            return audio, sample_rate
        
        audio_array, sample_rate = decode_audio_bytes(audio_bytes)

        # Convert the audio array to the format expected by the processor
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")

        # Perform inference with the model
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_features.to(device))

        # Decode the generated transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription

class ASRDataset(Dataset):

    def __init__(self, json_file:str = "", data_root:str = "data/", feature_extractor=None, tokenizer=None):

        self.data = self.prepare_data(json_file=json_file, data_root=data_root, feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]

    def prepare_data(self, json_file:str = "", data_root:str = "data/", feature_extractor=None, tokenizer=None) -> List[Dict]:

        #Returns a list of data sample dictionaries with extracted audio features and tokenized transcripts

        instances = []

        with open(data_root / json_file, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                instance = json.loads(line.strip())
                with open(data_root / "audio" / instance["audio"], "rb") as file:
                    # audio_bytes = file.read()
                    audio_array, sampling_rate = sf.read(file)
                    instances.append(
                        {**instance, 
                        "labels": tokenizer(instance["transcript"]).input_ids,
                        "input_features": feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0],}
                    )
        
        return instances

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        #replace padding with -100 to ignore loss correctly
        #Not sure what this does
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        #Check if begin-of-sentence is appended in previous tokenization step
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def main():
    # input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
    input_dir = Path(f"data/")
    # results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = "openai/whisper-tiny.en"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    all_instances = ASRDataset("asr.jsonl", input_dir, feature_extractor, tokenizer)

    train_set, test_set = random_split(all_instances, [0.9, 0.1])

    metric = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir="asr/src/models/whisper-ft",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
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
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    #Function to compute wer metric, copied
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=all_instances,
        eval_dataset=test_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    #Start training
    trainer.train()

if __name__ == "__main__":
    main()
