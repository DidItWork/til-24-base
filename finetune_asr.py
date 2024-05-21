import base64
import json
from typing import Dict, List
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from scoring.asr_eval import asr_eval
from dotenv import load_dotenv
import torch
import os
import soundfile as sf
from transformers import AutoProcessor, WhisperProcessor, Wav2Vec2Processor, WhisperForConditionalGeneration

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

class ASRManager:
    def __init__(self):
        # initialize the model here
        # Load the model and processo0r
        model_name = "openai/whisper-tiny"
        # self.model = WhisperForConditionalGeneration.from_pretrained("/workspace/models/whisper_large_v3").to(device)
        # self.processor = Wav2Vec2Processor.from_pretrained("/workspace/models/wav2vec2prrocessor")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        # self.processor = Wav2Vec2Processor.from_pretrained("asr/src/models/wav2vec2prrocessor")
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

def main():
    # input_dir = Path(f"/home/jupyter/{TEAM_TRACK}")
    input_dir = Path(f"data/")
    # results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    instances = []

    with open(input_dir / "asr.jsonl", "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "audio" / instance["audio"], "rb") as file:
                audio_bytes = file.read()
                instances.append(
                    {**instance, "b64": base64.b64encode(audio_bytes).decode("ascii")}
                )

    results = run_batched(instances)
    df = pd.DataFrame(results)
    df.to_csv(results_dir / "asr_results.csv", index=False)
    # calculate eval
    eval_result = asr_eval(
        [result["transcript"] for result in results],
        [result["prediction"] for result in results],
    )
    print(f"1-WER: {eval_result}")


def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 4
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        predictions = []
        for instance in _instances:
            # each is a dict with one key "b64" and the value as a b64 encoded string
            audio_bytes = base64.b64decode(instance["b64"])

            transcription = asr_manager.transcribe(audio_bytes)
            predictions.append(transcription)

        _results = predictions
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "transcript": _instances[i]["transcript"],
                    "prediction": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
