import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, WhisperForConditionalGeneration

class ASRManager:
    def __init__(self):
        # initialize the model here
        # Load the model and processo0r
        model_name = "openai/whisper-large-v3"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription

        # Function to decode audio bytes
        def decode_audio_bytes(audio_bytes):
            # Use soundfile to read the bytes
            audio, sample_rate = sf.read(audio_bytes)
            return audio, sample_rate
        
        audio_array, sample_rate = decode_audio_bytes(audio_bytes)

        # Convert the audio array to the format expected by the processor
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")

        # Perform inference with the model
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_values)

        # Decode the generated transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return transcription
