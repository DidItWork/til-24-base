import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, WhisperForConditionalGeneration
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

class ASRManager:
    def __init__(self):
        # Load the model and processor
        model_path = "models/whisper-ft"
        processor_path = "models/processor"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(processor_path)

    def decode_audio_bytes(self, audio_bytes):
        # Use soundfile to read the bytes
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
        return audio, sample_rate

    def transcribe(self, audio_bytes: bytes) -> str:
        audio_array, sample_rate = self.decode_audio_bytes(audio_bytes)
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_features.to(device))
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

    def transcribe_batch(self, audio_bytes_list: list) -> list:
        transcriptions = []
        for audio_bytes in audio_bytes_list:
            transcription = self.transcribe(audio_bytes)
            transcriptions.append(transcription)
        return transcriptions

if __name__ == "__main__":
    asr_manager = ASRManager()

    # Example usage
    audio_files = [b'first_audio_bytes', b'second_audio_bytes']  # List of audio byte objects
    transcriptions = asr_manager.transcribe_batch(audio_files)
    for transcription in transcriptions:
        print(transcription)
