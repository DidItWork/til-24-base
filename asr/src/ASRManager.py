import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, WhisperProcessor, Wav2Vec2Processor, WhisperForConditionalGeneration
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

class ASRManager:
    def __init__(self):
        # initialize the model here
        # Load the model and processor
        # model_path = "/workspace/models/whisper"
        # processor_path = "/workspace/models/processor"
        model_path = "openai/whisper-tiny"
        processor_path = "openai/whisper-tiny"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(processor_path)

        # self.model.save_pretrained("asr/src/models/whisper")
        # self.processor.save_pretrained("asr/src/models/processor")

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

if __name__ == "__main__":
    asr_manager = ASRManager()