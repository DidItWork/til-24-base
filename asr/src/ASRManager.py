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

        # model_path = "openai/whisper-tiny.en"
        # processor_path = "openai/whisper-tiny.en"
        model_path = "/home/benluo/til-24-base/asr/src/models/whisper-ft/checkpoint-4000"
        processor_path = "/home/benluo/til-24-base/asr/src/models/processor"
        # self.model = WhisperForConditionalGeneration.from_pretrained("/workspace/models/whisper_large_v3").to(device)
        # self.processor = Wav2Vec2Processor.from_pretrained("/workspace/models/wav2vec2prrocessor")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        # self.processor = Wav2Vec2Processor.from_pretrained("asr/src/models/wav2vec2prrocessor")
        self.processor = AutoProcessor.from_pretrained(processor_path)

        # self.model.save_pretrained("/home/benluo/til-24-base/asr/src/models/whisper")
        # self.processor.save_pretrained("/home/benluo/til-24-base/asr/src/models/processor")

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

    while True:
        pass