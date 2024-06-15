from typing import List
from transformers import AutoProcessor, Owlv2ForObjectDetection
import torch
from PIL import Image
import io
from numpy import argmax
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class VLMManager:
    def __init__(self):
        # initialize the model here
        
        # Path to OWLV2 weights folder in docker image
        model_path = "owlv2"
        processor_path = "owlv2"

        self.image_width = 1520
        self.image_height = 870

        self.target_sizes = torch.tensor([[self.image_height, self.image_width]])

        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(device)


    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model

        image = Image.open(io.BytesIO(image))

        image = np.array(image.convert("RGB"))[:, :, ::-1]

        inputs = self.processor(text=[caption], images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

            predictions = self.processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=self.target_sizes)[0]

        predictions["boxes"] = predictions["boxes"].cpu()
        predictions["scores"] = predictions["scores"].cpu()
        predictions["labels"] = predictions["labels"].cpu()

        if predictions["labels"].shape[0] > 0:
            bbox = predictions["boxes"][argmax(predictions["scores"])].to(dtype=torch.int).tolist()
            x1, y1, x2, y2 = bbox
        else:
            return [0,0,0,0]

        return [x1,y1,x2-x1,y2-y1]

if __name__ == "__main__":
    vlm_manager = VLMManager()