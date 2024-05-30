from typing import List
import tempfile
import os


# YOLO world
from ultralytics import YOLOWorld



def bytes_to_file(image_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        return temp_file.name

class VLMManager:
    def __init__(self):
        # initialize the model here
        # self.model = YOLOWorld("/home/joozy/code/hackathon/til-24-base/vlm/src/models/yolov8x-worldv2-ft.pt")
        self.model = YOLOWorld("./models/yolov8x-worldv2-ft.pt")


    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        temp_image_path = bytes_to_file(image)

        # Define custom classes
        self.model.set_classes([caption])

        # Execute prediction for specified categories on an image
        results = self.model.predict(temp_image_path)

        # delete temp file
        os.remove(temp_image_path)

        if not results:
            return [0,0,0,0]
        if not results[0].boxes.xywh.numel():
            return [0,0,0,0]

        x1, y1, w, h = [round(num) for num in results[0].boxes.xywh[0].tolist()]

        return [int(x1-w/2),int(y1-h/2),int(w),int(h)]

if __name__ == "__main__":
    vlm_manager = VLMManager()