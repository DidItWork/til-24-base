from typing import List
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

class VLMManager:
    def __init__(self):
        # initialize the model here
        
        # model_name = "Qwen/Qwen-VL-Chat-Int4"
        # model_name = "microsoft/kosmos-2-patch14-224"
        # model_path = "google/owlv2-base-patch16-ensemble"
        # processor_path = "google/owlv2-base-patch16-ensemble"
        model_path = "/workspace/models/model"
        processor_path = "/workspace/models/processor"

        self.image_width = 1520
        self.image_height = 870

        self.target_sizes = torch.tensor([[self.image_height, self.image_width]])
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True).eval()

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, device_map=device)
        self.processor = AutoProcessor.from_pretrained(processor_path)

        # self.model.save_pretrained("vlm/src/models/model")
        # self.processor.save_pretrained("vlm/src/models/processor")


    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        
        # image = Image.open(io.BytesIO(image))

        # query = self.tokenizer.from_list_format([
        #     {"image":image},
        #     {"text": f"Locate the {caption}"},
        # ])

        # prompt = f"<grounding><phrase>{caption}</phrase>"

        image = Image.open(io.BytesIO(image))

        inputs = self.processor(text=[caption], images=image, return_tensors="pt")

        # with torch.no_grad():
        #     generate_ids = self.model.generate(
        #         pixel_values=inputs["pixel_values"].to(device),
        #         input_ids=inputs["input_ids"].to(device),
        #         attention_mask=inputs["attention_mask"].to(device),
        #         image_embeds=None,
        #         image_embeds_position_mask=inputs["image_embeds_position_mask"].to(device),
        #         use_cache=True,
        #         max_new_tokens=128,
        #     )

        # generated_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

        # response, entities = self.processor.post_process_generation(generated_text)

        # response, history = self.model.chat(self.tokenizer, query=query, history=None)

        # x1, y1, x2, y2 = entities[0][-1][0]

        # x1 *= self.image_width
        # x2 *= self.image_width
        # y1 *= self.image_height
        # y2 *= self.image_height

        # print(entities[0])

        # print(caption, [int(x1), int(y1), int(x2-x1), int(y2-y1)])

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
            )
            predictions = self.processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=self.target_sizes)[0]

        bbox = predictions["boxes"][torch.argmax(predictions["scores"])].to(dtype=torch.int).tolist()
        x1, y1, x2, y2 = bbox

        # print(caption, bbox)
        # return [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        return [x1,y1,x2-x1,y2-y1]

if __name__ == "__main__":
    vlm_manager = VLMManager()