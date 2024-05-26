from typing import List
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import io
from numpy import argmax

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_bytes):
    # load image
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    if "model" in checkpoint:
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    else:
        load_res = model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases

class VLMManager:
    def __init__(self):
        # initialize the model here
        
        # model_name = "Qwen/Qwen-VL-Chat-Int4"
        # model_name = "microsoft/kosmos-2-patch14-224"
        # model_path = "google/owlv2-base-patch16-ensemble"
        # processor_path = "google/owlv2-base-patch16-ensemble"
        # model_path = "IDEA-Research/grounding-dino-tiny"
        # processor_path = "IDEA-Research/grounding-dino-tiny"
        # weights_path = "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/weights/model_weights0.pth"

        # self.image_width = 1520
        # self.image_height = 870

        # self.target_sizes = torch.tensor([[self.image_height, self.image_width]])#

        # model_path = "/workspace/models/model"
        # processor_path = "/workspace/models/processor"

        #Grounding DINO
        config_file = "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
        checkpoint_path = "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/weights/groundingdino_swint_ogc.pth"  # change the path of the model
        self.box_threshold = 0.0
        self.text_threshold = 1.0
        self.token_spans = None
        self.cpu_only = not torch.cuda.is_available()
        
        #load model
        self.model = load_model(config_file, checkpoint_path, cpu_only=self.cpu_only)

        # self.model.save_pretrained("vlm/src/models/model")
        # self.processor.save_pretrained("vlm/src/models/processor")


    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model
        
        #Transformers owl inference pipeline

        # image = Image.open(io.BytesIO(image))

        # inputs = self.processor(text=[caption], images=image, return_tensors="pt")

        # with torch.no_grad():
        #     outputs = self.model(
        #         input_ids=inputs["input_ids"].to(device),
        #         attention_mask=inputs["attention_mask"].to(device),
        #         pixel_values=inputs["pixel_values"].to(device),
        #     )
        #     predictions = self.processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=self.target_sizes)[0]

        # if len(predictions["boxes"]):
        #     bbox = predictions["boxes"][torch.argmax(predictions["scores"])].to(dtype=torch.int).tolist()
        #     x1, y1, x2, y2 = bbox
        # else:
        #     return [0,0,0,0]

        # return [x1,y1,x2-x1,y2-y1]

        #Grounding DINO

        image_pil, image = load_image(image)

        # run model
        boxes_filt, pred_phrases = get_grounding_output(
            self.model, image, f"{caption.lower()}", self.box_threshold, self.text_threshold, cpu_only=self.cpu_only, token_spans=None
        )

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }

        # print(pred_dict)

        # for i, label in enumerate(pred_dict["labels"]):
        #     if caption not in label:
        #         pred_dict["labels"][i] = 0.0
        #     else:
        #         pred_dict["labels"][i] = float(pred_dict["labels"][i].split("(")[-1].strip(")"))

        if pred_dict["boxes"].shape[0]==0:
            return [0,0,0,0]

        x1, y1, w, h = (pred_dict["boxes"][argmax(pred_dict["labels"])]*torch.Tensor([size[0],size[1],size[0],size[1]])).tolist()

        # print(pred_dict["labels"][argmax(pred_dict["labels"])], argmax(pred_dict["labels"]))

        # print(int(x1-w/2),int(y1-h/2),int(w),int(h))

        return [int(x1-w/2),int(y1-h/2),int(w),int(h)]

if __name__ == "__main__":
    vlm_manager = VLMManager()