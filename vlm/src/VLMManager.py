from typing import List
from transformers import AutoProcessor, GroundingDinoForObjectDetection, Owlv2ForObjectDetection, AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import io
from numpy import argmax
import numpy as np
# import albumentations as A
# import supervision as sv

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict, annotate
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

device = "cuda" if torch.cuda.is_available() else "cpu"

# def load_image(image_bytes):
#     # load image
#     image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # load image

#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image, _ = transform(image_pil, None)  # 3, h, w
#     return image_pil, image

def load_image(image_bytes, training=True):
    # if training:
    # transform = A.Compose(
    #     [
    #         # A.GaussNoise(var_limit=(800.0, 900.0),p=1.0),
    #         # A.GaussNoise(var_limit=(0.0, 3000.0),p=1.0),
    #         # A.Blur(blur_limit=3, p=0.2),
    #         # A.HorizontalFlip(p=0.5),
    #     ]
    # )
    
    torch_transforms = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # image_source = Image.open(image_path).convert("RGB")
    image_source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # image = np.ones((1520,870,3))
    image = np.asarray(T.RandomResize([870], max_size=1520)(image_source)[0])
    # image_transformed, _ = torch_transforms(transform(image=image)["image"], None)
    image_transformed, _ = torch_transforms(image, None)
    # image_transformed = transform(image=image)["image"]
    return image_source, image_transformed



def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    print(model)
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
        model_path = "google/owlv2-base-patch16-ensemble"
        processor_path = "google/owlv2-base-patch16-ensemble"
        model_path = "/home/benluo/til-24-base/checkpoint-4500"
        processor_path = "/home/benluo/til-24-base/checkpoint-4500"
        # weights_path = "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/weights/model_weights0.pth"

        self.image_width = 1520
        self.image_height = 870

        self.target_sizes = torch.tensor([[self.image_height, self.image_width]])#

        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(device)

        #HuggingFace Grounding DINO

        # self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # self.model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

        
        # Grounding DINO
        # config_file = "/home/benluo/til-24-base/vlm/src/GroundingDINO_SwinB_cfg.py"  # change the path of the model config file
        # checkpoint_path = "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/weights/groundingdino_swinb_cogcoor.pth"  # change the path of the model
        # # config_file = "GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
        # # checkpoint_path = "model_weights.pth"  # change the path of the model
        # self.box_threshold = 0.35
        # self.text_threshold = 0.25
        # self.token_spans = None
        # self.cpu_only = not torch.cuda.is_available()
        
        # # #load model
        # self.model = load_model(config_file, checkpoint_path, cpu_only=self.cpu_only).to(device)

        # print(self.model)

        # Do not train image backbone
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False

        # for param in self.model.transformer.encoder.layers.parameters():
        #     param.requires_grad = False
        
        # for param in self.model.transformer.decoder.layers.parameters():
        #     param.requires_grad = False
        
        # for module in self.model.transformer.decoder.layers:
        #     for param in module.ca_text.parameters():
        #         param.requires_grad = True
        #     for param in module.catext_dropout.parameters():
        #         param.requires_grad = True
        #     for param in module.catext_norm.parameters():
        #         param.requires_grad = True
        
        # print(self.model)
        
        # print([param.requires_grad for param in self.model.backbone.parameters()])

        # self.model.eval()

        # self.model.save_pretrained("vlm/src/models/model")
        # self.processor.save_pretrained("vlm/src/models/processor")


    def identify(self, image: bytes, caption: str) -> List[int]:
        # perform object detection with a vision-language model

        #HuggingFace Grounding DINO

        # image = Image.open(io.BytesIO(image))

        # # print(caption)
        
        # inputs = self.processor(images=image, text=caption, return_tensors="pt").to(device)
        
        # with torch.inference_mode():
        #     outputs = self.model(**inputs)

        #     # convert outputs (bounding boxes and class logits) to COCO API
        #     target_sizes = torch.tensor([image.size[::-1]])
        #     results = self.processor.image_processor.post_process_object_detection(
        #         outputs, threshold=0.1, target_sizes=target_sizes
        #     )[0]

        # if results["boxes"].shape[0] == 0:
        #     return [0,0,0,0]

        # x1, y1, x2, y2 = [round(i, 1) for i in results["boxes"][0].tolist()]
        # # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # #     box = [round(i, 1) for i in box.tolist()]
        # #     print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")
        # return [x1, y1, x2-x1, y2-y1]
        #Transformers owl inference pipeline

        image = Image.open(io.BytesIO(image))

        inputs = self.processor(text=[[caption]], images=image, return_tensors="pt").to(device)

        # print(inputs["input_ids"].shape)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # print(outputs)
            predictions = self.processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=self.target_sizes)[0]

        if len(predictions["boxes"]):
            bbox = predictions["boxes"][torch.argmax(predictions["scores"])].to(dtype=torch.int).tolist()
            x1, y1, x2, y2 = bbox
        else:
            return [0,0,0,0]

        return [x1,y1,x2-x1,y2-y1]

        #Grounding DINO

        image_pil, image = load_image(image)

        # run model
        boxes_filt, logits, pred_phrases = predict(
            self.model, image, f"{caption.lower()} .", self.box_threshold, self.text_threshold, "cuda", True
        )

        # print(caption, logits, pred_phrases)

        phrase_lengths = torch.tensor([len(pred_phrase) for pred_phrase in pred_phrases])

        # visualize pred
        size = image_pil.size
        # pred_dict = {
        #     "boxes": boxes_filt,
        #     "size": [size[1], size[0]],  # H,W
        #     "labels": pred_phrases,
        # }

        # print(pred_dict)

        # for i, label in enumerate(pred_dict["labels"]):
        #     if caption not in label:
        #         pred_dict["labels"][i] = 0.0
        #     else:
        #         pred_dict["labels"][i] = float(pred_dict["labels"][i].split("(")[-1].strip(")"))

        if boxes_filt.shape[0]==0:
            return [0,0,0,0]

        x1, y1, w, h = (boxes_filt[argmax(logits*phrase_lengths)]*torch.Tensor([size[0],size[1],size[0],size[1]])).tolist()


        # print(pred_dict["labels"][argmax(pred_dict["labels"])], argmax(pred_dict["labels"]))

        # annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes_filt, logits=logits, phrases=pred_phrases)

        # sv.plot_image(annotated_frame, (16, 16))

        # print(int(x1-w/2),int(y1-h/2),int(w),int(h))

        return [int(x1-w/2),int(y1-h/2),int(w),int(h)]

if __name__ == "__main__":
    vlm_manager = VLMManager()