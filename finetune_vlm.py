from groundingdino.util.train import load_model, load_image,train_image, train_image_batch, annotate
import cv2
import os
import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.optim as optim
from tqdm import tqdm
from vlm.src.VLMManager import VLMManager
from scoring.vlm_eval import vlm_eval
from pathlib import Path
import base64
import pandas as pd
from typing import List, Dict, Any, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model
# model = load_model("/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/weights/groundingdino_swint_ogc.pth")

# Dataset paths
images_files=sorted(os.listdir("/home/benluo/til-24-base/data/images"))
ann_file="/home/benluo/til-24-base/data/train_annotations.csv"
test_file="/home/benluo/til-24-base/data/test_annotations.csv"

vlm_manager = VLMManager()

def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 4
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    
    for index in tqdm(range(0, len(instances), batch_size)):
        predictions = []
        _instances = instances[index : index + batch_size]
        
        for instance in _instances:
            image_bytes = base64.b64decode(instance["b64"])
            predictions.append(vlm_manager.identify(image_bytes, instance["caption"]))

        _results = predictions
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "bbox": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (numpyarray): input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (str):  Input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)



def read_dataset(ann_file):
    ann_Dict= defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)  
        # Iterate over each row in the csv file
        # using reader object
        for row in ann_reader:
            #print(row)
            img_n=os.path.join("/home/benluo/til-24-base/data/images",row['image_name'])
            x1=int(row['bbox_x'])
            y1=int(row['bbox_y'])
            x2=x1+int(row['bbox_width'])
            y2=y1+int(row['bbox_height'])
            label=row['label_name']
            ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + " ."

class CustomDataset(Dataset):
    def __init__(self, ann_file="", data_dir = "/home/benluo/til-24-base/data/images"):
        ann_Dict= defaultdict(lambda: defaultdict(list))
        self.ann_list = []
        with open(ann_file) as file_obj:
            ann_reader= csv.DictReader(file_obj)  
            # Iterate over each row in the csv file
            # using reader object
            for row in ann_reader:
                #print(row)
                img_n=os.path.join(data_dir,row['image_name'])
                x1=int(row['bbox_x'])
                y1=int(row['bbox_y'])
                x2=x1+int(row['bbox_width'])
                y2=y1+int(row['bbox_height'])
                label=row['label_name']
                # self.ann_list.append({
                #     "image_path":img_n,
                #     "boxes":[[x1,y1,x2,y2]],
                #     "captions":[label]
                # })
                ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
                ann_Dict[img_n]['captions'].append(label)

        for img_path in ann_Dict:
            self.ann_list.append({
                "image_path":img_path,
                "boxes":ann_Dict[img_path]["boxes"],
                "captions":ann_Dict[img_path]["captions"],
            })
    
    def __len__(self):

        return len(self.ann_list)

    def __getitem__(self, idx):

        ori_img, img = load_image(self.ann_list[idx]["image_path"], training=True)


        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # cv2.imshow("transformed", img)

        # cv2.waitKey(0)

        # return None


        return img, ori_img.shape, self.ann_list[idx]["boxes"], self.ann_list[idx]["captions"], preprocess_caption(caption=" . ".join(set(self.ann_list[idx]["captions"])))

def collate_fn(batch):

    img_tensors, ori_shape, boxes, captions, caption_string = zip(*batch)

    img_tensors = torch.stack(img_tensors, dim=0)

    return {
        "img":img_tensors,
        "ori_shape":ori_shape,
        "boxes":boxes,
        "captions":captions,
        "caption_string":caption_string,
    }

def train(model, ann_file, epochs=1, save_path='weights/model_weights',save_epoch=1):
    # Read Dataset
    ann_Dict = read_dataset(ann_file)

    train_data = CustomDataset(ann_file, data_dir="/home/benluo/til-24-base/data/images")
    # test_data = CustomDataset(test_file, data_dir="/home/benluo/til-24-base/data/images")

    train_dataloader = DataLoader(dataset=train_data,
                                   batch_size=4,
                                   num_workers=4,
                                   shuffle=True,
                                   collate_fn=collate_fn)
    
    # test_dataloader = DataLoader(dataset=test_data,
    #                                batch_size=8,
    #                                num_workers=4,
    #                                collate_fn=collate_fn)

    # Check dataloader output    
    # for batch in custom_dataloader:
    #     print(batch)
    #     break

    input_dir = Path(f"data/")

    test_instances = []
    truths = []
    counter = 0
    test_score = 0

    with open("/home/benluo/til-24-base/data/test_vlm.jsonl", "r") as f:
        for line in f:
            # if counter > 500:
            #     break
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(input_dir / "images" / instance["image"], "rb") as file:
                image_bytes = file.read()
                for annotation in instance["annotations"]:
                    test_instances.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "b64": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    )
                    truths.append(
                        {
                            "key": counter,
                            "caption": annotation["caption"],
                            "bbox": annotation["bbox"],
                        }
                    )
                counter += 1
    
    # Add optimizer
    optimizer = optim.Adam(vlm_manager.model.parameters(), lr=1e-5, weight_decay=1e-4)

    lr_scheduler = None

    # Add Learning Rate Scheduler
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                          step_size=2,
    #                                          gamma=0.5)
    
    # Ensure the model is in training mode
    vlm_manager.model.train()

    #For evaluation
    best_score = 0

    for epoch in range(epochs):
        total_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()

            loss = train_image_batch(
                model=vlm_manager.model,
                data_dict=batch
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # print(f"Iter {idx}/{len(custom_dataloader)}, Loss: {loss.item()}")

    # for epoch in range(epochs):
    #     total_loss = 0  # Track the total loss for this epoch
    #     for idx, (IMAGE_PATH, vals) in enumerate(ann_Dict.items()):
    #         image_source, image = load_image(IMAGE_PATH)
    #         bxs = vals['boxes']
    #         captions = vals['captions']

    #         # Zero the gradients
    #         optimizer.zero_grad()
            
    #         # Call the training function for each image and its annotations
    #         loss = train_image(
    #             model=model,
    #             image_source=image_source,
    #             image=image,
    #             caption_objects=captions,
    #             box_target=bxs,
    #         )
            
    #         # Backpropagate and optimize
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item()  # Accumulate the loss
    #         print(f"Processed image {idx+1}/{len(ann_Dict)}, Loss: {loss.item()}")


        # Print the average loss for the epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        #test
        vlm_manager.model.eval()
        
        with torch.inference_mode():
            results = run_batched(test_instances)
            # calculate eval
            test_score = vlm_eval(
                [truth["bbox"] for truth in truths],
                [result["bbox"] for result in results],
            )

        vlm_manager.model.train()
        
        print(f"IoU@0.5: {test_score}, Best IoU@0.5: {best_score}")

        if test_score > best_score:
            best_score = test_score
            torch.save(vlm_manager.model.state_dict(), f"{save_path}_best.pth")

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(ann_Dict)}, IoU@0.5: {test_score}, Best IoU@0.5: {best_score}")

        # if (epoch%save_epoch)==0:
        #     # Save the model's weights after each epoch
        #     torch.save(vlm_manager.model.state_dict(), f"{save_path}{epoch}.pth")
        #     print(f"Model weights saved to {save_path}{epoch}.pth")



if __name__=="__main__":
    train(model=vlm_manager.model, ann_file=ann_file, epochs=20, save_path='/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/weights/model_weights_b2')
