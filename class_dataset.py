import json
import csv
from tqdm import tqdm
from pathlib import Path
from numpy import random

def class_to_id(caption:str)->int:

    label2id = {
        "fighter":0,
        "commercial":1,
        "missile":2,
        "drone":3,
        "helicopter":4,
        "light":5,
        "cargo":6,
        # "plane":7,
        # "aircraft":8,
    }
    
    for label in label2id:
        if label in caption:
            return label2id[label]
    
    return -1

def convert(input_dir:str, output_dir:str)->None:

    #Returns train and test CSVs to output_dir

    train_instances = []
    test_instances = []

    print(f"Reading from {input_dir}/vlm.jsonl...")

    input_dir = Path(f"{input_dir}")

    image_id = 0
    object_id = 0

    #Ratio of train to test data instances
    train_test_ratio = 3

    with open(input_dir / "vlm.jsonl", "r") as f:
        for line in tqdm(f):
            if line.strip() == "":
                continue
            
            # if count%(train_test_ratio+1)==0:
            #     test_instances.append(json.loads(line.strip()))
            # count+=1
            # continue

            instance = json.loads(line.strip())

            data_instance = {
                "image_id": image_id,
                "image": instance["image"],
                "width": 870,
                "height": 1520,
                "objects":{
                    "id":[],
                    "area":[],
                    "bbox":[],
                    "category":[]
                },
            }

            for annotation in instance["annotations"]:
                x1, y1, w, h = annotation["bbox"]
                data_instance["objects"]["id"].append(object_id)
                data_instance["objects"]["bbox"].append([1.0*x1, 1.0*y1, 1.0*w, 1.0*h]) #COCO Format
                data_instance["objects"]["category"].append(class_to_id(annotation["caption"]))
                data_instance["objects"]["area"].append(w*h)
                
                object_id+=1

            if image_id%(train_test_ratio+1):
                train_instances.append(data_instance)
            else:
                test_instances.append(data_instance)
            image_id += 1

    
    
    # return None

    print(f"Writing to {output_dir}...")
    #Write to jsonl
    # with open(output_dir+"/train_class_vlm.jsonl", "w") as trainfile:
    #     for line in train_instances:
    #         json_str = json.dumps(line)
    #         trainfile.write(json_str+"\n")

    # with open(output_dir+"/test_class_vlm.jsonl", "w") as testfile:
    #     for line in test_instances:
    #         json_str = json.dumps(line)
    #         testfile.write(json_str+"\n")

if __name__=="__main__":

    data_dir = "/home/benluo/til-24-base/data/"
    output_dir = "/home/benluo/til-24-base/data"

    convert(data_dir, output_dir)