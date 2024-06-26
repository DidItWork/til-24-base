import json
import csv
from tqdm import tqdm
from pathlib import Path
from numpy import random

def convert(input_dir:str, output_dir:str)->None:

    #Returns train and test CSVs to output_dir

    train_instances = []
    test_instances = []

    print(f"Reading from {input_dir}/vlm.jsonl...")

    input_dir = Path(f"{input_dir}")

    count = 0

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
            if count%(train_test_ratio+1):
                # for annotation in instance["annotations"]:
                #     x1, y1, w, h = annotation["bbox"]
                #     train_instances.append(
                #         {
                #             "label_name": annotation["caption"],
                #             "bbox_x": x1,
                #             "bbox_y": y1,
                #             "bbox_width": w,
                #             "bbox_height": h,
                #             "image_name": instance["image"],
                #             "image_width": 1520,
                #             "image_height": 870,
                #         }
                #     )
                train_instances.append(instance)
            else:
                # for annotation in instance["annotations"]:
                #     x1, y1, w, h = annotation["bbox"]
                    # annotation["caption"] = annotation["caption"].lower()
                #     if random.rand()>0.5:
                #         annotation["caption"] = annotation["caption"].replace("black", "dark")
                #     if random.rand()>0.5:
                #         annotation["caption"] = annotation["caption"].replace("cargo", "freight")
                #     if random.rand()>0.5:
                #         annotation["caption"] = annotation["caption"].replace("commercial", "passenger")
                #     if random.rand()>0.5:
                #         annotation["caption"] = annotation["caption"].replace("missile","rocket")

                test_instances.append(instance)

                    # test_instances.append(
                    #     {
                    #         "label_name": annotation["caption"],
                    #         "bbox_x": x1,
                    #         "bbox_y": y1,
                    #         "bbox_width": w,
                    #         "bbox_height": h,
                    #         "image_name": instance["image"],
                    #         "image_width": 1520,
                    #         "image_height": 870,
                    #     }
                    # )
            count += 1

    print(f"Writing to {output_dir}...")
    #Write to csv
    # with open(output_dir+"/train_annotations.csv", "w") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=list(train_instances[0].keys()))
    #     writer.writeheader()
    #     writer.writerows(train_instances)
    
    with open(output_dir+"/train_vlm.jsonl", "w") as trainfile:
        for line in train_instances:
            json_str = json.dumps(line)
            trainfile.write(json_str+"\n")

    with open(output_dir+"/test_vlm.jsonl", "w") as testfile:
        for line in test_instances:
            json_str = json.dumps(line)
            testfile.write(json_str+"\n")
    
    # return None

    # with open(output_dir+"/test_annotations.csv", "w") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=list(test_instances[0].keys()))
    #     writer.writeheader()
    #     writer.writerows(test_instances)

if __name__=="__main__":

    data_dir = "/home/benluo/til-24-base/data/"
    output_dir = "/home/benluo/til-24-base/data"

    convert(data_dir, output_dir)