import json
import csv
from tqdm import tqdm
from pathlib import Path

def convert(input_dir:str, output_dir:str)->None:

    #Returns train and test CSVs to output_dir

    train_instances = []
    test_instances = []

    print(f"Reading from {input_dir}/vlm.jsonl...")

    input_dir = Path(f"{input_dir}")

    count = 0

    #Ratio of train to test data instances
    train_test_ratio = 4

    with open(input_dir / "vlm.jsonl", "r") as f:
        for line in tqdm(f):
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            for annotation in instance["annotations"]:
                x1, y1, w, h = annotation["bbox"]
                if count%train_test_ratio:
                    train_instances.append(
                        {
                            "label_name": annotation["caption"],
                            "bbox_x": x1,
                            "bbox_y": y1,
                            "bbox_width": w,
                            "bbox_height": h,
                            "image_name": instance["image"],
                            "image_width": 1520,
                            "image_height": 870,
                        }
                    )
                else:
                    test_instances.append(
                        {
                            "label_name": annotation["caption"],
                            "bbox_x": x1,
                            "bbox_y": y1,
                            "bbox_width": w,
                            "bbox_height": h,
                            "image_name": instance["image"],
                            "image_width": 1520,
                            "image_height": 870,
                        }
                    )
                count += 1

    print(f"Writing to {output_dir}...")
    #Write to csv
    with open(output_dir+"/train_annotations.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(train_instances[0].keys()))
        writer.writeheader()
        writer.writerows(train_instances)

    with open(output_dir+"/test_annotations.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(test_instances[0].keys()))
        writer.writeheader()
        writer.writerows(test_instances)

if __name__=="__main__":

    data_dir = "/home/benluo/til-24-base/data/"
    output_dir = "/home/benluo/til-24-base/vlm/Grounding-Dino-FineTuning/multimodal-data/annotation/"

    convert(data_dir, output_dir)