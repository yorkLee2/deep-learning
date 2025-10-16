import yaml
import os


class_names=["car", "medium truck", "large truck"]
# Construct the dataset structure for the YAML file
dataset_structure = {
    "path": "C:/Users/hyc49/Desktop/pr2/pr2",
    "train": "C:/Users/hyc49/Desktop/pr2/pr2/images/train",
    "val": "C:/Users/hyc49/Desktop/pr2/pr2/images/val",
    "test": "",
    "names": {i: name for i, name in enumerate(class_names)}
}

# Specify the path to the output YAML file
yaml_file_path = "C:/Users/hyc49/Desktop/pr2/pr2/Data.yaml"

with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(dataset_structure, yaml_file, sort_keys=False)

print(f"YAML configuration saved to {yaml_file_path}")




