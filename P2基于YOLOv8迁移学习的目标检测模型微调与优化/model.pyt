import pandas as pd
import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image

# 数据集路径
dir_path = "C:/Users/hyc49/Desktop/pr2/pr2"



def convert_labels_to_yolo_format(image_id,class_id, cx, cy, w, h, img_width=960, img_height=540):
    """转换标签到 YOLO 格式 (归一化中心坐标 + 宽高)"""
    class_id -= 1  # YOLO 类别索引从 0 开始
    return [class_id, cx / img_width, cy / img_height, w / img_width, h / img_height]

train_label_path = dir_path+'/train/labels.txt'
# Define the path to val labels.txt file
val_label_path = dir_path+'/val/labels.txt'

def parse_labels(labels_path,output_dir):
    annotations = {}
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_id, class_id, cx, cy, w, h = map(float, parts)  # Convert all to float for processing
            image_name = f"{int(image_id):05}.jpeg"  # Adjust the format
            if image_name not in annotations:
                annotations[image_name] = []
            annotations[image_name].append(convert_labels_to_yolo_format(image_id,int(class_id), cx, cy, w, h))

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    i=0
    # Write annotations to individual files
    for image_name, labels in annotations.items():
        i+=1
        base_name = os.path.splitext(image_name)[0]  # Remove the extension from image_name
        txt_file_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_file_path, 'w') as file:
            for label in labels:
                # Assuming convert_labels_to_yolo_format returns a list that needs to be joined into a string
                line = ' '.join(map(str, label)) + '\n'
                file.write(line)
        if i %100==0:
          print(f" {i} labels completed")



output_dir_train = dir_path+"/labels/train"
output_dir_val = dir_path+"/labels/val"
parse_labels(train_label_path,output_dir_train)
print(f'All {len(os.listdir(output_dir_train))} train labels has been parsed to {output_dir_train}')
parse_labels(val_label_path,output_dir_val)
print(f'All {len(os.listdir(output_dir_val))} val labels has been parsed to {output_dir_val}')



#Building the YOLO format

def copy_images(src_dir, dst_dir, format='JPEG'):
    """
    Copies images from the source directory to the destination directory.

    Parameters:
    - src_dir: The directory containing the original images.
    - dst_dir: The directory where the images will be saved.
    - format: The format in which to save the images. Defaults to 'JPEG'.
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    i=0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            # Load the image
            image = Image.open(src_path)

            # Save the image to the new directory, optionally converting the format
            image.save(dst_path, format=format)
            i+=1
        if i %100==0:
          print(f" {i} images completed")



src_dir_train = dir_path+'/train/images'
dst_dir_train = dir_path+'/images/train'

src_dir_val = dir_path+'/val/images'
dst_dir_val = dir_path+'/images/val'


copy_images(src_dir_train, dst_dir_train)
print(f'{len(os.listdir(dst_dir_train))} Train images have been copied into {dst_dir_train}')
copy_images(src_dir_val, dst_dir_val)
print(f'{len(os.listdir(dst_dir_val))} Val images have been copied into {dst_dir_val}')
