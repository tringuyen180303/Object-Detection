import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import xml.etree.ElementTree as ET
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.annotations_dir = os.path.join(root, 'Annotations')
        self.root = root
        self.images_dir = os.path.join(root, 'JPEGImages')
        self.transform = transform
        self.image_files = sorted(os.listdir(self.images_dir))
        self.annotation_files = sorted(os.listdir(self.annotations_dir))
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self._label_to_int = {cls: idx for idx, cls in enumerate(self.classes)}
    def __len__(self):
        return len(self.annotations_dir)
    
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Load annotation
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        boxes, labels = self.parse_voc_xml(annotation_path)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)


        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, labels, boxes
    
    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in self._label_to_int:
                continue
            label = self._label_to_int[name]

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels

# dataset = MyDataset(root="VOC/VOC2012_train_val/VOC2012_train_val", transform=None)
# image, target = dataset[0]

# print("Image shape:", image.size)  # (width, height)
# print("Bounding boxes:", target["boxes"])
# print("Labels:", target["labels"])
# print("Image ID:", target["image_id"])
