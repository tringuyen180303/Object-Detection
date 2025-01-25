import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import xml.etree.ElementTree as ET
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root, transform=None, max_boxes=30):
        self.annotations_dir = os.path.join(root, 'Annotations')
        self.root = root
        self.images_dir = os.path.join(root, 'JPEGImages')
        self.transform = transform
        self.image_files = sorted(os.listdir(self.images_dir))
        self.annotation_files = sorted(os.listdir(self.annotations_dir))
        self.max_boxes = max_boxes
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

        # Pad boxes and labels to the same size
        #boxes = self.pad_boxes_to_size(boxes, self.max_boxes)
        #labels = self.pad_labels_to_size(labels, self.max_boxes)
        one_hot_labels = self.one_hot_encode(labels)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, one_hot_labels, boxes
    
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

    def one_hot_encode(self, labels):
        # Number of classes (including background)
        num_classes = len(self.classes) + 1

        # Initialize a tensor of zeros
        one_hot = torch.zeros(num_classes, dtype=torch.int64)

        # Set the corresponding label index to 1
        for label in labels:
            one_hot[label] = 1

        return one_hot
    
    def pad_boxes_to_size(self, boxes, max_size):
    # If there are fewer than max_size boxes, pad with [0, 0, 0, 0]
        num_boxes = boxes.shape[0]
        if num_boxes < max_size:
            padding = torch.zeros(max_size - num_boxes, 4)  # Pad with zeros for each box (4 coordinates)
            boxes = torch.cat([boxes, padding], dim=0)  # Concatenate the boxes and the padding
        return boxes

    def pad_labels_to_size(self, labels, max_size):
        # If there are fewer than max_size labels, pad with a background label (usually 0)
        num_labels = len(labels)
        if num_labels < max_size:
            padding = torch.zeros(max_size - num_labels, dtype=torch.int64)  # Padding with background label (0)
            labels = torch.cat([labels, padding], dim=0)  # Concatenate the labels and the padding
        return labels
    
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


# Initialize dataset and DataLoader
# dataset = MyDataset(root="VOC/VOC2012_train_val/VOC2012_train_val", transform=transform)
# data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# # Test the DataLoader
# for images, labels, boxes in data_loader:
#     print(f"Image Batch Shape: {images.shape}")  # Shape: [batch_size, channels, height, width]
#     print(f"Labels: {labels}")                   # List of tensors (1D per image)
#     print(f"Boxes: {boxes}")                     # List of tensors (variable size per image)
#     break

# print("Image shape:", image.size)  # (width, height)
# print("Bounding boxes:", target["boxes"])
# print("Labels:", target["labels"])
# print("Image ID:", target["image_id"])

