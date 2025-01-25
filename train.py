from dataset import MyDataset
from model import DETR
import torchvision
import torch
from torchvision.transforms import v2
from torchvision import transforms
import torch.nn as nn
# Create the dataset

def collate_fn(batch):
    images = []
    labels = []
    boxes = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        labels.append(sample[1])  # Labels tensor
        boxes.append(sample[2])  # Bounding boxes tensor
    
    # Stack images into a single tensor
    images = torch.stack(images, dim=0)  # Shape: [batch_size, channels, height, width]

    return images, labels, boxes

if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
    train_dataset = MyDataset(root="VOC/VOC2012_train_val/VOC2012_train_val", transform=transform)
    test_dataset = MyDataset(root="VOC/VOC2012_test/VOC2012_test", transform=transform)

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DETR(num_classes=21, num_queries=30)
    model.to(device)

    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_bbox = torch.nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for id, (images, labels, boxes) in enumerate(data_loader):
            print(images)
            print(labels)
            print(boxes)
            images = images.to(device)
            labels = labels.to(device)
            boxes = boxes.to(device)
            
            # labels = torch.tensor(labels, device=device)
            
            # boxes = torch.tensor(boxes, device=device)
            
            
            logits, bbox = model(images)
            print("logit", logits)
            print("bbox", bbox)
            loss_value = criterion_class(logits, labels)
            loss_bbox = criterion_bbox(bbox, boxes)
            total_loss = loss_value + loss_bbox

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item()}")

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss_bbox = 0
            total_loss_class = 0

            for (images, labels, boxes) in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                boxes = boxes.to(device)

                logits, bbox = model(images)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                total_loss_bbox += criterion_bbox(bbox, boxes)
                total_loss_class += criterion_class(logits, labels)

            avg_loss_bbox = total_loss_bbox / total

            print(f"Accuracy: {100 * correct / total}%")
