from dataset import MyDataset
from model import DETR
import torchvision
import torch
from torchvision.transforms import v2
import torch.nn as nn
# Create the dataset
transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = MyDataset(root="VOC/VOC2012_train_val/VOC2012_train_val", transform=None)
test_dataset = MyDataset(root="VOC/VOC2012_test/VOC2012_test", transform=transform)

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Create the model
model = DETR(num_classes=21)
model.to("cuda")

epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion_class = torch.nn.CrossEntropyLoss()
criterion_bbox = torch.nn.L1Loss()

# Training loop
for epoch in range(epochs):
    model.train()
    for id, (images, labels, boxes) in enumerate(data_loader):
        images = images.to("cuda")
        labels = labels.to("cuda")
        boxes = boxes.to("cuda")

        optimizer.zero_grad()
        logits, bbox = model(images)
        loss_value = criterion_class(logits, labels)
        loss_bbox = criterion_bbox(bbox, boxes)
        total_loss = loss_value + loss_bbox

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
            images = images.to("cuda")
            labels = labels.to("cuda")
            boxes = boxes.to("cuda")

            logits, bbox = model(images)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            total_loss_bbox += criterion_bbox(bbox, boxes)
            total_loss_class += criterion_class(logits, labels)

        avg_loss_bbox = total_loss_bbox / total

        print(f"Accuracy: {100 * correct / total}%")
