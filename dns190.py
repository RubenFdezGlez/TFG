import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import cv2
import os
import torch.optim as optim
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

yolo_model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
yolo_model.to(device)  # Move the model to the device

# Load image
images_path = "./datasets/train/images"
'''
dog_crops = []
for classname in os.listdir(images_path):
    for image in os.listdir(os.path.join(images_path, classname)):
        img = os.path.join(images_path, classname, image)
        img = cv2.imread(img)

        # Detect the dog
        results = yolo_model.predict(img, conf=0.5)

        # Extrae bounding boxes de perros

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if yolo_model.names[cls_id] == "dog":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = img[y1:y2, x1:x2]
                    dog_crops.append(crop)
'''
num_classes = 130
# Load the densenet model
model = models.densenet201(weights = 'DEFAULT')
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)  # Move the model to the device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets
train_dataset = ImageFolder("datasets/train/images", transform=transform)
val_dataset = ImageFolder("datasets/val/images", transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Entrenamiento
for epoch in range(20):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Test de detección de perros
test_dataset = ImageFolder("datasets/test/images", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Testear la detección de perros
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Guardar el modelo
torch.save(model.state_dict(), "dog_classifier.pth")


'''
# Clasifica cada crop
for i, crop in enumerate(dog_crops):
    input_tensor = transform(crop).unsqueeze(0)  # batch de 1
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(1).item()
        print(f"Perro #{i+1} → Raza predicha: {predicted_class}")
'''