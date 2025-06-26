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
import time

def testYOLOModel(model, images_path, device):
    dog_crops = []
    total_images = sum(len(files) for _, _, files in os.walk(images_path))

    start_time_det = time.time()
    for classname in os.listdir(images_path):
        for image in os.listdir(os.path.join(images_path, classname)):
            img = os.path.join(images_path, classname, image)
            img = cv2.imread(img)

            # Detect the dog
            results = model.predict(img, conf=0.5, device=device, verbose=False)

            # Extract the bounding boxes and crop the images
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img[y1:y2, x1:x2]
                dog_crops.append(crop)

    end_time_det = time.time()
    print(f"Tiempo total de detección: {end_time_det - start_time_det:.2f} segundos")
    print(f"Tiempo de detección por imagen: {(end_time_det - start_time_det) / total_images:.4f} segundos")


# Initial cleanup
torch.cuda.empty_cache()
num_classes = 130
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pretrained YOLOv8 model and move it to the gpu if available
yolo_model = YOLO('yolov8n.pt').to(device)

# Train the model to identify better some dog species
# results = yolo_model.train(
#     data = './TFG_Dataset/train_hr_dns.yaml',  # Path to the dataset configuration file
#     epochs = 20,  # Number of training epochs
#     device = device, # Use the GPU if available
#     workers = 4, # Number of workers for data loading - TITAN XP
#     imgsz = 640, # Image size for training
#     batch = 12, # Batch size for training - TITAN XP
#     single_cls = True,  # Single class training
# )

# Load images
# images_path = "./TFG_Dataset/test/images"
# testYOLOModel(yolo_model, images_path, device)

torch.cuda.empty_cache()

# Load the densenet model
model = models.densenet201(weights = 'DEFAULT')
dropout_rate = 0.2
model.classifier = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.classifier.in_features, num_classes)
)
model = model.to(device)  # Move the model to the device

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the datasets for training and validation
train_dataset = ImageFolder("TFG_Dataset/train/images", transform=transform_train)
val_dataset = ImageFolder("TFG_Dataset/val/images", transform=transform)

# Create the data loaders 
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True, num_workers=4)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

best_val_loss = float('inf')
early_stop_counter = 0

# Training + validation loop
for epoch in range(35):
    # Training
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

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Train loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "dog_classifier.pth")
    
    print(f"Epoch {epoch+1}, Validation loss: {val_loss:.4f}")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Calcular el tiempo de pruebas total
start_time_cls = time.time()

# Test de detección de perros
test_dataset = ImageFolder("TFG_Dataset/test/images", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=4)

# Testear la detección de perros
all_preds = []
all_labels = []
val_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, zero_division=0.0))

# Calcular el tiempo de pruebas total
end_time_cls = time.time()
print(f"Tiempo total de pruebas: {end_time_cls - start_time_cls:.2f} segundos")
# Calcular el tiempo de pruebas por imagen
test_time_per_image = (end_time_cls - start_time_cls) / len(test_dataset)
print(f"Tiempo de pruebas por imagen: {test_time_per_image:.4f} segundos")

'''
# Clasifica cada crop
for i, crop in enumerate(dog_crops):
    input_tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(1).item()
        print(f"Perro #{i+1} → Raza predicha: {predicted_class}")

# Guardar los crops en una carpeta
output_dir = "dog_crops"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
'''