from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    class CustomDataset(Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths
            self.src_path = "./data/low-resolution/"
            self.data = np.zeros((len(image_paths), 3, 256, 256), dtype=np.float32)
            self.labels = np.zeros((len(image_paths), 1), dtype=np.float32)

            for i in range(len(image_paths)):
                path = image_paths[i]

                image_path = self.src_path + path
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256))
                image = np.transpose(image, (2, 0, 1))

                label = int(path.split("-")[1][1:]) - 1

                self.data[i] = image
                self.labels[i] = label

            self.data = torch.tensor(self.data, dtype=torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]
            label = self.labels[idx]
            return image, label
                

    num_classes = 130

    # Identify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = YOLO("yolo11n.yaml").to(device)


    # Read file .lst and process it
    lst_file = "./data/low-resolution/train.lst"
    with open(lst_file, "r") as f:
        lines = f.readlines()

    train_image_paths = [line.strip() for line in lines]
    train_image_paths = [path[3:] for path in train_image_paths]

    train_image_paths = train_image_paths[:1000]  # Limit to 1000 images

    # Create the train dataset and the dataloader
    train_dataset = CustomDataset(train_image_paths)

    # Normalize the dataset
    train_dataset.data = train_dataset.data / 255.0  # Normalize to [0, 1]


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )


    # Define your optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Define your learning rate scheduler
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze(1).long().to(device)
            
            optimizer.zero_grad()

            outputs = torch.tensor(model(images))
            outputs.requires_grad = True
            
             # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Update the learning rate
        #scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")