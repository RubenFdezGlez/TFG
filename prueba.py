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
            self.data = np.zeros((len(image_paths), 256, 256, 3), dtype=np.float32)
            self.labels = np.zeros((len(image_paths), 1), dtype=np.float32)

            for i in range(len(image_paths)):
                path = image_paths[i]

                image_path = self.src_path + path
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256))
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                torch.tensor(image, dtype=torch.float32)

                label = int(path.split("-")[1][1:])
                torch.tensor(label, dtype=torch.float32)

                self.data[i] = image
                self.labels[i] = label

            self.data = torch.tensor(self.data, dtype=torch.float32)
            print(self.data.shape)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
                

    num_classes = 130

    # Identify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load a model
    model = YOLO("yolo11n.yaml").to(device)  # build from YAML and transfer weights´


    # Read file .lst and process it
    lst_file = "./data/low-resolution/train.lst"
    with open(lst_file, "r") as f:
        lines = f.readlines()

    train_image_paths = [line.strip() for line in lines]
    train_image_paths = [path[3:] for path in train_image_paths]

    # Create the train dataset and the dataloader
    train_dataset = CustomDataset(train_image_paths)

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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = torch.Tensor(model(images)).to(device)
            labels = torch.Tensor(labels).long().to(device)
            print("outputs.shape")
            print("labels.shape")
            print(outputs.shape)
            print(labels.shape)
            print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Update the learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")