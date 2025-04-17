'''
   @author: Ruben Fernandez Gonzalez 
'''

# Import necessary libraries
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
            self.src_path = "./data/train/"
            self.data = np.zeros((len(image_paths), 3, 256, 256), dtype=np.float32)
            self.labels = np.zeros((len(image_paths), 1), dtype=np.float32)

            for i in range(len(image_paths)):
                path = image_paths[i]

                image_path = self.src_path + path
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (256, 256))
                image = np.transpose(image, (2, 0, 1))
                image = image / 255.0  # Normalize the image to [0, 1]

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
    model = YOLO("yolo11n.pt").to(device)

    # Read file .lst and process it
    lst_file = "./data/train/train.lst"
    with open(lst_file, "r") as f:
        lines = f.readlines()

    train_image_paths = [line.strip() for line in lines]
    train_image_paths = [path[3:] for path in train_image_paths]

    train_image_paths = train_image_paths[:1000]  # Limit to 1000 images


    # Create the train dataset and the dataloader
    train_dataset = CustomDataset(train_image_paths)


    results = model(train_dataset.data, augment=True, verbose=True)

    src_bboxes = "./bboxes/" 

    dog_crops = []  # List to store cropped dog images
    iter = 0  # Initialize iteration counter

    # Process results list
    for result in results:
        for box in result.boxes:
            class_name = int(box.cls[0])  # Class ID of the detected object
            confidence = float(box.conf[0])  # Confidence score of the detection

            if model.names[class_name] == "dog" and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                
                image = result.orig_img  # Original image
                dog_crop = image[y1:y2, x1:x2]  # Crop the dog from the image
                dog_crops.append(dog_crop)  # Append the cropped dog image to the list

                # Save the cropped dog image (optional)
                #file_name = src_bboxes + "dog_" + str(iter) + ".jpg"
                #result.save(filename = file_name)  # Save the entire image with all the bounding boxes
                #cv2.imwrite(file_name, dog_crop)
        
        #boxes = result.boxes  # Boxes object for bounding box outputs
        #masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        #obb = result.obb  # Oriented boxes object for OBB outputs


        iter += 1  # Increment iteration counter

    # Create a DataLoader for the training dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
    )
    
    # Load Yolo11n model for classification
    model_cls = YOLO("yolo11n-cls.yaml").to(device)

    # Define the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cls.parameters(), lr=0.001, weight_decay=0.0001)


    # Training loop
    num_epochs = 20
    model_cls.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze(1).long().to(device)
            
            optimizer.zero_grad()

            outputs = model_cls(images)
            
            # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")