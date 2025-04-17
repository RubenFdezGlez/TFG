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
                #torch.tensor(image, dtype=torch.float32)

                label = int(path.split("-")[1][1:]) - 1
                #torch.tensor(label, dtype=torch.float32)

                self.data[i] = image
                self.labels[i] = label

            self.data = torch.tensor(self.data, dtype=torch.float32)
            print(self.data.shape)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            #image = self.data[idx].permute(2, 0, 1)  # Reorder dimensions to [channels, height, width]
            label = self.labels[idx]
            return image, label
                

    num_classes = 130

    # Identify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = YOLO("yolo11n.pt").to(device)


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

    results = model(train_dataset.data, augment=True, verbose=True)

    src_bboxes = "./bboxes/" 

    dog_crops = []  # List to store cropped dog images
    iter = 0  # Initialize iteration counter

    # Process results list
    for result in results:
        for box in result.boxes:
            class_id = box.cls  # Class ID of the detected object
            confidence = box.conf  # Confidence score of the detection

            if model.names[class_id] == "dog" and confidence > 0.5:
                print(f"Detected a dog with confidence: {confidence:.2f}")
                x1, y1, x2, y2 = box.xyxy  # Bounding box coordinates
                
                dog_crop = image[int(y1):int(y2), int(x1):int(x2)]  # Crop the dog from the image
                dog_crops.append(dog_crop)  # Append the cropped dog image to the list


            #boxes = result.boxes  # Boxes object for bounding box outputs
            #masks = result.masks  # Masks object for segmentation masks outputs
            #keypoints = result.keypoints  # Keypoints object for pose outputs
            #probs = result.probs  # Probs object for classification outputs
            #obb = result.obb  # Oriented boxes object for OBB outputs
            #result.show()  # display to screen
            result.save(filename = src_bboxes + "-num-" + i )  # save to disk

        iter += 1  # Increment iteration counter

    '''
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
        '''