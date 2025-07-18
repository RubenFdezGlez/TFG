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
import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil
import time

# Class made to follow the dataset structure of YOLO (In images --> train, val, test folders, instead of all images in the same folder)
class DatasetReorganizer:
    def __init__(self, dst_path, splits, classes_names):
        self.src_path_images = "TFG_Dataset/images"
        self.src_path_labels = "TFG_Dataset/labels"
        self.dst_path = dst_path
        self.splits = splits
        self.file_type_images = "images"
        self.file_type_labels = "labels"
        self.classes_names = classes_names
        
    def reorganize(self):
        # Create destination directories if they don't exist
        for split in self.splits.keys():
            os.makedirs(os.path.join(self.dst_path, split), exist_ok=True)

        # Iterate through the source directory and move files to destination directories
        for class_folder in os.listdir(self.src_path_images):
            class_path_images = os.path.join(self.src_path_images, class_folder)
            class_path_labels = os.path.join(self.src_path_labels, class_folder)

            if not os.path.isdir(class_path_images) or not os.path.isdir(class_path_labels):
                continue

            class_name = class_folder.split("-")[-1].lower()
            for char in '_ ':
                class_name = class_name.replace(char, '')

            # Remove unwanted files 
            for img in os.listdir(class_path_images):
                if img.startswith('image'):
                    os.remove(os.path.join(class_path_images, img))


            # Shuffle files for randomization
            files_images = os.listdir(class_path_images)
            files_images.sort()
            random.seed(33) 
            random.shuffle(files_images)

            files_labels = os.listdir(class_path_labels)
            files_labels.sort()
            random.seed(33) 
            random.shuffle(files_labels)

            # Split files into the splits
            total = len(files_images)
            train_end = int(splits['train'] * total)
            val_end = train_end + int(splits['val'] * total)

            split_imgs = {
                'train': files_images[:train_end],
                'val': files_images[train_end:val_end],
                'test': files_images[val_end:]
            }

            split_labels = {
                'train': files_labels[:train_end],
                'val': files_labels[train_end:val_end],
                'test': files_labels[val_end:]
            }

            # Move the files to the new folders
            for (split, img_list), (split, label_list) in zip(split_imgs.items(), split_labels.items()):
                os.makedirs(os.path.join(self.dst_path, split, self.file_type_images, class_name), exist_ok=True)
                os.makedirs(os.path.join(self.dst_path, split, self.file_type_labels, class_name), exist_ok=True)
                for img, label in zip(img_list, label_list):
                    src_image = os.path.join(class_path_images, img)
                    src_label = os.path.join(class_path_labels, label)
                    dst_image = os.path.join(self.dst_path, split, self.file_type_images, class_name, img)
                    dst_label = os.path.join(self.dst_path, split, self.file_type_labels, class_name, label)
                    shutil.copy(src_image, dst_image)
                    im = Image.open(dst_image)
                    width, height = im.size 
                    prop_w = 640 / width
                    prop_h = 640 / height
                    im = im.resize((640, 640))
                    im = im.convert("RGB")
                    im.save(dst_image)

                    shutil.copy(src_label, dst_label)
                    tree = ET.parse(dst_label)
                    root = tree.getroot()

                    with open(dst_label, 'w') as file:

                        # Iterate through each object in the XML file
                        for obj in root.findall('object'):
                            # Extract class ID
                            class_name = obj.find('name').text.lower()
                            for char in '_ ':
                                class_name = class_name.replace(char, '')
                            class_id = np.where(classes_names == class_name)[0][0]

                            # Check the coordinates of the bounding box
                            headbndbox = obj.find('headbndbox')
                            bodybndbox = obj.find('bodybndbox')
                            x1_h = int(headbndbox.find('xmin').text) * prop_w
                            y1_h = int(headbndbox.find('ymin').text) * prop_h
                            x2_h = int(headbndbox.find('xmax').text) * prop_w
                            y2_h = int(headbndbox.find('ymax').text) * prop_h
                            x1_b = int(bodybndbox.find('xmin').text) * prop_w
                            y1_b = int(bodybndbox.find('ymin').text) * prop_h
                            x2_b = int(bodybndbox.find('xmax').text) * prop_w
                            y2_b = int(bodybndbox.find('ymax').text) * prop_h

                            x1 = min(x1_h, x1_b)
                            y1 = min(y1_h, y1_b)
                            x2 = max(x2_h, x2_b)
                            y2 = max(y2_h, y2_b)

                            # Convert to YOLO format (normalized coordinates)
                            x_center = (x1 + x2) / 2 / 640
                            y_center = (y1 + y2) / 2 / 640
                            w = (x2 - x1) / 640
                            h = (y2 - y1) / 640

                            file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

                    # Override the file
                    if dst_label.endswith(".jpg.xml"):
                        os.rename(dst_label, dst_label.replace(".jpg.xml", ".txt"))
                    if dst_label.endswith(".png.xml"):
                        os.rename(dst_label, dst_label.replace(".png.xml", ".txt"))
                    if dst_label.endswith(".jpeg.xml"):
                        os.rename(dst_label, dst_label.replace(".jpeg.xml", ".txt"))

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths = "train"):
        self.image_paths = image_paths + "/"
        self.src_path = "./datasets/" + image_paths + "images/"
        self.data = []

        for class_folder in os.listdir(self.src_path):

            for image_file in os.listdir(os.path.join(self.src_path, class_folder)):
                if image_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.data.append([image_file, class_folder])
                if image_file.endswith(('.txt')):
                    os.remove(os.path.join(self.src_path, class_folder, image_file))

        self.class_map = {class_name: i for i, class_name in enumerate(os.listdir(self.src_path))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_file, class_folder = self.data[idx] 
        img = cv2.imread(image_file)
        img = cv2.resize(img, (224, 224))

        class_id = self.class_map[class_folder] 
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
        class_id = torch.tensor([class_id])

        return img_tensor, class_id

if __name__ == '__main__':

    # Initial cleanup and setup
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = {
        'train': 0.7,
        'val': 0.1,
        'test': 0.2,
    }

    num_classes = 130
    classes_names = np.array([
        "shibadog", "frenchbulldog", "siberianhusky", "malamute", "pomeranian",
        "ibizanhound", "borderterrier", "airedale", "cairn", "miniaturepoodle",
        "irishsetter", "affenpinscher", "afghanhound", "otterhound", "staffordshirebullterrier",
        "norwichterrier", "lakelandterrier", "germanshepherd", "leonberg", "australianterrier",
        "tibetanterrier", "englishsetter", "welshspringerspaniel", "schipperke", "africanhuntingdog",
        "blenheimspaniel", "norfolkterrier", "curlycoatedretriever", "pembroke", "tibetanmastiff",
        "newfoundland", "filabraziliero", "bedlingtonterrier", "sussexspaniel", "greatdane",
        "irishterrier", "scotchterrier", "lhasa", "irishwolfhound", "westhighlandwhiteterrier",
        "briard", "brabancongriffo", "dhole", "bloodhound", "redbone", "norwegianelkhound",
        "flatcoatedretriever", "vizsla", "kelpie", "bluetick", "saluki", "dandiedinmont",
        "standardschnauzer", "doberman", "entlebucher", "scottishdeerhound", "wirehairedfoxterrier",
        "sealyhamterrier", "germanshorthairedpointer", "rottweiler", "bernesemountaindog",
        "blackandtancoonhound", "walkerhound", "borzoi", "whippet", "irishwaterspaniel", "kuvasz",
        "saintbernard", "mexicanhairless", "groenendael", "malinois", "bouvierdesflandres",
        "greatpyrenees", "englishfoxhound", "chesapeakebayretriever", "brittanyspaniel", "bullmastiff",
        "americanstaffordshireterrier", "dingo", "rhodesianridgeback", "silkyterrier", "boxer",
        "eskimodog", "bostonbull", "gordonsetter", "greaterswissmountaindog", "kerryblueterrier",
        "samoyed", "giantschnauzer", "softcoatedwheatenterrier", "appenzeller", "keeshond",
        "chinesecresteddog", "collie", "toyterrier", "weimaraner", "clumber", "australianshepherd",
        "italiangreyhound", "basset", "maltesedog", "miniatureschnauzer", "basenji", "blacksable",
        "canecarso", "japanesespaniel", "japanesespitzes", "oldenglishsheepdog", "bordercollie",
        "shetlandsheepdog", "cockerspaniel", "englishspringer", "beagle", "toypoodle", "komondor",
        "cardigan", "bichonfrise", "standardpoodle", "chow", "yorkshireterrier", "chineseruraldog",
        "labradorretriever", "shihtzu", "chihuahua", "pekinese", "goldenretriever", "miniaturepinscher",
        "teddy", "papillon", "pug"
    ])

    #reorganizer = DatasetReorganizer("TFG_Dataset", splits, classes_names)
    #reorganizer.reorganize()
       
    # Load the model
    model = YOLO("yolo11n").to(device)
    
    # Constructs the bounding boxes for the training dataset
    results_val = model.train(data = "./TFG_Dataset/train_highres.yaml", 
        epochs = 130, 
        device = device, # Use the GPU if available
        workers = 4, # Number of workers for data loading - TITAN XP
        imgsz = 640, # Image size for training
        batch = 12, # Batch size for training - TITAN XP
        #freeze = 5, # Freeze the first 5 layers to do transfer learning
        # lr0 = 0.001, # Initial learning rate
        # lrf=0.01, # Final learning rate
        # augment=True,
        # flipud=1.0,  # Random vertical flip
        # degrees=15,  # Random rotation degrees
        # hsv_v=0.15,  # Random brightness
        # hsv_h=0.05,  # Random hue
        # hsv_s=0.15,  # Random saturation
        # translate=0.05,  # Random translation
        patience=10,          
        cos_lr=True,
        cache=True,
        plots = True, 
        exist_ok = True, 
        save_period = 25, 
        name="hr_yf_1", 
        project = "runs/train"
    )

    '''
    metrics = model.val(data = "./datasets/train.yaml", split="test", device = device, conf = 0.5, iou = 0.5, imgsz = 640)
    # Display the results
    print("Results:")
    print(metrics.box.map)         # mAP@0.5:0.95 promedio
    print(metrics.box.map50)       # mAP@0.5 promedio
    print(metrics.box.map75)       # mAP@0.75
    print(metrics.box.maps)        # lista con mAP@0.5:0.95 por clase

    metrics = get_classification_metrics(conf_matrix, classes_names)

    # Show the classification metrics
    for class_name, values in metrics.items():
        print(f"Class: {class_name}")
        print(f"  Precision: {values['precision']:.4f}")
        print(f"  Recall: {values['recall']:.4f}")
        print(f"  F1-Score: {values['f1']:.4f}")
    print("/////////////")
    '''
    '''
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs

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

        iter += 1  # Increment iteration counter
    '''