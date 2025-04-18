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
import os # For file operations (Renaming files, etc.)
import xml.etree.ElementTree as ET
from PIL import Image

if __name__ == '__main__':

    # Class to transform the XML files to YOLO format
    class XMLTransformer:
        def __init__(self, xml_file, classes_names):
            self.xml_file = xml_file
            self.classes_names = classes_names

        def transform(self):
            with open(self.xml_file, 'w') as file:
                # Create the XML tree and get the root element
                tree = ET.parse(self.xml_file)
                root = tree.getroot()

                # Extract image dimensions
                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                # Iterate through each object in the XML file
                for obj in root.findall('object'):
                    
                    # Extract class ID
                    class_name = obj.find('name').text.lower()
                    class_id = self.classes_names.index(class_name) if class_name in self.classes_names else -1

                    # Check the coordinates of the bounding box
                    bndbox = obj.find('bndbox')
                    x1 = int(bndbox.find('xmin').text)
                    y1 = int(bndbox.find('ymin').text)
                    x2 = int(bndbox.find('xmax').text)
                    y2 = int(bndbox.find('ymax').text)

                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                
                    # Write to file in YOLO format
                    file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

                    

    # Define a custom dataset class
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

    classes_names = np.array([
        "shibadog", "frenchbulldog", "siberianhusky", "malamuten", "pomeranian",
        "ibizanhound", "borderterrier", "airedale", "cairn", "miniaturepoodle",
        "irishsetter", "affenpinscher", "afghanhound", "otterhound", "staffordshirebullterrier",
        "norwichterrier", "lakelandterrier", "germanshepherd", "leonberg", "australianterrier",
        "tibetanterrier", "englishsetter", "welshspringerspaniel", "schipperke", "africanhuntingdog",
        "blenheimspaniel", "norfolkterrier", "curlycoatedretriever", "pembroke", "tibetanmastiff",
        "newfoundland", "filabrazileiro", "bedlingtonterrier", "sussexspaniel", "greatdane",
        "irishterrier", "scotchterrier", "lhasa", "irishwolfhound", "westhighlandwhiteterrier",
        "briard", "brabancongriffo", "dhole", "bloodhound", "redbone", "norwegianelkhound",
        "flatcoatedretriever", "vizsla", "kelpie", "bluetick", "saluki", "dandiedinmont",
        "standardschnauzer", "doberman", "entlebucher", "scottishdeerhound", "wirehairedfoxterrier",
        "sealyhamterrier", "germanshorthairedpointer", "rottweiler", "bernesemountaindog",
        "blackandtancoonhound", "walkerhound", "borzoi", "whippet", "irishwaterspaniel", "kuvasz",
        "saintbernard", "mexicanhairless", "groenendael", "malinois", "bouvierdesflandres",
        "greatpyrenees", "englishfoxhound", "chesapeakebayretriever", "britannyspaniel", "bullmastiff",
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

    # Identify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = YOLO("yolo11n.pt").to(device)

    # Transform the XML files to YOLO format
    for xml_file in os.listdir("./labels/Low-Annotations"):
        if xml_file.endswith(".xml"):
            transformer = XMLTransformer(xml_file, classes_names)
            transformer.transform()

    # Rename .lst files to be compatible with YOLO (In the .yaml file)
    os.rename("./data/lstfiles/train.lst", "./data/lstfiles/train.txt")
    os.rename("./data/lstfiles/validation.lst", "./data/lstfiles/validation.txt")

    # Read file paths from the .txt file (Training data)
    lst_file = "./data/lstfiles/train.txt"
    with open(lst_file, "r") as f:
        lines = f.readlines()

    train_image_paths = [line.strip() for line in lines]
    train_image_paths = [path[3:] for path in train_image_paths]

    train_image_paths = train_image_paths[:1000]  # Currently limited

    # Create the train dataset and the dataloader
    train_dataset = CustomDataset(train_image_paths)



    # Constructs the bounding boxes for the training dataset
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

    model_cls.train(model = "train.yaml", epochs = 20, batch = 64, device = device, workers = 4)