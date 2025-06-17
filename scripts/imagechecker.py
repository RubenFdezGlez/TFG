import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil
import time

# Make a program that checks if the images size is correct comparing them with the labels

def check_image_sizes(image_dir, label_dir):
    for dir in os.listdir(image_dir):
        for label_file in os.listdir(os.path.join(label_dir, dir)):
            if label_file.endswith(".xml"):
                # Get the corresponding image file
                image_file = label_file.replace(".xml", "")
                image_path = os.path.join(image_dir, dir, image_file)
                label_path = os.path.join(label_dir, dir, label_file)

                if not os.path.exists(image_path):
                    print(f"Image not found for label: {label_file}")
                    continue

                # Check the image size
                image = Image.open(image_path)
                width, height = image.size

                # Parse the XML label file
                tree = ET.parse(label_path)
                root = tree.getroot()
                for size in root.findall("size"):
                    label_width = int(size.find("width").text)
                    label_height = int(size.find("height").text)
                    if (width, height) != (label_width, label_height):
                        print(f"Size mismatch for {os.path.join(dir, image_file)}: {width}x{height} != {label_width}x{label_height}")

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

                            label_width = int(root.find('size/width').text)
                            label_height = int(root.find('size/height').text)

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

                            if label_width != width or label_height != height:
                                file.write(f"{class_id} {y_center} {x_center} {h} {w}\n") 
                            else:
                                file.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

                    # Override the file
                    if dst_label.endswith(".jpg.xml"):
                        os.rename(dst_label, dst_label.replace(".jpg.xml", ".txt"))
                    if dst_label.endswith(".png.xml"):
                        os.rename(dst_label, dst_label.replace(".png.xml", ".txt"))
                    if dst_label.endswith(".jpeg.xml"):
                        os.rename(dst_label, dst_label.replace(".jpeg.xml", ".txt"))

if __name__ == "__main__":
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

    reorganizer = DatasetReorganizer("TFG_Dataset", splits, classes_names)
    reorganizer.reorganize()

    image_directory = "./TFG_Dataset/images"
    label_directory = "./TFG_Dataset/labels"
    check_image_sizes(image_directory, label_directory)
    print("Image size check completed.")
