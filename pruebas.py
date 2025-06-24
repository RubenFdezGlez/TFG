from ultralytics import YOLO
import torch
import numpy as np
import os
from PIL import Image
import random
import shutil
import time
from sklearn.metrics import classification_report, confusion_matrix

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

# Initial cleanup
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLO model
model = YOLO("./runs/train/hr_yf_1/weights/best.pt")  # Path to your trained YOLOv8 model
model.to(device)  # Move the model to the GPU if available

# Evaluación ignorando las clases (todo tratado como una sola clase)
results = model.val(
    data='./TFG_Dataset/train_highres.yaml',
    split='test',
    imgsz=640,
    device=device,
    batch=12,
    single_cls=True,  # Tratar todas las clases como una sola
    verbose=True
)

# Mostrar resultados
print(f"mAP50 (solo localización): {results.box.map50:.4f}")
print(f"mAP50-95 (solo localización): {results.box.map:.4f}")

# # Images and labels directories
# image_dir = "TFG_Dataset/test/images"
# label_dir = "TFG_Dataset/test/labels"

# # IoU threshold for detection
# IOU_THRESHOLD = 0.7

# def box_iou(box1, box2):
#     """Compute IoU between 2 bounding boxes [x1, y1, x2, y2]"""
#     xi1 = max(box1[0], box2[0])
#     yi1 = max(box1[1], box2[1])
#     xi2 = min(box1[2], box2[2])
#     yi2 = min(box1[3], box2[3])
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

#     box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
#     box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])

#     union = box1_area + box2_area - inter_area
#     return inter_area / union if union > 0 else 0

# # Metrics initialization
# true_labels = []
# pred_labels = []
# detection_tp = 0
# detection_total = 0

# for class_folder in os.listdir(image_dir):
#     cls_dir = os.path.join("./TFG_Dataset/test/images", class_folder)
#     for image in os.listdir(cls_dir):
#         img_path = os.path.join(cls_dir, image)

#         label_path = os.path.join(label_dir, class_folder, image.replace('.jpg', '.txt').replace('.png', '.txt')).replace('.jpeg', '.txt')

#         # Ground truth
#         gt_boxes = []
#         with open(label_path, "r") as f:
#             for line in f:
#                 cls, x, y, w, h = map(float, line.strip().split())
#                 gt_boxes.append((int(cls), x, y, w, h))
#         detection_total += len(gt_boxes)

#         # YOLO prediction
#         result = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)[0]
#         preds = result.boxes

#         for pred_box in preds:
#             pred_cls = int(pred_box.cls.item())
#             xyxy = pred_box.xyxy[0].cpu().numpy()

#             matched = False
#             for gt_cls, cx, cy, w, h in gt_boxes:
#                 # Convert YOLO to xyxy
#                 img_w, img_h = result.orig_shape
#                 cx *= img_w
#                 cy *= img_h
#                 w *= img_w
#                 h *= img_h
#                 x1 = cx - w/2
#                 y1 = cy - h/2
#                 x2 = cx + w/2
#                 y2 = cy + h/2
#                 gt_box = [x1, y1, x2, y2]

#                 iou = box_iou(gt_box, xyxy)
#                 if iou >= IOU_THRESHOLD:
#                     detection_tp += 1
#                     true_labels.append(gt_cls)
#                     pred_labels.append(pred_cls)
#                     matched = True
#                     break  # Don't check other ground truth boxes for this prediction

# # Resultados
# print("\n Detection:")
# detection_recall = detection_tp / detection_total if detection_total > 0 else 0
# print(f"Correct detections: {detection_tp}/{detection_total} ({detection_recall*100:.2f}%)")

# print("\n Clasification (on correct detections):")
# print(classification_report(true_labels, pred_labels, zero_division=0))
