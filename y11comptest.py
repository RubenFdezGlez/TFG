from ultralytics import YOLO
import torch
import time
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    # Initial cleanup
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    det_model = YOLO('runs/yolo11det/1/weights/best.pt').to(device)

    # Train the model to identify better some dog species
    # det_results = det_model.train(
    #     data = './TFG_Dataset/train_hr_dns.yaml',  # Path to the dataset configuration file
    #     epochs = 20,  # Number of training epochs
    #     device = device, # Use the GPU if available
    #     workers = 4, # Number of workers for data loading - TITAN XP
    #     imgsz = 640, # Image size for training
    #     batch = 12, # Batch size for training - TITAN XP
    #     single_cls = True,  # Single class training
    #     name="1", 
    #     project = "runs/yolo11det"
    # )

    # Get the performance and detection results on the test set
    det_results = det_model.val(
        data='./TFG_Dataset/train_hr_dns.yaml',  # Path to the dataset configuration
        split='test',  # Use the test split for validation
        imgsz=640,  # Image size for validation
        device=device,  # Use the GPU if available
        workers = 4, # Number of workers for data loading - TITAN XP
        batch = 12, # Batch size for training - TITAN XP
    )
    time_elapsed = det_results.speed['preprocess'] + det_results.speed['inference'] + det_results.speed['postprocess']
    print(f"Test set completed in {time_elapsed:.2f} seconds.")
    print(f"Time taken to process each image: {time_elapsed / len(os.listdir('./TFG_Dataset/test/images')):.2f} seconds")

    # Print the results
    print("Detection Results:")
    print(f"mAP50: {det_results.box.map50}")
    print(f"mAP50-95: {det_results.box.map}")



    torch.cuda.empty_cache()
    #cls_model = YOLO('./runs/train/yolo11n_detcls_final/weights/best.pt').to(device)
    cls_model = YOLO('./runs/train/hr_yf_1/weights/best.pt').to(device)

    # torch.cuda.empty_cache()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    total_time = 0
    for class_name in os.listdir("./TFG_Dataset/test/images"):
        class_dir = os.path.join("./TFG_Dataset/test/images", class_name)
        class_id = np.where(classes_names == class_name)[0][0]

        if os.path.isdir(class_dir):
            for image in os.listdir(class_dir):
                img_path = os.path.join(class_dir, image)

                results_pred = cls_model(img_path, verbose=False, device=device)

                total_time += results_pred[0].speed['preprocess'] + results_pred[0].speed['inference'] + results_pred[0].speed['postprocess']

                #class_id_pred = results_pred[0].probs.top1
                class_id_pred = results_pred[0].boxes.cls.cpu().numpy()[0] if results_pred[0].boxes.cls.size() > 0 else -1
                class_name_pred = results_pred[0].names[class_id_pred].split("-")[-1].lower()
                for char in '_ ':
                    class_name_pred = class_name_pred.replace(char, '')

                if class_name_pred == class_name:
                    confusion_matrix[class_id, class_id] += 1
                else:
                    confusion_matrix[class_id, np.where(classes_names == class_name_pred)[0][0]] += 1

        #print(f"Processed class: {class_name}, Confusion Matrix Row: {confusion_matrix[class_id]}")

print(f"Time taken to process the test dataset: {total_time:.2f} seconds")
print(f"Time taken to process each image: {total_time / len(os.listdir('./TFG_Dataset/test/images')):.2f} seconds")

overall_precision = 0
overall_recall = 0
overall_f1_score = 0

# Calculate the precision, recall, and F1-score for each class and overall
for i in range(num_classes):
    tp = confusion_matrix[i, i]
    fp = confusion_matrix[:, i].sum() - tp
    fn = confusion_matrix[i, :].sum() - tp
    print(f"Class {i}: TP={tp}, FP={fp}, FN={fn}")

    class_name = classes_names[i]
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Class: {class_name}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    overall_precision += precision
    overall_recall += recall
    overall_f1_score += f1_score

# Average the overall metrics
overall_precision /= num_classes
overall_recall /= num_classes
overall_f1_score /= num_classes

print("\nOverall Metrics:")
print(f"Overall Precision: {overall_precision:.4f}")
print(f"Overall Recall: {overall_recall:.4f}")
print(f"Overall F1-Score: {overall_f1_score:.4f}")

print("\nConfusion Matrix:")
df_cfm = pd.DataFrame(confusion_matrix, index=classes_names, columns=classes_names)
plt.figure(figsize=(30, 22.5))
cfm_plot = sn.heatmap(df_cfm, cmap='viridis', vmin=0, vmax=300)
cfm_plot.figure.savefig("cfm.png")