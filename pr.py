from ultralytics import YOLO
import torch
import cv2
import numpy as np

if __name__ == '__main__':
    # Initial cleanup
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    det_model = YOLO('./runs/detect/train9/weights/best.pt').to(device)
    cls_model = YOLO('./runs/train/cls_11_1/weights/best.pt').to(device)

    # Test the joint model on a sample image
    img = cv2.imread("TFG_Dataset/test/images/affenpinscher/n107858.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection
    det_results = det_model.predict(img_rgb, device=device, conf=0.5)

    for box in det_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the detected dog
        crop = img_rgb[y1:y2, x1:x2]

        # Perform classification
        cls_results = cls_model.predict(crop, device=device)
        pred_class = cls_results[0].probs.top1
        raza = cls_results[0].names[pred_class].split('-')[2]

        # Dibujar caja con nombre de raza (en lugar de "dog")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, raza, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)

    # Guardar resultado
    cv2.imwrite("output.jpg", img)