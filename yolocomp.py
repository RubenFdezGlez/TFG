from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Initial cleanup
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load a pretrained YOLOv8 model and move it to the gpu if available
    # det_model = YOLO('yolo11n.pt').to(device)

    # Train the model to identify better some dog species
    # det_results = det_model.train(
    #     data = './TFG_Dataset/train_hr_dns.yaml',  # Path to the dataset configuration file
    #     epochs = 20,  # Number of training epochs
    #     device = device, # Use the GPU if available
    #     workers = 4, # Number of workers for data loading - TITAN XP
    #     imgsz = 640, # Image size for training
    #     batch = 12, # Batch size for training - TITAN XP
    #     single_cls = True,  # Single class training
    # )

    # Load images
    # images_path = "./TFG_Dataset/test/images"
    # testYOLOModel(yolo_model, images_path, device)

    torch.cuda.empty_cache()

    cls_model = YOLO('yolo11n-cls').to(device)

    cls_results = cls_model.train(
        data="./TFG_Dataset/images",  # Path to the dataset configuration file
        epochs=25,  # Number of training epochs
        device=device,  # Use the GPU if available
        workers=4,  # Number of workers for data loading - TITAN XP
        imgsz=640,  # Image size for training
        batch=12,  # Batch size for training - TITAN XP
        freeze=5,
        plots=True,
        exist_ok=True,
        save_period=25,
        cache=True,
        deterministic=True,
        name="yolo11n_detcls_final",
        project="runs/train"
    )

    # Validate the model
    metrics = cls_model.val()  # no arguments needed, uses the dataset and settings from training
    metrics.top1  # top1 accuracy
    metrics.top5  # top5 accuracy