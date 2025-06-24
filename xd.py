from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Initial cleanup
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_model = YOLO('yolo11n.pt').to(device)

    results = cls_model.predict('dddd.jpg')

    #print(results)
    print(results[0].speed)
    for result in results:
        print("&&&&")
        print(results[0].boxes)
