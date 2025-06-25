from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Initial cleanup
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_model = YOLO('yolov11n_cls.pt').to(device)

    results = cls_model.predict('dog2.jpg')

    for result in results:
        print("&&&&")
        #print(results[0].plot())
        print(results[0].names[results[0].probs.top1])
        #results[0].show()

