from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
print(model.info())