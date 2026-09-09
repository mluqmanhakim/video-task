from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

results = model.train(
    data="/Users/luqman/Documents/GitHub/video-task/clean_dataset",
    epochs=20,
    imgsz=224,
    batch=32,
    patience=10,
    device="cpu",
    workers=4,
    project="runs/apron_classifier",
    name="yolo11n",
)