from ultralytics import YOLO
import cv2


frame = cv2.imread('frame_1.png')

model = YOLO("yolo26m.pt")


results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    classes=[0], # COCO class 0 = person
    verbose=False
)


if results[0].boxes.id is not None:

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    print(ids)