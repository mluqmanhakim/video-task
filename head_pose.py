from sixdrepnet import SixDRepNet
import cv2
from ultralytics import YOLO

frame = cv2.imread("output-2.jpg")

pose_model = SixDRepNet(gpu_id=-1)
face_model = YOLO("yolov8x-face.pt")

results = face_model(
    frame,
    conf=0.4,
    verbose=False
)
box = results[0].boxes[0]
x1, y1, x2, y2 = (
    box.xyxy[0]
    .cpu()
    .numpy()
    .astype(int)
)

pad_x = int((x2 - x1) * 0.2)
pad_y = int((y2 - y1) * 0.2)

px1 = max(0, x1 - pad_x)
py1 = max(0, y1 - pad_y)
px2 = min(frame.shape[1], x2 + pad_x)
py2 = min(frame.shape[0], y2 + pad_y)

face_crop = frame[py1:py2, px1:px2]


pitch, yaw, roll = pose_model.predict(face_crop)

print("Pitch:", pitch)
print("Yaw:", yaw)
print("Roll:", roll)

        

# cv2.imshow("Face", face_crop)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

