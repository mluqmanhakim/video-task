import cv2
import numpy as np

from ultralytics import YOLO
from sixdrepnet import SixDRepNet

from util import check_if_enter_store, check_if_exit_store, check_point_in_stop_zone

frame = cv2.imread("frame_620.jpg")

person_model = YOLO("yolo26x.pt")
pose_model = SixDRepNet(gpu_id=-1)
face_model = YOLO("yolov8x-face.pt")


store_points = [(220, 390), (360, 480), (770, 615)]
STOP_ZONE = np.array([[524, 180], [1120, 327], [1110, 464], [375, 255]], dtype=np.int32)

results = person_model.track(
    frame,
    persist=True,
    tracker="yolo_tracker.yaml",
    classes=[0],  # COCO class 0 = person
    verbose=False,
)

if results[0].boxes.id is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids = results[0].boxes.id.cpu().numpy().astype(int)

    # print(len(ids))

    for box, person_id in zip(boxes, ids):
        x1, y1, x2, y2 = box.astype(int)
        foot_x = int((x1 + x2) / 2)
        foot_y = int(y2)
        current_position = (foot_x, foot_y)

        # print(person_id, current_position)

        cv2.circle(frame, current_position, 5, (0, 0, 255), -1)

        inside_stop_zone = check_point_in_stop_zone(
            point=current_position, polygon=STOP_ZONE
        )

        if inside_stop_zone:
            print(f"ID-{person_id}")

            # Crop person
            person_crop = frame[
                max(0, y1) : min(frame.shape[0], y2),
                max(0, x1) : min(frame.shape[1], x2),
            ]
            if person_crop.size > 0:
                

                face_results = face_model(person_crop, conf=0.4, verbose=False)
                if len(face_results) < 1 or len(face_results[0].boxes) < 1:
                    continue

                face_box = face_results[0].boxes[0]
                face_x1, face_y1, face_x2, face_y2 = (
                    face_box.xyxy[0].cpu().numpy().astype(int)
                )

                pad_x = int((face_x2 - face_x1) * 0.2)
                pad_y = int((face_y2 - face_y1) * 0.2)

                px1 = max(0, face_x1 - pad_x)
                py1 = max(0, face_y1 - pad_y)
                px2 = min(person_crop.shape[1], face_x2 + pad_x)
                py2 = min(person_crop.shape[0], face_y2 + pad_y)

                face_crop = person_crop[py1:py2, px1:px2]

                pitch, yaw, roll = pose_model.predict(face_crop)

                cv2.imwrite(f"ID-{person_id}.jpg", person_crop)

                print("Pitch:", pitch)
                print("Yaw:", yaw)
                print("Roll:", roll)


    cv2.imwrite('frame.jpg', frame)
