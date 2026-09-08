from collections import defaultdict, deque
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sixdrepnet import SixDRepNet
from ultralytics import YOLO

from util import (
    detect_face,
    detect_head_pose,
)

current_dir = Path.cwd()
config_path = current_dir.parents[0] / "config" / "task2.json"
model_dir = current_dir.parents[0] / "model"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

yolo_model_path = model_dir / config["yolo_model_filename"]
face_model_path = model_dir / config["face_model_filename"]
pose_model_path = model_dir / config["pose_model_filename"]
yolo_model = YOLO(model=yolo_model_path)
face_model = YOLO(model=face_model_path)
pose_model = SixDRepNet(dict_path=pose_model_path, gpu_id=-1)

video_path = current_dir.parents[0] / "input" / config["input_video_filename"]
output_dir = current_dir.parents[0] / "output"
output_path = output_dir / config["output_video_filename"]
output_csv_path = output_dir / config["output_csv_filename"]


shelf_a = np.array([[725, 340], [815, 240], [1060, 400], [1010, 500]], dtype=np.int32)

shelf_ab = np.array([[540, 435], [680, 355], [970, 550], [830, 720]], dtype=np.int32)

shelf_b = np.array([[215, 580], [442, 530], [640, 720], [285, 720]], dtype=np.int32)

shelf_c = np.array([[343, 460], [670, 315], [740, 360], [410, 530]], dtype=np.int32)

shelf_d = np.array([[148, 430], [260, 400], [580, 720], [285, 720]], dtype=np.int32)

all_shelfs = [shelf_a, shelf_ab, shelf_b, shelf_c, shelf_d]


def check_point_in_polygon(point, polygon):
    """
    Check whether a point is inside a polygon.
    """
    x, y = point
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def check_position_inside_any_shelf(position):
    for i, shelf in enumerate(all_shelfs):
        in_shelf = check_point_in_polygon(position, shelf)
        if in_shelf:
            return True
    return False


def get_shelf_interest(position, person_img, p_id):
    face_crop = detect_face(face_model=face_model, person_img=person_img)
    yaw = None
    if face_crop is not None and face_crop.size > 0:
        yaw = detect_head_pose(pose_model=pose_model, face_img=face_crop)
    else:
        return None

    in_shelf_d = check_point_in_polygon(position, shelf_d)

    if in_shelf_d:
        if yaw >= 15 and yaw < 50:
            return "D"

    in_shelf_b = check_point_in_polygon(position, shelf_b)

    if in_shelf_b:
        if yaw > -40 and yaw < 15:
            return "B"

    in_shelf_ab = check_point_in_polygon(position, shelf_ab)

    if in_shelf_ab:
        if yaw >= 20 and yaw < 50:
            return "B"
        elif yaw > -20 and yaw < 25:
            return "A"
        else:
            print("ID", p_id, "yaw", yaw)

    in_shelf_c = check_point_in_polygon(position, shelf_c)

    if in_shelf_c:
        if yaw >= -30 and yaw <= 2:
            return "C"

    return None


def put_person_label(person_id, label, frame, label_position):
    label_text = f"ID {person_id}: {label}"
    x1, y1 = label_position
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def main():
    video_path = "/Users/luqman/Downloads/Hendricks_Retail_Video_Analytics_Take_Home_Assessment_Brief_v5 1/raw_videos/interior.mp4"
    # video_path = "/Users/luqman/Documents/GitHub/video-task/input/trim1.mp4"
    video_path = "/Users/luqman/Downloads/trim6.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0

    # frame_path = "/Users/luqman/Documents/GitHub/video-task/frame_1.jpg"
    # frame = cv2.imread(frame_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # if frame_number < 90:
        #     continue

        # if frame_number == 120:
        #     break

        # print(">>> frame", frame_number)

        results = yolo_model.track(
            frame,
            persist=True,
            tracker="yolo_tracker.yaml",
            classes=[0],  # COCO class 0 = person
            verbose=False,
        )

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, person_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)

                # Use feet position
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                current_position = (foot_x, foot_y)

                person_crop = frame[
                    max(0, y1) : min(frame.shape[0], y2),
                    max(0, x1) : min(frame.shape[1], x2),
                ]
                if person_crop.size <= 0:
                    continue

                in_any_shelf = check_position_inside_any_shelf(current_position)

                # print("ID", person_id)

                if in_any_shelf:
                    person_label = "-"

                    shelf_interest = get_shelf_interest(
                        current_position, person_crop, person_id
                    )
                    if shelf_interest:
                        person_label = f"Shelf {shelf_interest}"
                    put_person_label(
                        person_id, person_label, frame, label_position=(x1, y1)
                    )

            # save_path = f"/Users/luqman/Documents/GitHub/video-task/temp_out/frame_{frame_number}.jpg"
            # cv2.imwrite(save_path, frame)

        cv2.imshow("Display Window", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("DONE")
