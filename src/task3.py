from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from util import put_person_label, check_point_in_polygon


STORE_ZONE = np.array([[0, 165], [1140, 450], [1280, 720], [0, 720]], dtype=np.int32)


def main():
    current_dir = Path.cwd()
    config_path = current_dir.parents[0] / "config" / "task3.json"
    model_dir = current_dir.parents[0] / "model"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    yolo_model_path = model_dir / config["yolo_model_filename"]
    yolo_model = YOLO(model=yolo_model_path)

    staff_model_path = model_dir / config["staff_model_filename"]
    staff_model = YOLO(staff_model_path)

    video_path = current_dir.parents[0] / "input" / config["input_video_filename"]
    output_dir = current_dir.parents[0] / "output"
    output_path = output_dir / config["output_video_filename"]
    output_csv_path = output_dir / config["output_csv_filename"]
    video_path = current_dir.parents[0] / "input" / config["input_video_filename"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number % 5 != 0:
            continue

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

                inside_store = check_point_in_polygon(current_position, STORE_ZONE)

                if not inside_store:
                    continue

                staff_model_result = staff_model.predict(
                    source=person_crop,
                    imgsz=224,
                    verbose=False,
                )[0]

                class_id = staff_model_result.probs.top1
                confidence = float(staff_model_result.probs.top1conf)

                if class_id == 0 and confidence >= 0.75:
                    person_label = "Staff"
                    put_person_label(
                        person_id, person_label, frame, label_position=(x1, y1)
                    )
                    break

        cv2.imshow("Display Window", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # save_video.write(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done")
