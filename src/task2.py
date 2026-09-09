from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sixdrepnet import SixDRepNet
from ultralytics import YOLO

from util import (
    put_top_right_text,
    check_position_inside_any_shelf,
    get_shelf_interest,
    put_person_label,
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

# Minimum number of frames a person must remain interacting to be considered 'interested to the shelf'.
INTERESTED_FRAMES = 90


def main():
    video_path = current_dir.parents[0] / "input" / config["input_video_filename"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0

    shelf_a_counter = defaultdict(int)
    shelf_b_counter = defaultdict(int)
    shelf_c_counter = defaultdict(int)
    shelf_d_counter = defaultdict(int)

    shelf_a_people = set()
    shelf_b_people = set()
    shelf_c_people = set()
    shelf_d_people = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

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

                if in_any_shelf:
                    person_label = "-"

                    shelf_interest = get_shelf_interest(
                        position=current_position,
                        person_img=person_crop,
                        face_model=face_model,
                        pose_model=pose_model,
                    )
                    if shelf_interest:
                        timer = 0

                        if shelf_interest == "A":
                            shelf_a_counter[person_id] += 1
                            timer = int(shelf_a_counter[person_id] / 30)
                            if shelf_a_counter[person_id] >= INTERESTED_FRAMES:
                                shelf_a_people.add(person_id)

                        elif shelf_interest == "B":
                            shelf_b_counter[person_id] += 1
                            timer = int(shelf_b_counter[person_id] / 30)
                            if shelf_b_counter[person_id] >= INTERESTED_FRAMES:
                                shelf_b_people.add(person_id)

                        elif shelf_interest == "C":
                            shelf_c_counter[person_id] += 1
                            timer = int(shelf_c_counter[person_id] / 30)
                            if shelf_c_counter[person_id] >= INTERESTED_FRAMES:
                                shelf_c_people.add(person_id)

                        elif shelf_interest == "D":
                            shelf_d_counter[person_id] += 1
                            timer = int(shelf_d_counter[person_id] / 30)
                            if shelf_d_counter[person_id] >= INTERESTED_FRAMES:
                                shelf_d_people.add(person_id)

                        person_label = f"Shelf {shelf_interest} - {str(timer)}s"

                        put_person_label(
                            person_id, person_label, frame, label_position=(x1, y1)
                        )

            put_top_right_text(
                text=f"A: {len(shelf_a_people)}",
                image_height=height,
                image_width=width,
                image=frame,
            )
            put_top_right_text(
                text=f"B: {len(shelf_b_people)}",
                image_height=height,
                image_width=width,
                image=frame,
                height_margin=30,
            )
            put_top_right_text(
                text=f"C: {len(shelf_c_people)}",
                image_height=height,
                image_width=width,
                image=frame,
                height_margin=60,
            )
            put_top_right_text(
                text=f"D: {len(shelf_d_people)}",
                image_height=height,
                image_width=width,
                image=frame,
                height_margin=90,
            )

        # cv2.imshow("Display Window", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        save_video.write(frame)

    cap.release()
    cv2.destroyAllWindows()

    result_data = {
        "Interested in Shelf A": len(shelf_a_people),
        "Interested in Shelf B": len(shelf_b_people),
        "Interested in Shelf C": len(shelf_c_people),
        "Interested in Shelf D": len(shelf_d_people),
    }
    df = pd.DataFrame(result_data, index=[0])
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
    print("Done")
