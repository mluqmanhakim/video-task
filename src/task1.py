from collections import defaultdict, deque
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sixdrepnet import SixDRepNet
from ultralytics import YOLO

from util import (
    check_if_enter_store,
    check_if_exit_store,
    check_point_in_stop_zone,
    put_top_right_text,
    detect_face,
    detect_head_pose,
)

current_dir = Path.cwd()
config_path = current_dir.parents[0] / "config" / "task1.json"
model_dir = current_dir.parents[0] / "model"

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Minimum number of frames a person must remain almost stationary to be considered stopped.
STOP_FRAMES = config["stop_frames"]
# Maximum movement in pixels between frames to consider a person stationary.
STOP_DISTANCE = config["stop_distance"]
# Number of frames to remember for each person
HISTORY_LENGTH = config["history_length"]
STOP_ZONE = np.array(config["stop_zone"], dtype=np.int32)
ENTRANCE_A = config["entrance_a"]
ENTRANCE_B = config["entrance_b"]

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


cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
save_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

positions = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
stop_counter = defaultdict(int)
person_state = defaultdict(lambda: "UNK")
entered_counter = 0
pass_by_counter = 0
frame_number = 0

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

            # Draw ID + state
            # label = f"ID {person_id}: " f"{person_state[person_id]}"
            # cv2.putText(
            #     frame,
            #     label,
            #     (x1, y1 - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (0, 255, 0),
            #     2,
            # )

            if person_state[person_id] not in ["ENTERED", "PASSED_BY", "EXITED"]:
                # Use feet position
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)
                current_position = (foot_x, foot_y)

                # Store trajectory
                previous_position = None
                if len(positions[person_id]) > 0:
                    previous_position = positions[person_id][-1]

                positions[person_id].append(current_position)
                inside_stop_zone = check_point_in_stop_zone(current_position, STOP_ZONE)

                if inside_stop_zone and previous_position:
                    # Euclidean distance
                    movement = np.linalg.norm(
                        np.array(current_position) - np.array(previous_position)
                    )
                    # STOP detection
                    if movement < STOP_DISTANCE:
                        stop_counter[person_id] += 1
                    else:
                        stop_counter[person_id] = 0

                # Person has stopped
                if stop_counter[person_id] >= STOP_FRAMES:
                    if person_state[person_id] == "UNK":
                        person_state[person_id] = "STOPPED"

                # Detect entrance crossing
                person_entered = False
                if previous_position is not None:
                    person_entered = check_if_enter_store(
                        previous_position, current_position, ENTRANCE_A, ENTRANCE_B
                    )
                    if person_entered:
                        person_state[person_id] = "ENTERED"
                        entered_counter += 1

                person_exited = False
                if previous_position is not None:
                    person_exited = check_if_exit_store(
                        previous_position, current_position, ENTRANCE_A, ENTRANCE_B
                    )
                    if person_exited:
                        person_state[person_id] = "EXITED"

                if person_state[person_id] == "STOPPED":
                    if frame_number % 5 == 0:
                        person_crop = frame[
                            max(0, y1) : min(frame.shape[0], y2),
                            max(0, x1) : min(frame.shape[1], x2),
                        ]
                        if person_crop.size > 0:
                            face_crop = detect_face(
                                face_model=face_model, person_img=person_crop
                            )
                            if face_crop is not None and face_crop.size > 0:
                                yaw = detect_head_pose(
                                    pose_model=pose_model, face_img=face_crop
                                )
                                if yaw >= 30 and yaw <= 90:
                                    person_state[person_id] = "INTERESTED"

                if person_state[person_id] == "INTERESTED" and not inside_stop_zone:
                    person_state[person_id] = "PASSED_BY"
                    pass_by_counter += 1

            counter_text_1 = f"Entered: {entered_counter}, Passed-by: {pass_by_counter}"
            counter_text_2 = f"Total: {entered_counter + pass_by_counter}"
            put_top_right_text(counter_text_1, height, width, frame)
            put_top_right_text(counter_text_2, height, width, frame, height_margin=40)

    save_video.write(frame)

cap.release()
cv2.destroyAllWindows()

result_data = {
    "Total Interested": entered_counter + pass_by_counter,
    "Interested Entered": entered_counter,
    "Interested Passed By": pass_by_counter,
}
df = pd.DataFrame(result_data, index=[0])
df.to_csv(output_csv_path, index=False)

print("Done")
