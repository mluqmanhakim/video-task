import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque


from util import check_if_enter_store, check_if_exit_store, check_point_in_stop_zone


def put_top_right_text(text, image_height, image_width, image, height_margin=0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1
    color = (0, 255, 150)  # Green in BGR
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Set top-right coordinates with a 20-pixel margin from the edges
    margin = 20
    x = image_width - text_width - margin
    y = (
        text_height + margin + height_margin
    )  # Y flows downward, so add height to pull it into view

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


model = YOLO("yolo26x.pt")

VIDEO_PATH = "/Users/luqman/Downloads/Hendricks_Retail_Video_Analytics_Take_Home_Assessment_Brief_v5 1/raw_videos/entrance.mp4"

VIDEO_PATH = "/Users/luqman/Downloads/enter.mp4"


# Minimum number of frames a person must remain almost stationary to be considered stopped.
STOP_FRAMES = 90
# Maximum movement in pixels between frames to consider a person stationary.
STOP_DISTANCE = 5
# Number of frames to remember for each person
HISTORY_LENGTH = 120

STOP_ZONE = np.array([[524, 180], [1120, 327], [1110, 464], [375, 255]], dtype=np.int32)
ENTRANCE_A = (280, 280)
ENTRANCE_B = (1110, 464)


positions = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
stop_counter = defaultdict(int)
person_state = defaultdict(lambda: "UNK")


entered_counter = 0
pass_by_counter = 0


cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

OUTPUT_PATH = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
save_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))


frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # YOLO + ByteTrack
    results = model.track(
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
            label = f"ID {person_id}: " f"{person_state[person_id]}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
                
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

                if person_state[person_id] == "STOPPED" and not inside_stop_zone:
                    person_state[person_id] = "PASSED_BY"
                    pass_by_counter += 1
            
            counter_text_1 = f"Entered: {entered_counter}, Passed-by: {pass_by_counter}"
            counter_text_2 = f"Total: {entered_counter + pass_by_counter}"
            put_top_right_text(counter_text_1, height, width, frame)
            put_top_right_text(counter_text_2, height, width, frame, height_margin=40)

    cv2.imshow("Store Analysis", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # save_video.write(frame)

cap.release()
cv2.destroyAllWindows()
