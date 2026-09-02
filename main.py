import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque


from .util import check_if_enter_store


# =========================
# Configuration
# =========================

VIDEO_PATH = "/Users/luqman/Downloads/Hendricks_Retail_Video_Analytics_Take_Home_Assessment_Brief_v5 1/raw_videos/entrance.mp4"

MODEL_PATH = "yolo11n.pt"

# Minimum number of frames a person must remain almost stationary to be considered stopped.
STOP_FRAMES = 30

# Maximum movement in pixels between frames to consider a person stationary.
STOP_DISTANCE = 5

# Number of frames to remember for each person
HISTORY_LENGTH = 120



# model = YOLO(MODEL_PATH)




positions = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
stop_counter = defaultdict(int)
person_state = defaultdict(lambda: "WALKING")






cap = cv2.VideoCapture(VIDEO_PATH)
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1


    # =========================
    # YOLO + ByteTrack
    # =========================

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

        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = box.astype(int)

            # Use feet position instead of bounding-box center.
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)

            current_position = (foot_x, foot_y)


            # Store trajectory
            previous_position = None
            if len(positions[person_id]) > 0:
                previous_position = positions[person_id][-1]

            positions[person_id].append(current_position)


            # Detect movement
            movement = 0
            if previous_position is not None:
                # Euclidean distance
                movement = np.linalg.norm(np.array(current_position) - np.array(previous_position))


            # STOP detection
            if movement < STOP_DISTANCE:
                stop_counter[person_id] += 1
            else:
                stop_counter[person_id] = 0

            # Person has stopped
            if stop_counter[person_id] >= STOP_FRAMES:
                if person_state[person_id] == "WALKING":
                    person_state[person_id] = "STOPPED"


            # Detect entrance crossing
            person_entered = check_if_enter_store(previous_position, current_position, ENTRANCE_A, ENTRANCE_B)
            if person_entered:
                person_state[person_id] = "ENTERED"

                # skip frame for this person


            # Detect passed by
            # =========================

            inside_stop_zone = check_point_in_polygon(current_position, STOP_ZONE)

            if person_state[person_id] == "STOPPED" and not inside_stop_zone:
                person_state[person_id] = "PASSED_BY"
                # skip frame for this person


 
            # =========================
            # Draw ID + state
            # =========================

            label = (
                f"ID {person_id}: "
                f"{person_state[person_id]}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )



    # Display
    cv2.imshow("CCTV Person Analysis", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()