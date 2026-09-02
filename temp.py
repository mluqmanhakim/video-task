import cv2
import numpy as np





cap = cv2.VideoCapture('/Users/luqman/Downloads/Hendricks_Retail_Video_Analytics_Take_Home_Assessment_Brief_v5 1/raw_videos/entrance.mp4')


# Check if the video opened successfully
if cap.isOpened():
    # Get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")
else:
    print("Error: Could not open video.")


# frame_number = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_number += 1


# cap.release()


# ENTRANCE_A = (375, 255)
# ENTRANCE_B = (1110, 464)

frame = cv2.imread('frame_1.png')


# cv2.circle(frame, point3, 5, (0, 0, 255), -1)
# cv2.circle(frame, point4, 5, (0, 0, 255), -1)

# STOP_ZONE = np.array([
#     [524, 180],
#     [1110, 464],
#     [1120, 327],
#     [375, 255]
# ], dtype=np.int32)

# cv2.polylines(frame, [STOP_ZONE], True, (255, 0, 0), 2)


# cv2.line(frame, ENTRANCE_A, ENTRANCE_B, (0, 255, 255), 3)

# cv2.imwrite('output.jpg', frame)