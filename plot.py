import cv2
import numpy as np

ENTRANCE_A = (280, 280)
ENTRANCE_B = (1110, 464)

frame = cv2.imread('frame.jpg')

STOP_ZONE = np.array([
    [524, 180],
    [1120, 327],
    [1110, 464],
    [280, 280],
], dtype=np.int32)

cv2.polylines(frame, [STOP_ZONE], True, (255, 0, 0), 2)

cv2.line(frame, ENTRANCE_A, ENTRANCE_B, (0, 255, 255), 3)

# cv2.circle(frame, point3, 5, (0, 0, 255), -1)
# cv2.circle(frame, point4, 5, (0, 0, 255), -1)

cv2.imwrite('plot.jpg', frame)