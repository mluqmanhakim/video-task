import cv2
import numpy as np

ENTRANCE_A = (280, 280)
ENTRANCE_B = (1110, 464)

frame = cv2.imread('frame_2280.jpg')






shelf_c = np.array([[343, 460], [670, 315], [740, 360], [410, 530]], dtype=np.int32)


shelf_ab = np.array([[540, 435], [680, 355], [970, 550], [830, 720]], dtype=np.int32)

shelf_b = np.array([[215, 580], [442, 530], [640, 720], [285, 720]], dtype=np.int32)

shelf_d = np.array([[148, 430], [260, 400], [580, 720], [285, 720]], dtype=np.int32)



cv2.polylines(frame, [shelf_d], True, (0, 0, 255), 2)

cv2.polylines(frame, [shelf_b], True, (0, 0, 255), 2)





# cv2.line(frame, ENTRANCE_A, ENTRANCE_B, (0, 255, 255), 3)

point2d = (645, 403)

cv2.circle(frame, point2d, 5, (0, 0, 255), -1)


# cv2.circle(frame, point4, 5, (0, 0, 255), -1)

cv2.imwrite('plot_shelf.jpg', frame)