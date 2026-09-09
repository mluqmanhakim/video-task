import cv2
import numpy as np


frame = cv2.imread('frame_2290.jpg')

shelf_a = np.array([[725, 340], [815, 240], [1060, 400], [1010, 500]], dtype=np.int32)
shelf_ab = np.array([[540, 435], [680, 355], [970, 550], [830, 720]], dtype=np.int32)
shelf_b = np.array([[215, 580], [442, 530], [640, 720], [285, 720]], dtype=np.int32)
shelf_c = np.array([[343, 460], [670, 315], [740, 360], [410, 530]], dtype=np.int32)
shelf_d = np.array([[148, 430], [260, 400], [580, 720], [285, 720]], dtype=np.int32)




cv2.polylines(frame, [shelf_a], True, (0, 0, 255), 2)

cv2.polylines(frame, [shelf_ab], True, (10, 100, 255), 2)

cv2.polylines(frame, [shelf_b], True, (100, 100, 255), 2)

cv2.polylines(frame, [shelf_c], True, (100, 255, 10), 2)

cv2.polylines(frame, [shelf_d], True, (200, 10, 100), 2)





cv2.imwrite('plot_shelf.jpg', frame)