import cv2

VIDEO_PATH = "/Users/luqman/Downloads/clip-stop-1.mp4"
FRAME_INDEX = 600

cap = cv2.VideoCapture(VIDEO_PATH)


for i in range(FRAME_INDEX, FRAME_INDEX + 50, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, frame = cap.read()
    if success:
        cv2.imwrite(f"frame_{i}.jpg", frame)

cap.release()
