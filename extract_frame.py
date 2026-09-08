import cv2



FRAME_INDEX = 1

video_path = '/Users/luqman/Downloads/Hendricks_Retail_Video_Analytics_Take_Home_Assessment_Brief_v5 1/raw_videos/interior.mp4'

cap = cv2.VideoCapture(video_path)


for i in range(FRAME_INDEX, FRAME_INDEX + 50, 10):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, frame = cap.read()
    if success:
        cv2.imwrite(f"frame_{i}.jpg", frame)

cap.release()
