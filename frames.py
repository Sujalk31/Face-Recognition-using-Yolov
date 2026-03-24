import cv2
import os

video_path = "/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/TMKOC Part3.mp4"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f"{output_folder}/frame_{frame_id}.jpg", frame)
    frame_id += 1

cap.release()
print("Frames extracted:", frame_id)