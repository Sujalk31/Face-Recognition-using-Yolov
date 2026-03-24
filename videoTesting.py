from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/runs/detect/train2/weights/best.pt")

# Input video
video_path = "/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/TMKOC Part3.mp4"

cap = cv2.VideoCapture(video_path)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video
out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("Detection", annotated_frame)

    # Save video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()