from ultralytics import YOLO

# Load trained model
model = YOLO("/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/runs/detect/train2/weights/best.pt")

# Evaluate on validation dataset
metrics = model.val(data="/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/data.yaml")

print(metrics)