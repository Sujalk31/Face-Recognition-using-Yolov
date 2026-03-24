from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train the model using your YAML dataset
model.train(
    data="/Users/coding/ResoluteAI Tasks/Face Recognition Using YoloV/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)