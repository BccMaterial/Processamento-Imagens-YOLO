from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # build a new model from YAML

# Train the model
results = model.train(data="data.yaml", epochs=30, imgsz=640)
