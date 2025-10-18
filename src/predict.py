from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
source = "video.mp4"
model.predict(source, save=True, imgsz=320, conf=0.5, show=True)
