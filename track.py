from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolo11n.pt")

results = model.track("data/2025-04-22_08-34-09_footage.mp4", show=True)
