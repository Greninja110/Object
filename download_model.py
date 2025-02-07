from ultralytics import YOLO
import torch

# Specify your device ("cuda", "cpu", or a specific GPU index like "cuda:0")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Instantiating YOLOv8 will automatically download the "yolov8s.pt" weights
model = YOLO("yolov8s.pt")  # this will download if yolov8s.pt isn't present

# 2) Move model to the specified device
model.to(device)

print(f"[INFO] YOLOv8 model loaded on {device}!")
