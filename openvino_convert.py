from ultralytics import YOLO

# Load your trained YOLOv11s PyTorch model (.pt file)
# Replace 'path/to/your/my_model.pt' with the actual path to your trained model
model = YOLO('my_model_yolo11s_120_epochs_new_label/my_model.pt')

# Export the model to OpenVINO IR format
# 'imgsz' should match the input image size used during your model's training (e.g., 640 for 640x640)
# 'half=True' can be used for FP16 quantization if your target hardware supports it for smaller size and faster inference.
# For CPU, 'half=False' (default) often works best for full precision.
results = model.export(format='openvino', imgsz=640, half=False)

print(f"Model successfully exported to OpenVINO IR. Output path: {results}")