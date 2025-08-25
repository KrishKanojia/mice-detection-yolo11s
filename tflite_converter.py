from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("my_model_yolo11s_120_epochs/my_model.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'


