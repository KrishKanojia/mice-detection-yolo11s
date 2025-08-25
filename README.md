# mice-detection-yolo11s
This repository contains a YOLO11s object detection model trained on a custom mice dataset consisting of 8,000 annotated images.

The model has been exported into multiple formats for flexible deployment across different environments:

## Available Model Formats

1. PyTorch (training + inference)

2. OpenVINO (optimized for Intel hardware)
https://drive.google.com/drive/folders/1ByYjEqASw1bUO4frAEjH2kPV541HOmzK?usp=sharing

4. TensorFlow Lite (TFLite) (optimized for mobile & edge devices)
https://drive.google.com/drive/folders/1Vzrt9h6RP37M8kWwWkkea5Jc4fVL0haL?usp=sharing

## Use Cases

Automated mice detection from camera

## Steps to run code:
1. Simple model:
python yolo_detect.py --model my_model_yolo11s_120_epochs_new_label/my_model.pt --source test_images --resolution 640x480 --thresh 0.5

2. Openvino model:
python yolo_detect_openvino.py --model "my_model_openvino_model_120_epochs_new_label/my_model.xml" --source recording.mp4 --resolution 640x480 --thresh 0.5


### Here are all the arguments for yolo_detect.py and  yolo_detect_openvino.py:


- **--model** *(required)*  
  Path to a model file (e.g. `my_model.pt`).  
  If the model isn't found, it will default to `yolov8s.pt`.

- **--source** *(required)*  
  Input source for inference. Options include:  
  - Image file → `test.jpg`  
  - Folder of images → `my_images/test/`  
  - Video file → `testvid.mp4`  
  - USB camera index → `usb0`  
  - Raspberry Pi Picamera index → `picamera0`

- **--thresh** *(optional)*  
  Minimum confidence threshold for displaying detected objects.  
  Default: **0.5** (example: `0.4`).

- **--resolution** *(optional)*  
  Output resolution in **WxH** format.  
  If not specified, the program will match the source resolution.  
  Example: `1280x720`.

- **--record** *(optional)*  
  Record inference results into a video file (`demo1.avi`).  
  ⚠️ Requires `--resolution` to be specified.
## Yolo11s 120 epochs results
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/001d24f7-9b3d-4ae2-b36a-a37b3d32281f" />

## Original Label Image
<img src="https://github.com/user-attachments/assets/b6c6f95e-e6b1-48b0-a300-3d432bf47122" width="500"/>



## Predicted Image
<img src="https://github.com/user-attachments/assets/01989397-72a7-42ab-a8de-5d2a08a1b7fd" width="500"/>



