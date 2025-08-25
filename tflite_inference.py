import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self, resolution=(640,480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--model', help='Path to YOLO model file (example: "my_model_float32.tflite")',
                    required=True)
parser.add_argument('--labels', help='Name of the labelmap file (example: "labelmap.txt")',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.model
MODEL_DIR = args.model_dir
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, MODEL_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details ONCE
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Get tensor index ONCE
output_tensor_index = output_details[0]['index']
input_tensor_index = input_details[0]['index']

while True:
    t1 = cv2.getTickCount()

    frame1 = videostream.read()
    if frame1 is None:
        break

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_tensor_index, input_data)
    interpreter.invoke()

    # Get the single output tensor from the model
    output = interpreter.get_tensor(output_tensor_index)[0]

    # Initialize empty lists for detection results
    boxes = []
    classes = []
    scores = []

    # The output tensor has shape [5, 8400]. We loop through the 8400 potential detections.
    for i in range(output.shape[1]):
        # Get confidence score, which is at index 4 for a single-class model
        confidence = output[4, i]

        # Filter out detections below the confidence threshold
        if confidence > min_conf_threshold:
            # Since it's a single-class model, the class index is always 0
            class_id = 0
            
            # Get bounding box coordinates in x_center, y_center, width, height format
            x_center, y_center, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
            
            # Convert to xmin, ymin, xmax, ymax format
            xmin = int((x_center - w / 2) * imW)
            ymin = int((y_center - h / 2) * imH)
            xmax = int((x_center + w / 2) * imW)
            ymax = int((y_center + h / 2) * imH)

            # Append to our lists
            boxes.append([ymin, xmin, ymax, xmax])
            classes.append(class_id)
            scores.append(confidence)

    # Drawing bounding boxes and labels
    for i in range(len(scores)):
        ymin, xmin, ymax, xmax = boxes[i]
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        object_name = labels[int(classes[i])]
        label = f'{object_name}: {int(scores[i]*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw framerate
    frame_rate_calc = cv2.getTickFrequency() / (cv2.getTickCount() - t1)
    cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()