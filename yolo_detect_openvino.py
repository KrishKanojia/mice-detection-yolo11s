import os
import sys
import argparse
import glob
import time
import yaml
import cv2
import numpy as np
# from ultralytics import YOLO # Not needed for OpenVINO inference once labels are in metadata.yaml
from openvino.runtime import Core, Layout, Type

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to OpenVINO model XML file (e.g., "path/to/my_model.xml")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float) # Ensure threshold is float
parser.add_argument('--iou-thresh', help='IOU threshold for Non-Maximum Suppression (NMS) (example: "0.4")',
                    default=0.45, type=float) # Add IOU threshold for NMS
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path_base = args.model
img_source = args.source
min_thresh = args.thresh
iou_thresh = args.iou_thresh
user_res = args.resolution
record = args.record

# --- OpenVINO Model Loading and Setup ---
# Determine paths for .xml, .bin, and metadata.yaml
if model_path_base.endswith('.xml'):
    model_xml_path = model_path_base
    model_bin_path = model_path_base.replace('.xml', '.bin')
    model_dir = os.path.dirname(model_xml_path)
elif model_path_base.endswith('.bin'):
    model_bin_path = model_path_base
    model_xml_path = model_path_base.replace('.bin', '.xml')
    model_dir = os.path.dirname(model_bin_path)
else: # If user passes a .pt file, derive OpenVINO paths (assuming same directory)
    model_dir = os.path.dirname(model_path_base)
    model_name_without_ext = os.path.splitext(os.path.basename(model_path_base))[0]
    model_xml_path = os.path.join(model_dir, f"{model_name_without_ext}.xml")
    model_bin_path = os.path.join(model_dir, f"{model_name_without_ext}.bin")

# Check if OpenVINO model files exist
if not os.path.exists(model_xml_path) or not os.path.exists(model_bin_path):
    print(f'ERROR: OpenVINO model files (.xml or .bin) not found based on {model_path_base}.')
    print('Please ensure your model is exported to OpenVINO format and specified correctly.')
    sys.exit(0)

# --- Load class names (labels) from metadata.yaml ---
# Path to the metadata.yaml file (corrected to use model_dir)
metadata_yaml_path = os.path.join(model_dir, 'metadata.yaml')

labels = {}
if os.path.exists(metadata_yaml_path):
    try:
        with open(metadata_yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        if 'names' in metadata and isinstance(metadata['names'], dict):
            labels = metadata['names']
            print(f"Labels loaded from metadata.yaml: {labels}")
        else:
            print("Warning: 'names' key not found or not a dictionary in metadata.yaml. Using fallback labels.")
            labels = {0: 'object_0'} # Fallback
    except Exception as e:
        print(f"Error loading labels from metadata.yaml: {e}. Using fallback labels.")
        labels = {0: 'object_0'} # Fallback
else:
    print("Warning: metadata.yaml not found. Using fallback labels.")
    labels = {0: 'object_0'} # Fallback


# Initialize OpenVINO Runtime
ie = Core()

# Read the OpenVINO model from IR files
model_ov = ie.read_model(model=model_xml_path, weights=model_bin_path)

# Compile the model for inference on a specific device (e.g., "CPU", "GPU", "NPU", "MULTI")
compiled_model = ie.compile_model(model=model_ov, device_name="CPU")

# Get input and output layers for model interaction
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Get expected input dimensions for the OpenVINO model
ov_input_height = input_layer.shape[2]
ov_input_width = input_layer.shape[3]
print(f"OpenVINO Model Input Shape: {input_layer.shape}")
print(f"OpenVINO Model Output Shape: {output_layer.shape}")

# --- Input Source Setup ---
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

source_type = None
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize_display = False
if user_res:
    resize_display = True
    display_res_W, display_res_H = int(user_res.split('x')[0]), int(user_res.split('x')[1])
else:
    display_res_W, display_res_H = None, None

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb','picamera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at (--resolution argument).')
        sys.exit(0)
    
    record_name = 'demo_openvino.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (display_res_W, display_res_H))


# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(os.path.join(img_source, '*'))
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_res_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_res_H)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    picam_capture_width = display_res_W if resize_display else ov_input_width
    picam_capture_height = display_res_H if resize_display else ov_input_height
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (picam_capture_width, picam_capture_height)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# --- UPDATED Post-processing function for OpenVINO YOLO output ---
def postprocess_openvino_output(raw_output, original_frame_shape, input_width, input_height, conf_threshold, iou_threshold, labels):
    """
    Decodes raw OpenVINO YOLO output, applies confidence threshold and NMS,
    and scales bounding boxes to the original frame size.

    Args:
        raw_output (np.array): The raw output tensor from OpenVINO inference.
                                Expected shape: [1, 5, 8400] for single-class.
                                Or [1, 5+num_classes, num_boxes] for multi-class.
        original_frame_shape (tuple): (height, width, channels) of the original frame.
        input_width (int): Width model expects (e.g., 640).
        input_height (int): Height model expects (e.g., 640).
        conf_threshold (float): Minimum confidence to keep a detection.
        iou_threshold (float): IOU threshold for Non-Maximum Suppression.
        labels (dict): Dictionary mapping class IDs to class names.

    Returns:
        list: A list of dictionaries, each containing 'box', 'conf', 'cls'.
    """
    detections = []
    
    # --- Handling the [1, 5, 8400] single-class output format ---
    # This is for YOLO models where the output is (batch, [x,y,w,h,conf], num_boxes)
    if raw_output.ndim == 3 and raw_output.shape[1] == 5 and len(labels) == 1:
        output_data = raw_output[0].T # Transpose to (num_boxes, 5)

        boxes_xywh = output_data[:, :4]  # x_center, y_center, width, height
        confidences = output_data[:, 4] # This is the confidence for the single class
        
        # Since it's a single-class model, all valid detections belong to class 0 (mouse)
        class_ids = np.zeros(output_data.shape[0], dtype=np.int32) 

        # Filter by confidence threshold
        valid_indices = np.where(confidences > conf_threshold)[0]

        if len(valid_indices) == 0:
            return []

        boxes_filtered_xywh = boxes_xywh[valid_indices]
        confidences_filtered = confidences[valid_indices]
        class_ids_filtered = class_ids[valid_indices]

        # Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes_xyxy = np.copy(boxes_filtered_xywh)
        boxes_xyxy[:, 0] = boxes_filtered_xywh[:, 0] - boxes_filtered_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_filtered_xywh[:, 1] - boxes_filtered_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_filtered_xywh[:, 0] + boxes_filtered_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_filtered_xywh[:, 1] + boxes_filtered_xywh[:, 3] / 2

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences_filtered.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) == 0:
            return []
        
        # Extract final detections after NMS and scale coordinates
        orig_h, orig_w, _ = original_frame_shape
        scale_x = orig_w / input_width
        scale_y = orig_h / input_height

        for i in indices.flatten():
            box = boxes_xyxy[i]
            x1, y1, x2, y2 = box
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': confidences_filtered[i],
                'cls': class_ids_filtered[i]
            })

    # --- Handling the general YOLOv8/v11s multi-class output format (batch, 5+num_classes, num_boxes) ---
    # This branch is for models with multiple classes where raw_output.shape[1] == (5 + number_of_classes)
    elif raw_output.ndim == 3 and raw_output.shape[1] == (5 + len(labels)) and len(labels) > 1:
        raw_output = raw_output[0].T # Transpose to (num_boxes, 5 + num_classes)
        
        boxes_xywh = raw_output[:, :4]
        object_conf = raw_output[:, 4]
        class_scores = raw_output[:, 5:]

        max_class_scores = np.max(class_scores, axis=1)
        predicted_class_ids = np.argmax(class_scores, axis=1)
        overall_confidences = object_conf * max_class_scores

        valid_indices = np.where(overall_confidences > conf_threshold)[0]

        if len(valid_indices) == 0:
            return []

        boxes_filtered_xywh = boxes_xywh[valid_indices]
        confidences_filtered = overall_confidences[valid_indices]
        class_ids_filtered = predicted_class_ids[valid_indices]

        boxes_xyxy = np.copy(boxes_filtered_xywh)
        boxes_xyxy[:, 0] = boxes_filtered_xywh[:, 0] - boxes_filtered_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_filtered_xywh[:, 1] - boxes_filtered_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_filtered_xywh[:, 0] + boxes_filtered_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_filtered_xywh[:, 1] + boxes_filtered_xywh[:, 3] / 2

        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences_filtered.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) == 0:
            return []
        
        orig_h, orig_w, _ = original_frame_shape
        scale_x = orig_w / input_width
        scale_y = orig_h / input_height

        for i in indices.flatten():
            box = boxes_xyxy[i]
            x1, y1, x2, y2 = box
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': confidences_filtered[i],
                'cls': class_ids_filtered[i]
            })
    
    # --- Handling a flattened output format (batch=1, N_detections * 6) after some processing ---
    # This is for cases where the model might output pre-NMS detections already in [x1, y1, x2, y2, conf, class_id] format.
    # It might be 2D (num_detections, 6) or 3D (1, num_detections, 6) where '6' is [x1, y1, x2, y2, conf, class_id].
    elif raw_output.ndim >= 2 and raw_output.shape[-1] == 6:
        if raw_output.ndim == 3:
            raw_output = raw_output.squeeze(0) # Remove batch dimension if present

        boxes = raw_output[:, :4]
        confidences = raw_output[:, 4]
        class_ids = raw_output[:, 5].astype(np.int32)

        valid_indices = np.where(confidences > conf_threshold)[0]
        
        if len(valid_indices) == 0:
            return []

        boxes_filtered = boxes[valid_indices]
        confidences_filtered = confidences[valid_indices]
        class_ids_filtered = class_ids[valid_indices]

        indices = cv2.dnn.NMSBoxes(boxes_filtered.tolist(), confidences_filtered.tolist(), conf_threshold, iou_threshold)
        
        if len(indices) == 0:
            return []

        orig_h, orig_w, _ = original_frame_shape
        scale_x = orig_w / input_width
        scale_y = orig_h / input_height

        for i in indices.flatten():
            box = boxes_filtered[i]
            x1, y1, x2, y2 = box

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            detections.append({
                'box': [x1, y1, x2, y2],
                'conf': confidences_filtered[i],
                'cls': class_ids_filtered[i]
            })
    else:
        print(f"WARNING: Unexpected OpenVINO model output shape: {raw_output.shape}. Cannot parse detections.")
        return []

    return detections

# Begin inference loop
while True:
    t_start = time.perf_counter()

    frame = None
    original_frame = None

    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Reached end of the video file or unable to read from camera. Exiting program.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from the Picamera. Exiting program.')
            break

    # IMPORTANT: Assign original_frame immediately after reading the frame
    original_frame = frame.copy() 

    # --- Preprocess frame for OpenVINO inference ---
    input_for_ov = cv2.resize(frame, (ov_input_width, ov_input_height))
    input_for_ov = input_for_ov.transpose((2, 0, 1))
    input_for_ov = np.expand_dims(input_for_ov, 0)
    input_for_ov = input_for_ov.astype(np.float32) / 255.0

    raw_ov_predictions = compiled_model([input_for_ov])[output_layer]

    processed_detections = postprocess_openvino_output(
        raw_ov_predictions,
        original_frame.shape,
        ov_input_width,
        ov_input_height,
        min_thresh,
        iou_thresh,
        labels
    )

    object_count = 0
    for det in processed_detections:
        xmin, ymin, xmax, ymax = det['box']
        conf = det['conf']
        classidx = det['cls']
        classname = labels.get(classidx, f'Class {classidx}')

        color = bbox_colors[classidx % len(bbox_colors)]
        cv2.rectangle(original_frame, (xmin, ymin), (xmax, ymax), color, 2)

        label = f'{classname}: {int(conf*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(original_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
        cv2.putText(original_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        object_count += 1

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(original_frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    cv2.putText(original_frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    display_frame = original_frame
    if resize_display:
        display_frame = cv2.resize(original_frame, (display_res_W, display_res_H))

    cv2.imshow('OpenVINO YOLO Detection Results', display_frame)
    if record: recorder.write(display_frame)

    key = None
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture_openvino.png', original_frame)

    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()