import cv2
import argparse
import numpy as np
import datetime
from openpyxl import Workbook
import os

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',
                help='path to yolo config file', default=r'yolov3-tiny.cfg')
ap.add_argument('-w', '--weights',
                help='path to yolo pre-trained weights', default=r'yolov3-tiny.weights')
ap.add_argument('-cl', '--classes',
                help='path to text file containing class names', default=r'coco.names')
args = ap.parse_args()

# Get names of output layers
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw bounding box + label with background
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{classes[class_id]} {confidence:.2f}"
    color = COLORS[class_id]

    # Bounding box
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y_label = max(y, th + baseline)
    cv2.rectangle(img, (x, y_label - th - baseline), (x + tw, y_label), color, cv2.FILLED)

    # Label text
    cv2.putText(img, label, (x, y_label - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# Window setup
window_title = "Rubiks Detector"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

# Load classes
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

# Random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load model
net = cv2.dnn.readNet(args.weights, args.config)

# Video input
cap = cv2.VideoCapture(r'Demo.mp4')

# Get video properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (use 'XVID' for .avi)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Excel sheet
book = Workbook()
sheet = book.active

while cv2.waitKey(1) < 0:
    hasframe, image = cap.read()
    if not hasframe:
        break

    Height, Width = image.shape[:2]

    # Create blob from resized image (for YOLO detection only)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (416, 416)),
                                 1.0/255.0, (416, 416), [0,0,0], True, crop=False)
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))   # ✅ keep as 'outs'

    class_ids, confidences, boxes = [], [], []
    conf_threshold = 0.3
    nms_threshold = 0.2

    for out_layer in outs:
        for detection in out_layer:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                time_ref = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
                sheet.append((classes[class_id], time_ref))

                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    book.save('detections.xlsx')

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        box = boxes[i]
        x, y, w, h = box
        draw_pred(image, class_ids[i], confidences[i], x, y, x+w, y+h)

    # Inference time
    t, _ = net.getPerfProfile()
    label = 'Inference: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Show window
    cv2.imshow(window_title, image)

    # Write processed frame to output video
    out.write(image)   # ✅ this now works

cap.release()
out.release()
book.save('detections.xlsx')
cv2.destroyAllWindows()
