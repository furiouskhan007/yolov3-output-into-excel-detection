import cv2
import argparse
import numpy as np
import xlsxwriter
import datetime
from openpyxl import Workbook

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', help='path to yolo config file', default=r'yolov3-tiny.cfg')
ap.add_argument('-w', '--weights', help='path to yolo pre-trained weights', default=r'yolov3-tiny.weights')
ap.add_argument('-cl', '--classes', help='path to text file containing class names', default=r'coco.names')
args = ap.parse_args()

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

window_title = "Rubiks Detector"
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_title, 640, 640)

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

# Define video capture for the default camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

book = Workbook()
sheet = book.active

while cv2.waitKey(1) < 0:
    hasframe, image = cap.read()

    if not hasframe:
        break

    image = cv2.resize(image, (620, 480))
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), [0, 0, 0], True, crop=False)
    Width = image.shape[1]
    Height = image.shape[0]
    net.setInput(blob)

    outs = net.forward(getOutputsNames(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.25
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
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
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    book.save('detections.xlsx')

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0))

    cv2.imshow(window_title, image)

cap.release()
cv2.destroyAllWindows()
