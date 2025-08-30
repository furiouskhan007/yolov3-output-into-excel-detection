import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help='path to yolo config file', default=r'yolov3-tiny.cfg')
ap.add_argument('-w', '--weights', 
                help='path to yolo pre-trained weights', default=r'yolov3-tiny.weights')
ap.add_argument('-cl', '--classes', 
                help='path to text file containing class names', default=r'coco.names')
ap.add_argument('-i', '--image', 
                help='path to input image', default=r'input.jpg')
args = ap.parse_args()

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

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

# Load classes
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load model
net = cv2.dnn.readNet(args.weights, args.config)

# Load image
image = cv2.imread(args.image)
Height, Width = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416, 416), [0,0,0], True, crop=False)
net.setInput(blob)
outs = net.forward(getOutputsNames(net))

class_ids, confidences, boxes = [], [], []
conf_threshold = 0.5
nms_threshold = 0.4

# Collect detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# Apply NMS and draw results
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0] if isinstance(i, (list, np.ndarray)) else i
    box = boxes[i]
    x, y, w, h = box
    draw_pred(image, class_ids[i], confidences[i], x, y, x+w, y+h)

# Show result
cv2.imshow("Detections", image)
cv2.imwrite("output.jpg", image)  # save if needed
cv2.waitKey(0)
cv2.destroyAllWindows()
