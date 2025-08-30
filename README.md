# YOLOv3-Tiny Object Detection → Timestamped Excel

A lightweight object detection pipeline leveraging YOLOv3-Tiny to detect objects in video or image streams, capture the detection timestamps, and export the results into a structured Excel (.xlsx) file for analysis, reporting, or further processing.

## Features

- Fast and efficient detection using the compact YOLOv3-Tiny model suitable for resource-constrained environments.
- Automatic timestamp logging for each detected object—perfect for tracking events over time.
- Excel export with clear columns for timestamp, object class, confidence score, and bounding box coordinates.

Versatile input support—process live video, recorded footage, or individual images.
# Python Compatibility 
* Python 3.10.4

## Libraries Used
1. opencv-python
2. PyTesseract
3. Flask
4. Deskew
5. openpyxl (for Excel output)

- YOLO configuration and weights:
- yolov3-tiny.cfg
- yolov3-tiny.weights
- coco.names (or your label names file)

## Installation

- Clone this repository:
```git clone https://github.com/furiouskhan007/yolov3-tiny-Object-detection-Timestamp-Excel.git```<br>
```cd yolov3-tiny-Object-detection-Timestamp-Excel```<br>
- Set up a virtual environment (optional but recommended):
```python -m venv venv```<br>
```source venv/bin/activate  # or `venv\Scripts\activate` on Windows```<br>
- Install required Python packages:
```pip install opencv-python numpy pandas openpyxl```<br>
- Download or place the YOLO config, weights, and names file into a models/ directory (or adjust as needed in your code).


## Screenshots of App
![alt text](https://github.com/furiouskhan007/yolov3-output-into-excel-detection/blob/main/output.jpg?raw=true)
