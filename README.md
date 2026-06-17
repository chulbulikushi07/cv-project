# Real-Time Object Detection with YOLOv8 and OpenCV

## Overview

This project demonstrates real-time object detection using the YOLOv8 Nano model and OpenCV. It supports both webcam-based live detection and image-based detection.

The application detects objects, draws bounding boxes, displays class labels and confidence scores, and calculates FPS (Frames Per Second) in real time.

## Features

* Real-time object detection using a webcam
* YOLOv8 Nano (`yolov8n.pt`) model integration
* Automatic bounding box and label generation
* Confidence score visualization
* Live FPS counter
* Single-image object detection support
* OpenCV-based video processing

## Technologies Used

* Python
* OpenCV
* Ultralytics YOLOv8
* NumPy (dependency of OpenCV/YOLO)

## Project Structure

```text
project/
│
├── opencv.py          # Real-time webcam object detection
├── yolo.py            # Single image object detection
├── yolov8n.pt         # YOLOv8 Nano model
└── README.md
```

## Installation

### Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Install dependencies

```bash
pip install ultralytics opencv-python
```

### Download the YOLOv8 Nano model

The model will automatically download the first time it is used:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

## Usage

### Real-Time Webcam Detection

Run:

```bash
python opencv.py
```

Features:

* Opens the default webcam
* Performs object detection on every frame
* Displays FPS in real time
* Press `q` to quit

### Image Detection

Run:

```bash
python yolo.py
```

This script loads an image, performs object detection, and displays the results.

## Example Output

Detected objects are displayed with:

* Bounding boxes
* Class names
* Confidence scores
* FPS counter (webcam mode)

## Future Improvements

* Video file detection support
* Object tracking
* Custom-trained YOLO models
* Detection statistics dashboard
* Saving annotated images and videos
* GPU acceleration benchmarking

## Learning Outcomes

Through this project, I learned:

* Computer vision fundamentals
* Real-time video processing with OpenCV
* YOLOv8 object detection workflow
* Working with webcam streams
* Performance measurement using FPS
* Integrating deep learning models into Python applications

## Author

**Kushagra Sanwal**

Computational Mathematics Student
Mahindra University
