---
title: Part-2-EcoClassify
app_file: app.py
sdk: gradio
sdk_version: 5.49.1
---

# EcoClassify Part 2: Real-Time Recyclables Detection

**EcoClassify Part 2** is an AI-powered **real-time object detection system** that identifies and classifies recyclable materials using a custom-trained **YOLOv8** model. The system detects **plastic, paper, glass, and metal** through live webcam feed with instant visual feedback and session statistics tracking.

Built for **local deployment** using **OpenCV** for maximum performance and real-time responsiveness.

---

## Features

### Real-Time Detection
- **Live webcam processing** at 25-30 FPS (GPU) or 5-10 FPS (CPU)
- **Mirrored camera view** for intuitive user interaction
- **Bounding boxes** with confidence scores for each detection
- **Instant alerts** when new recyclable materials are detected

### Smart Statistics Dashboard
- **Current frame** detection count
- **Session total** tracking (last 100 detections)
- **Material breakdown** by type with color-coded indicators
- **Real-time FPS counter** for performance monitoring

### Performance Optimizations
- **GPU acceleration** with CUDA support (2x speedup with FP16)
- **Automatic device detection** (GPU/CPU)
- **Statistics caching** to reduce overhead
- **416x416 input size** for speed-accuracy balance

### Visual Feedback
- **Material-specific colors**: Plastic (Blue), Paper (Green), Metal (Orange), Glass (Purple)
- **8-second fade-out alerts** with progress bars
- **Semi-transparent overlays** for non-intrusive display
- **Clean, professional UI** design
---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (built-in or external)
- 4GB RAM minimum (8GB recommended)
- **Optional**: NVIDIA GPU with CUDA for acceleration

## Setup Instructions (Local Installation)

Follow these steps to run **EcoClassify Part 2** locally on your computer:

### Clone or download the project
```bash
git clone https://huggingface.co/spaces/charmainecmr/EcoClassifyPart2
cd EcoClassifyPart2
```

## Create and activate a virtual environment (recommended)
```
python -m venv venv
venv\Scripts\activate       # On Windows
# or
source venv/bin/activate    # On macOS/Linux
```

## Install all required dependencies
``` 
pip install -r requirements.txt
```

## Run the Application
```
python app.py
```
