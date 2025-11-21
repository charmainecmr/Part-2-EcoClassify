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

# Custom Yolov8 Model Flow Used
<img width="1600" height="896" alt="CV DIagram" src="https://github.com/user-attachments/assets/77d69826-9b1b-4ff9-a8c1-02934cca0388" />

---

### (Optional) Train the Model Using Google Colab

If you wish to train the model from scratch, you can use the provided https://colab.research.google.com/drive/11oeXcxLN2dHM9jyRHutJJyc7SGjmU73-?usp=sharing  
The model training file is also included in this repository for your reference or local training.

#### Steps to Replicate the Model in Colab:
1. Open the Colab Notebook:  
Click on the link above to open the Colab notebook in your browser.

2. Set Up the Environment:  
The notebook provides all the necessary setup to install libraries and dependencies.

3. Upload Your Dataset:  
You can upload your dataset or use the sample dataset provided.

4. Train the Model:  
Follow the instructions to train the classification model. Once trained, the model can classify recyclable materials (e.g., glass, plastic, paper, etc.).

5. Download the Model:  
After training, you can download the model weights and use them in your own application.

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
git clone https://github.com/charmainecmr/Part-2-EcoClassify.git
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

The application will:
1. Load the YOLOv8 model (`best.pt`)
2. Detect your GPU/CPU automatically
3. Open a webcam window for real-time detection
<img width="1590" height="930" alt="image" src="https://github.com/user-attachments/assets/a8bfc36a-f282-4b97-9efa-c4271da1fe03" />
---

## Controls

| Key | Action |
|-----|--------|
| **q** or **ESC** | Quit application |
| **s** | Save screenshot with timestamp |
| **r** | Reset session statistics |

---

