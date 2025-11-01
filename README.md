---
title: EcoClassify Part 2
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.31.4
app_file: app.py
pinned: false
---

# EcoClassify Part 2: Real-Time Recyclables Detection

**EcoClassify Part 2** is an AI-powered web application that detects and classifies recyclable materials such as **plastic, paper, glass, metal, and cardboard** in real time using a custom-trained **YOLOv8** object detection model.  
The app is built with **Gradio**, featuring a clean, responsive interface and a live statistics dashboard that visualizes detection frequency and material breakdown.

---

## Features

- **Live webcam detection** with bounding boxes  
- **Custom YOLOv8 model** trained on recyclable materials  
- **Real-time statistics** panel showing detected material counts  
- **Runs locally or on Hugging Face Spaces** (CPU-compatible)  
- **Optimized for scalability** — easy to retrain and redeploy with new datasets  

---

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