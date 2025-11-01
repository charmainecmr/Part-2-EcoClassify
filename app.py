import gradio as gr
import cv2
import numpy as np
from PIL import Image
from collections import Counter

# Load YOLO model
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

def load_model():
    if YOLO_AVAILABLE:
        model_path = "weights/best.pt"
        try:
            model = YOLO(model_path)
            print(f"YOLOv8 model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Could not load YOLO model: {e}")
    print("Using placeholder model until weights are ready.")
    return None

model = load_model()

# Global detection storage
detection_history = []
last_detected_materials = set()

def detect(frame):
    global detection_history, last_detected_materials
    
    if frame is None:
        return None, generate_stats_html([]), "", None
        
    if model is None:
        frame = np.array(frame)
        cv2.putText(frame, "Model not loaded yet", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame, generate_stats_html([]), "Model not loaded", None

    # Run detection
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    
    # Extract detections
    detections = []
    current_materials = set()
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = results[0].names[int(box.cls)]
            confidence = float(box.conf)
            detections.append({
                'label': label,
                'confidence': confidence
            })
            current_materials.add(label.lower())
    
    # Check for new materials detected
    new_materials = current_materials - last_detected_materials
    alert_html = None
    
    if new_materials:
        # Generate alert for new materials
        material = list(new_materials)[0]
        alert_html = generate_alert_html(material)
        last_detected_materials = current_materials
    elif not current_materials:
        last_detected_materials = set()
    
    # Update history (keep last 100)
    detection_history.extend(detections)
    detection_history = detection_history[-100:]
    
    # Generate stats
    stats_html = generate_stats_html(detections)
    status = f"Detected {len(detections)} items" if detections else "No items detected"
    
    return annotated_frame, stats_html, status, alert_html

def generate_alert_html(material):
    """Generate alert notification for detected material"""
    colors = {
        'plastic': ('#3b82f6', '#dbeafe'),
        'paper': ('#10b981', '#d1fae5'), 
        'metal': ('#f59e0b', '#fef3c7'),
        'glass': ('#8b5cf6', '#ede9fe'),
        'bottle': ('#3b82f6', '#dbeafe'),
        'can': ('#f59e0b', '#fef3c7')
    }
    
    emojis = {
        'plastic': '♻️',
        'paper': '📄',
        'metal': '🔩',
        'glass': '🍾',
        'bottle': '🍶',
        'can': '🥫'
    }
    
    color, bg_color = colors.get(material, ('#64748b', '#f1f5f9'))
    emoji = emojis.get(material, '♻️')
    
    # Audio element with beep sound
    audio_data = "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBjGH0fPTgjMGHm7A7+OZURE="
    
    return f"""
    <div style="position: fixed; top: 20px; right: 20px; z-index: 1000; 
                animation: slideIn 0.5s ease-out;">
        <div style="background: {bg_color}; border: 3px solid {color}; 
                    border-radius: 12px; padding: 16px 20px; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                    min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 32px;">{emoji}</div>
                <div>
                    <div style="font-weight: 700; color: {color}; font-size: 16px; 
                                text-transform: capitalize; margin-bottom: 4px;">
                        {material} Detected!
                    </div>
                    <div style="color: {color}; opacity: 0.8; font-size: 13px;">
                        New material identified
                    </div>
                </div>
            </div>
        </div>
        <audio autoplay>
            <source src="{audio_data}" type="audio/wav">
        </audio>
    </div>
    <style>
        @keyframes slideIn {{
            from {{
                transform: translateX(400px);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}
    </style>
    """

def generate_stats_html(current_detections):
    """Generate statistics panel HTML"""
    
    # Count current detections
    current_counts = Counter([d['label'] for d in current_detections])
    total_current = len(current_detections)
    
    # Count from history
    history_counts = Counter([d['label'] for d in detection_history])
    total_history = len(detection_history)
    
    # Material colors
    colors = {
        'plastic': '#3b82f6',
        'paper': '#10b981', 
        'metal': '#f59e0b',
        'glass': '#8b5cf6',
        'bottle': '#3b82f6',
        'can': '#f59e0b'
    }
    
    html = """
    <div style="font-family: system-ui, -apple-system, sans-serif;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white;">
            <h2 style="margin: 0 0 10px 0; font-size: 24px;">📊 Detection Statistics</h2>
            <p style="margin: 0; opacity: 0.9; font-size: 14px;">Real-time tracking</p>
        </div>
        
        <div style="background: #f8fafc; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 15px 0; font-size: 18px; color: #1e293b;">Current Frame</h3>
            <div style="font-size: 36px; font-weight: bold; color: #667eea; margin-bottom: 10px;">
                {total_current}
            </div>
            <div style="font-size: 14px; color: #64748b;">items detected</div>
    """.format(total_current=total_current)
    
    if current_counts:
        html += '<div style="margin-top: 15px;">'
        for label, count in current_counts.most_common():
            color = colors.get(label.lower(), '#94a3b8')
            html += f"""
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <div style="width: 12px; height: 12px; border-radius: 3px; 
                            background: {color}; margin-right: 8px;"></div>
                <span style="flex: 1; font-size: 14px; color: #475569;">{label}</span>
                <span style="font-weight: 600; color: #1e293b;">{count}</span>
            </div>
            """
        html += '</div>'
    
    html += """
        </div>
        
        <div style="background: #f1f5f9; padding: 20px; border-radius: 12px;">
            <h3 style="margin: 0 0 15px 0; font-size: 18px; color: #1e293b;">Session Total</h3>
            <div style="font-size: 28px; font-weight: bold; color: #10b981; margin-bottom: 10px;">
                {total_history}
            </div>
            <div style="font-size: 14px; color: #64748b;">total detections</div>
    """.format(total_history=total_history)
    
    if history_counts:
        html += '<div style="margin-top: 15px;">'
        for label, count in history_counts.most_common(5):
            color = colors.get(label.lower(), '#94a3b8')
            percentage = (count / total_history * 100) if total_history > 0 else 0
            html += f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 10px; height: 10px; border-radius: 2px; 
                                    background: {color}; margin-right: 6px;"></div>
                        <span style="font-size: 13px; color: #475569;">{label}</span>
                    </div>
                    <span style="font-size: 13px; font-weight: 600; color: #1e293b;">{count}</span>
                </div>
                <div style="background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="background: {color}; height: 100%; width: {percentage:.1f}%;"></div>
                </div>
            </div>
            """
        html += '</div>'
    
    html += """
        </div>
    </div>
    """
    
    return html

# Custom CSS
custom_css = """
#header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 20px;
}

#header h1 {
    color: white;
    font-size: 32px;
    margin: 0;
    font-weight: 700;
}

#header p {
    color: rgba(255, 255, 255, 0.9);
    margin: 8px 0 0 0;
    font-size: 16px;
}

.instructions {
    background: #f8fafc;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.instructions h3 {
    margin: 0 0 12px 0;
    color: #1e293b;
    font-size: 18px;
}

.instructions ol {
    margin: 0;
    padding-left: 20px;
    color: #64748b;
}

.instructions li {
    margin: 6px 0;
    line-height: 1.6;
}

.status-box {
    background: #f1f5f9;
    padding: 12px 16px;
    border-radius: 8px;
    font-weight: 500;
    text-align: center;
    border: 2px solid #e2e8f0;
}

#camera-feed {
    border-radius: 12px;
    border: 2px solid #e2e8f0;
}

footer {
    display: none !important;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css, title="♻️ Eco Classify Part 2 - Real-Time Detection") as demo:
    
    # Header
    with gr.Row(elem_id="header"):
        with gr.Column():
            gr.Markdown("""
            # Eco Classify Part 2 - Real-Time Detection Powered by YOLOv8
            """)
    
    with gr.Row():
        # Left column - Camera and instructions
        with gr.Column(scale=2):
            camera_input = gr.Image(
                sources=["webcam"], 
                label="📹 Live Camera Feed",
                type="numpy",
                streaming=True,
                elem_id="camera-feed"
            )
            
            status_text = gr.Textbox(
                label="Status",
                value="🎥 Ready to detect",
                interactive=False,
                elem_classes="status-box"
            )
            
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%); 
                        padding: 24px; border-radius: 12px; border: 2px solid #e2e8f0; margin-top: 16px;">
                <h3 style="margin: 0 0 16px 0; color: #1e293b; font-size: 18px; 
                           display: flex; align-items: center; gap: 8px; font-weight: 600;">
                    <span>How to use</span>
                </h3>
                <ol style="margin: 0; padding-left: 20px; color: #475569; line-height: 1.8;">
                    <li><strong>Click the camera icon</strong> above to start webcam</li>
                    <li><strong>Point camera</strong> at recyclable items</li>
                    <li><strong>View detections</strong> with bounding boxes in real-time</li>
                    <li><strong>Check statistics</strong> panel for material breakdown</li>
                </ol>
                <div style="background: #fef3c7; padding: 12px 16px; border-radius: 8px; 
                            margin-top: 16px; border-left: 4px solid #f59e0b;">
                    <strong style="color: #92400e;">Tip:</strong>
                    <span style="color: #78350f;"> Ensure good lighting for best results!</span>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 16px;">
                <div style="background: #f1f5f9; padding: 20px; border-radius: 12px; border: 1px solid #cbd5e1;">
                    <h4 style="margin: 0 0 8px 0; color: #334155; font-size: 15px;">🎯 Custom Model</h4>
                    <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.6;">
                        To use your trained weights, place them in <code style="background: #e2e8f0; 
                        padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #1e293b;">weights/best.pt</code>
                    </p>
                </div>
                
                <div style="background: linear-gradient(135deg, #ecfccb 0%, #d9f99d 100%); 
                            padding: 20px; border-radius: 12px; border: 1px solid #bef264;">
                    <h4 style="margin: 0 0 12px 0; color: #365314; font-size: 15px;">
                        Supported Materials
                    </h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #10b981; color: white; padding: 6px 12px; 
                                     border-radius: 6px; font-size: 13px; font-weight: 500;">Paper</span>
                        <span style="background: #3b82f6; color: white; padding: 6px 12px; 
                                     border-radius: 6px; font-size: 13px; font-weight: 500;">Plastic</span>
                        <span style="background: #f59e0b; color: white; padding: 6px 12px; 
                                     border-radius: 6px; font-size: 13px; font-weight: 500;">Metal</span>
                        <span style="background: #8b5cf6; color: white; padding: 6px 12px; 
                                     border-radius: 6px; font-size: 13px; font-weight: 500;">Glass</span>
                    </div>
                </div>
            </div>
            """)
        
        # Right column - Stats and info
        with gr.Column(scale=1):
            output_image = gr.Image(
                label="🎯 Detection Output",
                type="numpy"
            )
            
            stats_html = gr.HTML(
                value=generate_stats_html([]),
                label="Statistics"
            )
    
    # Set up streaming detection
    camera_input.stream(
        fn=detect,
        inputs=[camera_input],
        outputs=[output_image, stats_html, status_text],
        time_limit=30,
        stream_every=0.5
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )