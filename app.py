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
        model_path = "best.pt"
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

# SVG Icons
def get_material_icon(material):
    """Return SVG icon for material type"""
    icons = {
        'plastic': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>''',
        'paper': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 13H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 17H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M10 9H9H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>''',
        'metal': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
            <circle cx="12" cy="12" r="6" stroke="currentColor" stroke-width="2"/>
            <circle cx="12" cy="12" r="2" fill="currentColor"/>
        </svg>''',
        'glass': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6 2L5 8H19L18 2H6Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M5 8L7 22H17L19 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="9" y1="12" x2="15" y2="12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>''',
        'recycle': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 8L21 12L17 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M3 12H21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M7 16L3 12L7 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>'''
    }
    return icons.get(material, icons['recycle'])

def get_bell_icon():
    """Return animated bell SVG icon"""
    return '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="animation: ring 0.5s ease-in-out 3;">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>'''

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
    colors = {
        'plastic': ('#3b82f6', '#dbeafe'),
        'paper': ('#10b981', '#d1fae5'), 
        'metal': ('#f59e0b', '#fef3c7'),
        'glass': ('#8b5cf6', '#ede9fe'),
    }
    
    color, bg_color = colors.get(material, ('#64748b', '#f1f5f9'))
    icon_svg = get_material_icon(material)
    bell_icon = get_bell_icon()
    
    # Enhanced bell sound (longer, clearer notification sound)
    bell_audio = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
    
    return f"""
    <div id="alert-notification" style="position: fixed; top: 20px; right: 20px; z-index: 1000; 
                animation: slideInBounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);">
        <div style="background: {bg_color}; border: 3px solid {color}; 
                    border-radius: 16px; padding: 20px 24px; 
                    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
                    min-width: 280px; position: relative; overflow: hidden;">
            
            <!-- Bell icon in top left corner -->
            <div style="position: absolute; top: 12px; left: 12px; color: {color}; opacity: 0.3;">
                {bell_icon}
            </div>
            
            <div style="display: flex; align-items: center; gap: 16px; margin-top: 8px;">
                <div style="color: {color}; transform: scale(1.2);">{icon_svg}</div>
                <div style="flex: 1;">
                    <div style="font-weight: 800; color: {color}; font-size: 20px; 
                                text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px;">
                        🔔 {material} Detected!
                    </div>
                    <div style="color: {color}; opacity: 0.85; font-size: 14px; font-weight: 500;">
                        Recyclable material identified
                    </div>
                </div>
            </div>
            
            <!-- Progress bar animation -->
            <div style="position: absolute; bottom: 0; left: 0; right: 0; height: 4px; 
                        background: rgba(255,255,255,0.3); overflow: hidden;">
                <div style="height: 100%; background: {color}; 
                            animation: progress 3s linear forwards;"></div>
            </div>
        </div>
        
        <!-- Bell notification sound - plays 3 times -->
        <audio id="notification-bell" autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
        </audio>
    </div>
    
    <style>
        @keyframes slideInBounce {{
            0% {{
                transform: translateX(400px) scale(0.8);
                opacity: 0;
            }}
            60% {{
                transform: translateX(-20px) scale(1.05);
                opacity: 1;
            }}
            80% {{
                transform: translateX(10px) scale(0.98);
            }}
            100% {{
                transform: translateX(0) scale(1);
                opacity: 1;
            }}
        }}
        
        @keyframes ring {{
            0%, 100% {{ transform: rotate(0deg); }}
            10%, 30% {{ transform: rotate(-15deg); }}
            20%, 40% {{ transform: rotate(15deg); }}
        }}
        
        @keyframes progress {{
            from {{ width: 100%; }}
            to {{ width: 0%; }}
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.1); opacity: 0.8; }}
        }}
        
        #alert-notification {{
            animation: slideInBounce 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55),
                       fadeOut 0.5s ease-out 2.5s forwards;
        }}
        
        @keyframes fadeOut {{
            to {{
                opacity: 0;
                transform: translateX(400px);
            }}
        }}
    </style>
    
    <script>
        // Play bell sound multiple times
        (function() {{
            const audio = document.getElementById('notification-bell');
            let playCount = 0;
            if (audio) {{
                audio.addEventListener('ended', function() {{
                    playCount++;
                    if (playCount < 2) {{ // Play 2 times total
                        this.currentTime = 0;
                        this.play();
                    }}
                }});
            }}
        }})();
    </script>
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
    }
    
    html = """
    <div style="font-family: system-ui, -apple-system, sans-serif;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 12px; margin-bottom: 20px; color: white;">
            <h2 style="margin: 0 0 10px 0; font-size: 24px;">Detection Statistics</h2>
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

ol li::marker {
    color: #000000 !important;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css, title="Eco Classify Part 2 - Real-Time Detection") as demo:
    
    # Header with icon
    with gr.Row(elem_id="header"):
        with gr.Column():
            gr.Markdown("""
            # 🔔 Eco Classify Part 2 - Real-Time Detection with Notifications
            ### Powered by YOLOv8 with Bell Alerts
            """)
    
    with gr.Row():
        # Left column - Camera and instructions
        with gr.Column(scale=2):
            camera_input = gr.Image(
                sources=["webcam"], 
                label="Live Camera Feed",
                type="numpy",
                streaming=True,
                elem_id="camera-feed"
            )
                       
            gr.HTML("""
            <div style="background: white; padding: 24px; border-radius: 12px; 
                        border: 2px solid #e2e8f0; margin-top: 16px; 
                        box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <h3 style="margin: 0 0 16px 0; color: #1e293b; font-size: 20px; 
                           display: flex; align-items: center; gap: 8px; font-weight: 700;">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 16v-4M12 8h.01"/>
                    </svg>
                    <span><strong style="color: #0f172a;">How to use</strong></span>
                </h3>
                <ol style="margin: 0; padding-left: 24px; color: #334155; line-height: 2; font-size: 15px;">
                    <strong style="color: #0f172a;">
                    <li style="margin-bottom: 8px;"><strong style="color: #0f172a;">Click the camera icon above to start webcam</strong></li>
                    <li style="margin-bottom: 8px;"><strong style="color: #0f172a;">Point camera at recyclable items</strong></li>
                    <li style="margin-bottom: 8px;"><strong style="color: #0f172a;">🔔 Listen for bell notification when items detected</strong></li>
                    <li style="margin-bottom: 8px;"><strong style="color: #0f172a;">View detections with bounding boxes in real-time</strong></li>
                    <li style="margin-bottom: 8px;"><strong style="color: #0f172a;">Check statistics panel for material breakdown</strong></li>
                    </strong>
                </ol>
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                            padding: 14px 18px; border-radius: 10px; 
                            margin-top: 20px; border-left: 5px solid #f59e0b;
                            box-shadow: 0 2px 6px rgba(245, 158, 11, 0.15);
                            display: flex; align-items: center; gap: 10px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#92400e" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 16v-4M12 8h.01"/>
                    </svg>
                    <div>
                        <strong style="color: #92400e; font-size: 15px;">Tip:</strong>
                        <span style="color: #78350f; font-size: 14px;"> Enable sound for notification alerts!</span>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 16px;">
                <div style="background: #f1f5f9; padding: 20px; border-radius: 12px; border: 1px solid #cbd5e1;">
                    <h4 style="margin: 0 0 8px 0; color: #334155; font-size: 15px; display: flex; align-items: center; gap: 8px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                            <polyline points="13 2 13 9 20 9"/>
                        </svg>
                        Custom Model
                    </h4>
                    <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.6;">
                        To use your trained weights, place them in <code style="background: #e2e8f0; 
                        padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #1e293b;">best.pt</code>
                    </p>
                </div>
                
                <div style="background: linear-gradient(135deg, #ecfccb 0%, #d9f99d 100%); 
                            padding: 20px; border-radius: 12px; border: 1px solid #bef264;">
                    <h4 style="margin: 0 0 12px 0; color: #365314; font-size: 15px; display: flex; align-items: center; gap: 8px;">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#365314" stroke-width="2">
                            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                        </svg>
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
                label="Detection Output",
                type="numpy"
            )
            
            alert_box = gr.HTML(
                value="",
                label="Notifications"
            )
            
            stats_html = gr.HTML(
                value=generate_stats_html([]),
                label="Statistics"
            )
    
    # Set up streaming detection
    camera_input.stream(
        fn=detect,
        inputs=[camera_input],
        outputs=[output_image, stats_html, gr.Textbox(visible=False), alert_box],
        time_limit=30,
        stream_every=0.5
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )