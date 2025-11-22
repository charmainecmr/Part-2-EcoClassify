"""
Eco Classify - Real-Time Recyclables Detection System
Uses YOLOv8 for real-time object detection of recyclable materials
Supports: Plastic, Paper, Metal, Glass
"""

import cv2
import numpy as np
import torch
from collections import Counter
import time
from datetime import datetime

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

# Try to import YOLOv8 from ultralytics library
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    exit(1)


# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class RecyclablesDetector:
    """
    Real-time recyclables detection system using YOLOv8
    
    Features:
    - GPU/CPU automatic detection
    - Real-time alerts when new materials detected
    - Session statistics tracking
    - FP16 half-precision for GPU acceleration
    """
    
    def __init__(self, model_path="best.pt", confidence=0.5):
        """
        Initialize the detector with YOLOv8 model
        
        Args:
            model_path (str): Path to trained YOLOv8 model weights (.pt file)
            confidence (float): Confidence threshold for detections (0.0-1.0)
        """
        self.model_path = model_path
        self.confidence = confidence
        
        # ====================================================================
        # MODEL LOADING
        # ====================================================================
        print(f"Loading YOLOv8 model from {model_path}...")
        try:
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            
            # Auto-detect GPU (CUDA) or CPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Enable FP16 half-precision on GPU for 2x speedup
            self.use_half = self.device == 'cuda'
            
            # Display model loading info
            print(f"✓ Model loaded successfully")
            print(f"✓ Device: {self.device.upper()}")
            if self.device == 'cuda':
                print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
                print(f"✓ Using FP16 half-precision")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
        
        # ====================================================================
        # DETECTION TRACKING VARIABLES
        # ====================================================================
        
        # Store last 100 detections for session statistics
        self.detection_history = []
        
        # Track currently visible materials (for alert triggering)
        self.last_detected_materials = set()
        
        # Alert timing variables
        self.alert_start_time = None        # When alert started
        self.current_alert_material = None  # Which material triggered alert
        
        # Frame counter for stats update optimization
        self.frame_count = 0
        self.stats_update_interval = 5  # Update stats every 5 frames
        
        # ====================================================================
        # MATERIAL COLOR MAPPING (BGR format for OpenCV)
        # ====================================================================
        self.material_colors = {
            'plastic': (246, 130, 59),   # Blue
            'paper': (129, 185, 16),     # Green
            'metal': (11, 158, 245),     # Orange
            'glass': (246, 92, 139),     # Purple
        }
        
        # Cache for statistics to avoid regenerating every frame
        self.stats_cache = {"last_hash": None, "data": None}
    
    def detect_frame(self, frame):
        """
        Run YOLOv8 detection on a single frame
        
        Args:
            frame (numpy.ndarray): Input frame from webcam (BGR format)
            
        Returns:
            tuple: (annotated_frame, detections, current_materials)
                - annotated_frame: Frame with bounding boxes drawn
                - detections: List of detection dictionaries
                - current_materials: Set of detected material names
        """
        # Increment frame counter
        self.frame_count += 1
        
        # ====================================================================
        # RUN YOLOV8 INFERENCE
        # ====================================================================
        results = self.model.predict(
            frame,
            imgsz=416,              # Resize to 416x416 for speed (vs 640 default)
            conf=self.confidence,    # Confidence threshold
            iou=0.45,               # IOU threshold for Non-Max Suppression
            verbose=False,          # Suppress console output
            device=self.device,     # Use GPU or CPU
            half=self.use_half,     # FP16 precision on GPU
            max_det=300             # Maximum detections per frame
        )
        
        # Get frame with bounding boxes drawn by YOLOv8
        annotated_frame = results[0].plot()
        
        # ====================================================================
        # EXTRACT DETECTION INFORMATION
        # ====================================================================
        detections = []
        current_materials = set()
        
        # Check if any objects were detected
        if len(results[0].boxes) > 0:
            # Loop through each detected bounding box
            for box in results[0].boxes:
                # Get class label (e.g., "plastic", "paper")
                label = results[0].names[int(box.cls)]
                
                # Get confidence score (0.0-1.0)
                confidence = float(box.conf)
                
                # Store detection info
                detections.append({
                    'label': label,
                    'confidence': confidence
                })
                
                # Add to current materials set (lowercase for consistency)
                current_materials.add(label.lower())
        
        # ====================================================================
        # ALERT TRIGGERING LOGIC
        # ====================================================================
        
        # Find newly detected materials (not in previous frame)
        new_materials = current_materials - self.last_detected_materials
        
        if new_materials:
            # New material detected! Trigger alert
            material = list(new_materials)[0]  # Get first new material
            self.current_alert_material = material
            self.alert_start_time = time.time()  # Start alert timer
            self.last_detected_materials = current_materials
        elif not current_materials:
            # No materials detected - reset tracking
            self.last_detected_materials = set()
        
        # ====================================================================
        # UPDATE DETECTION HISTORY
        # ====================================================================
        
        # Add current detections to history
        self.detection_history.extend(detections)
        
        # Keep only last 100 detections (sliding window)
        self.detection_history = self.detection_history[-100:]
        
        return annotated_frame, detections, current_materials
    
    def draw_alert(self, frame, elapsed_time):
        """
        Draw alert notification overlay on frame
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            elapsed_time (float): Seconds since alert started
            
        Returns:
            numpy.ndarray: Frame with alert overlay
        """
        # Check if alert is active
        if self.current_alert_material is None:
            return frame
        
        material = self.current_alert_material
        color = self.material_colors.get(material, (100, 116, 139))
        
        # ====================================================================
        # CALCULATE FADE-OUT ANIMATION
        # ====================================================================
        total_duration = 8.0  # Alert displays for 8 seconds
        fade_start = 7.0      # Start fading at 7 seconds
        
        if elapsed_time > fade_start:
            # Calculate opacity during fade-out (1.0 → 0.0)
            opacity = 1 - ((elapsed_time - fade_start) / (total_duration - fade_start))
            opacity = max(0, min(1, opacity))  # Clamp to [0, 1]
        else:
            opacity = 1  # Full opacity
        
        # Don't draw if fully faded
        if opacity <= 0:
            return frame
        
        # ====================================================================
        # DRAW ALERT BOX
        # ====================================================================
        
        # Create overlay for transparency effect
        overlay = frame.copy()
        
        # Alert box position and dimensions
        alert_x, alert_y = 30, 30
        alert_w, alert_h = 400, 100
        
        # Draw filled rectangle (background)
        cv2.rectangle(overlay, (alert_x, alert_y), 
                     (alert_x + alert_w, alert_y + alert_h), 
                     color, -1)  # -1 = filled
        
        # Draw border
        cv2.rectangle(overlay, (alert_x, alert_y), 
                     (alert_x + alert_w, alert_y + alert_h), 
                     color, 3)  # 3px thick border
        
        # ====================================================================
        # ADD TEXT TO ALERT
        # ====================================================================
        
        # Main alert text
        text = f"{material.upper()} DETECTED!"
        cv2.putText(overlay, text, (alert_x + 20, alert_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Subtitle text
        cv2.putText(overlay, "Recyclable material identified", 
                   (alert_x + 20, alert_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ====================================================================
        # PROGRESS BAR (shows time remaining)
        # ====================================================================
        
        # Calculate remaining width (shrinks from 100% to 0%)
        progress_width = int((1 - elapsed_time / total_duration) * alert_w)
        progress_width = max(0, min(alert_w, progress_width))
        
        # Draw progress bar at bottom of alert
        cv2.rectangle(overlay, (alert_x, alert_y + alert_h - 5),
                     (alert_x + progress_width, alert_y + alert_h),
                     (255, 255, 255), -1)
        
        # ====================================================================
        # BLEND OVERLAY WITH FRAME (for transparency)
        # ====================================================================
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity * 0.7, 0, frame)
        
        return frame
    
    def draw_statistics(self, frame, detections):
        """
        Draw statistics panel on frame (with caching for performance)
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            detections (list): Current frame detections
            
        Returns:
            numpy.ndarray: Frame with statistics overlay
        """
        # ====================================================================
        # CHECK IF CACHE CAN BE USED (every 5 frames)
        # ====================================================================
        if self.frame_count % self.stats_update_interval != 0 and self.stats_cache["data"] is not None:
            # Use cached stats (avoid recalculating every frame)
            return self.draw_cached_stats(frame)
        
        # ====================================================================
        # CALCULATE CURRENT STATISTICS
        # ====================================================================
        
        # Count detections by material type in current frame
        current_counts = Counter([d['label'] for d in detections])
        total_current = len(detections)
        
        # Count detections from entire session history
        history_counts = Counter([d['label'] for d in self.detection_history])
        total_history = len(self.detection_history)
        
        # Cache the calculated data
        self.stats_cache["data"] = {
            'current_counts': current_counts,
            'total_current': total_current,
            'history_counts': history_counts,
            'total_history': total_history
        }
        
        # Draw using cached data
        return self.draw_cached_stats(frame)
    
    def draw_cached_stats(self, frame):
        """
        Draw statistics panel using cached data
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            
        Returns:
            numpy.ndarray: Frame with statistics overlay
        """
        data = self.stats_cache["data"]
        if data is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # ====================================================================
        # STATS PANEL POSITION (top-right corner)
        # ====================================================================
        panel_w = 280
        panel_x = w - panel_w - 20
        panel_y = 20
        
        # ====================================================================
        # DRAW SEMI-TRANSPARENT BACKGROUND
        # ====================================================================
        overlay = frame.copy()
        
        # Draw dark gray rectangle
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + 200),
                     (40, 40, 40), -1)  # Dark gray
        
        # Blend with frame (85% overlay, 15% original)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # ====================================================================
        # DRAW HEADER
        # ====================================================================
        cv2.putText(frame, "STATISTICS", (panel_x + 15, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ====================================================================
        # DRAW CURRENT FRAME COUNT
        # ====================================================================
        y_offset = panel_y + 70
        
        # Label
        cv2.putText(frame, "Current Frame:", (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Count (large blue number)
        cv2.putText(frame, str(data['total_current']), 
                   (panel_x + 15, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (102, 126, 234), 2)
        
        # ====================================================================
        # DRAW SESSION TOTAL
        # ====================================================================
        y_offset += 70
        
        # Label
        cv2.putText(frame, "Session Total:", (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Count (large green number)
        cv2.putText(frame, str(data['total_history']), 
                   (panel_x + 15, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (16, 185, 129), 2)
        
        # ====================================================================
        # DRAW MATERIAL BREAKDOWN (bottom section)
        # ====================================================================
        y_offset = panel_y + 210
        
        if data['history_counts']:
            # Show top 4 detected materials
            for i, (label, count) in enumerate(data['history_counts'].most_common(4)):
                # Get material color
                color = self.material_colors.get(label.lower(), (148, 163, 184))
                
                # Draw color indicator (circle)
                cv2.circle(frame, (panel_x + 20, y_offset + i * 25), 5, color, -1)
                
                # Draw material name and count
                text = f"{label}: {count}"
                cv2.putText(frame, text, (panel_x + 35, y_offset + i * 25 + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_stream(self):
        """
        Main processing loop for webcam stream
        Handles:
        - Frame capture
        - Detection processing
        - Overlay rendering
        - FPS calculation
        - Keyboard controls
        """
        # ====================================================================
        # INITIALIZE WEBCAM
        # ====================================================================
        cap = cv2.VideoCapture(0)  # 0 = default webcam
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # ====================================================================
        # DISPLAY STARTUP INFO
        # ====================================================================
        print("\n" + "="*60)
        print(" REAL-TIME RECYCLABLES DETECTION (MIRRORED)")
        print("="*60)
        print("Press 'q' or ESC to quit")
        print("Press 's' to take a screenshot")
        print("Press 'r' to reset statistics")
        print("="*60 + "\n")
        
        # ====================================================================
        # FPS CALCULATION VARIABLES
        # ====================================================================
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_display = 0
        
        # ====================================================================
        # MAIN LOOP
        # ====================================================================
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            # Check if frame was read successfully
            if not ret:
                print("Error: Cannot read frame from webcam")
                break
            
            # ----------------------------------------------------------------
            # MIRROR THE FRAME (like looking in a mirror)
            # ----------------------------------------------------------------
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip
            
            # ----------------------------------------------------------------
            # RUN DETECTION
            # ----------------------------------------------------------------
            annotated_frame, detections, current_materials = self.detect_frame(frame)
            
            # ----------------------------------------------------------------
            # DRAW ALERT IF ACTIVE
            # ----------------------------------------------------------------
            if self.alert_start_time is not None and self.current_alert_material is not None:
                # Calculate elapsed time since alert started
                elapsed = time.time() - self.alert_start_time
                
                if elapsed < 8.0:  # Alert lasts 8 seconds
                    annotated_frame = self.draw_alert(annotated_frame, elapsed)
                else:
                    # Alert expired - clear it
                    self.alert_start_time = None
                    self.current_alert_material = None
            
            # ----------------------------------------------------------------
            # DRAW STATISTICS PANEL
            # ----------------------------------------------------------------
            annotated_frame = self.draw_statistics(annotated_frame, detections)
            
            # ----------------------------------------------------------------
            # CALCULATE AND DISPLAY FPS
            # ----------------------------------------------------------------
            fps_frame_count += 1
            
            # Update FPS every second
            if time.time() - fps_start_time > 1.0:
                fps_display = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Draw FPS counter (bottom-left corner)
            cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", 
                       (20, annotated_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # ----------------------------------------------------------------
            # DISPLAY FRAME
            # ----------------------------------------------------------------
            cv2.imshow("Eco Classify - Real-Time Detection", annotated_frame)
            
            # ----------------------------------------------------------------
            # HANDLE KEYBOARD INPUT
            # ----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF  # Wait 1ms for key press
            
            if key == ord('q') or key == 27:  # 'q' or ESC key
                break  # Exit loop
                
            elif key == ord('s'):  # Screenshot
                # Generate timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                
                # Save frame
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
                
            elif key == ord('r'):  # Reset statistics
                self.detection_history = []
                self.stats_cache = {"last_hash": None, "data": None}
                print("Statistics reset")
        
        # ====================================================================
        # CLEANUP
        # ====================================================================
        cap.release()  # Release webcam
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print("\nSession ended. Total detections:", len(self.detection_history))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the application
    Initializes detector and starts webcam stream
    """
    # Initialize detector with trained YOLOv8 model
    detector = RecyclablesDetector(
        model_path="best.pt",  # Path to your trained model
        confidence=0.5         # Confidence threshold (0.5 = 50%)
    )
    
    # Start real-time detection
    detector.process_stream()


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
