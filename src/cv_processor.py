import cv2
import numpy as np
from ultralytics import YOLO

class TrainDetector:
    """
    Processes Camera 1 frames to detect the train approaching the crossing.
    Applies the Ultralytics YOLO26n model to identify 'train' objects
    and tracks their intersection with the sensor trip-line ROI.
    """
    def __init__(self, model=None, sensor_line_y=312, confidence_threshold=0.45):
        self.sensor_line_y = sensor_line_y
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO26n model (if a shared model instance is not passed)
        self.model = model or YOLO("yolo26n.pt")
        
        # State variables
        self.is_train_present = False
        self.consecutive_frames = 0
        self.trigger_threshold_frames = 2  # filter occasional brief false positives

    def process_frame(self, frame: np.ndarray):
        """
        Analyzes the frame and returns:
        - bool: True if train is detected crossing the sensor line, False otherwise
        - np.ndarray: The annotated frame with CV bounding boxes and status HUD
        """
        if frame is None:
            return False, None
            
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        # Run YOLO26n inference
        results = self.model(frame, verbose=False)[0]
        
        train_bbox = None
        detected_trains_at_line = 0
        
        # Iterate over bounding boxes
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # COCO Class ID 6 is 'train'
            # Let's also support large truck/vehicle (class 7) occasionally if class mapping skews
            if cls in [6, 7] and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, xyxy)
                train_bbox = (x1, y1, x2 - x1, y2 - y1)
                
                # Check intersection with sensor line
                if y1 <= self.sensor_line_y <= y2:
                    detected_trains_at_line += 1
                
                # Draw sleek neon cyan styling (BGR: [240, 240, 40])
                neon_cyan = [240, 240, 40]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), neon_cyan, 2, cv2.LINE_AA)
                
                # Label with confidence
                label = f"YOLO {self.model.names[cls]} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Semi-transparent label background
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1 - label_h - 6), (x1 + label_w + 10, y1), [40, 30, 20], -1)
                cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, neon_cyan, 1, cv2.LINE_AA)
                            
        # Decision logic for line crossing
        trip_detected = (detected_trains_at_line > 0)
        
        if trip_detected:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)
            
        # Detect transition edge trigger
        train_triggered = False
        if self.consecutive_frames >= self.trigger_threshold_frames:
            if not self.is_train_present:
                self.is_train_present = True
                train_triggered = True  # Edge trigger!
        else:
            if self.is_train_present and self.consecutive_frames == 0:
                self.is_train_present = False
        
        # Draw Sensor Line (Cam 1)
        sensor_color = [240, 240, 40] if not self.is_train_present else [40, 40, 255] # Cyan vs Red BGR
        x_start = int(w * 0.12)
        x_end = int(w * 0.88)
        cv2.line(annotated_frame, (x_start, self.sensor_line_y), (x_end, self.sensor_line_y), sensor_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, "YOLO RAIL SENSOR LINE", (x_start + 10, self.sensor_line_y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, sensor_color, 1, cv2.LINE_AA)
        
        # Render General Telemetry HUD on top-left
        hud_overlay = annotated_frame.copy()
        cv2.rectangle(hud_overlay, (10, 10), (220, 60), [30, 20, 15], -1)  # dark background
        cv2.addWeighted(hud_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "CAM 01 - APPROACH VIEW", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, [255, 255, 255], 1, cv2.LINE_AA)
        
        status_text = "STATUS: MONITORING" if not self.is_train_present else "STATUS: TRAIN PASSING"
        status_color = [100, 255, 100] if not self.is_train_present else [40, 40, 255] # Green vs Red
        cv2.putText(annotated_frame, status_text, (15, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

        return train_triggered, annotated_frame


class IntrusionDetector:
    """
    Processes Camera 2 frames inside a configured restricted zone polygon.
    Applies the Ultralytics YOLO26n model to identify 'person', 'car', 'truck', etc.,
    and verifies if their contact points intersect with the restricted area boundaries.
    """
    def __init__(self, zone_polygon: np.ndarray, model=None, confidence_threshold=0.35):
        self.zone_polygon = np.array(zone_polygon, dtype=np.int32)
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO26n model (if a shared model instance is not passed)
        self.model = model or YOLO("yolo26n.pt")
        
        # Target classes (COCO indices): 
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {0, 1, 2, 3, 5, 7}

    def process_frame(self, frame: np.ndarray, train_approaching: bool = False, enabled: bool = True):
        """
        Analyzes Camera 2 and checks for intrusions.
        Returns:
        - bool: True if intrusion detected, False otherwise
        - np.ndarray: The annotated frame with glowing hazard borders and intruder bbox
        """
        if frame is None:
            return False, None
            
        annotated_frame = frame.copy()
        
        # Run YOLO26n inference
        results = self.model(frame, verbose=False)[0]
        
        intrusion_active = False
        
        # Iterate over bounding boxes
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Check if class is in target classes and meets confidence threshold
            if cls in self.target_classes and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Compute bottom-center contact point on ground plane
                contact_x = int((x1 + x2) / 2)
                contact_y = y2
                
                # Check if contact point falls inside restricted zone polygon
                dist = cv2.pointPolygonTest(self.zone_polygon, (contact_x, contact_y), False)
                is_inside = dist >= 0
                
                # Draw contact point indicator
                cv2.circle(annotated_frame, (contact_x, contact_y), 4, [0, 255, 255], -1, cv2.LINE_AA)
                
                if is_inside and enabled:
                    intrusion_active = True
                    # Active intrusion: draw glowing red/amber box
                    alert_color = [40, 40, 255] if train_approaching else [40, 140, 255] # Red vs Amber
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), alert_color, 2, cv2.LINE_AA)
                    
                    label = f"WARNING: {self.model.names[cls]} {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    
                    # Semi-transparent background
                    box_overlay = annotated_frame.copy()
                    cv2.rectangle(box_overlay, (x1, y1 - label_h - 6), (x1 + label_w + 10, y1), [30, 20, 15], -1)
                    cv2.addWeighted(box_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1, cv2.LINE_AA)
                else:
                    # Object is on screen but outside danger zone: draw soft green/blue tracking box
                    safe_color = [100, 255, 100]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), safe_color, 1, cv2.LINE_AA)
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, safe_color, 1, cv2.LINE_AA)

        # --- Sleek Overlay Drawing ---
        if intrusion_active:
            alert_color = [40, 40, 255] if train_approaching else [40, 140, 255]
            status_text = "DANGER: BLOCKED ZONE!" if train_approaching else "WARNING: ZONE INTRUSION"
        else:
            alert_color = [240, 240, 40] if train_approaching else [100, 255, 100]
            status_text = "TRAIN EXPECTED - MONITORING" if train_approaching else "STATUS: ZONE CLEAR"
            
        # Draw translucent polygon overlay for restricted area
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [self.zone_polygon], alert_color)
        alpha = 0.22 if intrusion_active else 0.07
        cv2.addWeighted(overlay, alpha, annotated_frame, 1.0 - alpha, 0, annotated_frame)
        
        # Polygon border outline
        cv2.polylines(annotated_frame, [self.zone_polygon], True, alert_color, 2, cv2.LINE_AA)

        # Render Top-Left Info HUD
        hud_overlay = annotated_frame.copy()
        cv2.rectangle(hud_overlay, (10, 10), (280, 60), [30, 20, 15], -1)
        cv2.addWeighted(hud_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        cv2.putText(annotated_frame, "CAM 02 - CROSSING SAFETY ZONE", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, [255, 255, 255], 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, status_text, (15, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, alert_color, 1, cv2.LINE_AA)

        return intrusion_active, annotated_frame
