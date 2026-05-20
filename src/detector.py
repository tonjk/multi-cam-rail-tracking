from ultralytics import YOLO
import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.model = YOLO("yolo26n.pt")
        # COCO classes for person and vehicles: 
        # 0: person, 2: car, 3: motorcycle, 5: bus, 6: train, 7: truck
        self.target_classes = [0, 2, 3, 5, 6, 7]
        
        # Color mapping (BGR format for OpenCV, visually vibrant)
        self.class_colors = {
            0: (255, 200, 0),    # Cyan/Blue for person
            2: (0, 200, 255),    # Gold/Amber for car
            3: (200, 50, 255),   # Pinkish for motorcycle
            5: (100, 255, 100),  # Light green for bus
            6: (50, 50, 255),    # Red for train
            7: (255, 150, 50)    # Orange for truck
        }

    def predict(self, frame):
        # Filter predictions by the specified target classes
        results = self.model(frame, classes=self.target_classes, conf=0.5, iou=0.8, verbose=False)
        return results

    def draw_results(self, frame, results):
        annotated = frame.copy()
        
        if len(results) == 0:
            return annotated
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = result.names[cls_id]
            
            # Premium aesthetics colors
            color = self.class_colors.get(cls_id, (255, 255, 255))
            
            # Draw bounding box with anti-aliasing
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            
            # Draw semi-transparent background for text
            label = f"{name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 0, 0), -1)
            # Alpha blend the overlay with the annotated frame
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            
            # Draw text with anti-aliasing
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, color, 1, lineType=cv2.LINE_AA)
                        
        return annotated
