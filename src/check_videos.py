import cv2
from ultralytics import YOLO

def analyze_video(filepath, target_classes):
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n--- Analyzing {filepath} ---")
    print(f"FPS: {fps:.2f} | Total Frames: {total_frames} | Dimensions: {width}x{height}")
    
    # Load YOLO
    model = YOLO("yolo26n.pt")
    
    first_detect_frame = -1
    last_detect_frame = -1
    detections_count = 0
    
    frame_idx = 0
    while cap.isOpened():
        grabbed, frame = cap.read()
        if not grabbed:
            break
            
        # Run inference every 5 frames to speed up analysis
        if frame_idx % 5 == 0:
            results = model(frame, verbose=False)[0]
            detected = False
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls in target_classes and conf > 0.4:
                    detected = True
                    break
            
            if detected:
                detections_count += 1
                if first_detect_frame == -1:
                    first_detect_frame = frame_idx
                last_detect_frame = frame_idx
                
        frame_idx += 1
        
    cap.release()
    print(f"First Detection Frame: {first_detect_frame} (approx {first_detect_frame/fps:.2f}s)")
    print(f"Last Detection Frame: {last_detect_frame} (approx {last_detect_frame/fps:.2f}s)")
    print(f"Total detection samples: {detections_count}")
    return first_detect_frame, last_detect_frame, fps

if __name__ == "__main__":
    # Cam 1 targets: train (6)
    analyze_video("src/videos/Thai_train_passing.mp4", [6])
    
    # Cam 2 targets: person (0), car (2), motorcycle (3), bus (5), truck (7)
    analyze_video("src/videos/Cars_driving_railway_crossing.mp4", [0, 2, 3, 5, 7])
