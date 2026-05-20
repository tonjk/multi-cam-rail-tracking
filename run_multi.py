import cv2
import numpy as np
import argparse
from datetime import datetime
from src.detector import Detector

def parse_arguments():
    parser = argparse.ArgumentParser(description="M.E.N.T.O.R. Multi-Cam Rail Tracking System Dashboard")
    parser.add_argument(
        "--no-yolo", 
        action="store_true", 
        help="Disable YOLO object detection on startup (can be toggled in real-time using 'y')"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=None, 
        help="Override the native video playback frame rate (FPS)"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Video source paths
    video_paths = [
        "videos/long_train_1.mp4",
        "videos/long_train_2.mp4",
        "videos/long_train_3.mp4"
    ]
    
    # Custom names for camera streams
    cam_names = [
        "CAM 01 // OVERVIEW",
        "CAM 02 // EAST RAIL",
        "CAM 03 // CROSSING"
    ]
    
    # Target frame dimension per stream
    stream_w = 640
    stream_h = 360
    
    # Header dashboard height
    header_h = 45
    
    # Combined canvas size
    canvas_w = stream_w * len(video_paths)
    canvas_h = stream_h + header_h
    
    # Initialize detector
    detector = Detector()
    
    # Open captures
    caps = []
    native_fps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {path}")
            # Release any successfully opened captures before exiting
            for c in caps:
                c.release()
            return
        caps.append(cap)
        # Try to read native video FPS
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        if fps_val > 0:
            native_fps.append(fps_val)
        else:
            native_fps.append(30.0)
            
    # Determine the playback delay to match native video frame rates ("run by own frame")
    target_fps = args.fps if args.fps is not None else (sum(native_fps) / len(native_fps))
    playback_delay = max(1, int(1000 / target_fps))
    
    print("\n" + "="*60)
    print("  M.E.N.T.O.R. MULTI-CAMERA RAIL TRACKING DASHBOARD")
    print("="*60)
    print(f" * Synchronized FPS  : {target_fps:.2f} ({playback_delay} ms/frame)")
    print(f" * YOLO Detection    : {'DISABLED' if args.no_yolo else 'ENABLED'}")
    print("\n KEYBOARD CONTROLS:")
    print("  [Space]  : Pause / Play")
    print("  [Y]      : Toggle YOLO Detection On/Off dynamically")
    print("  [Any Key]: Step 1 frame forward when paused")
    print("  [Q]      : Exit Dashboard")
    print("="*60 + "\n")
    
    frame_count = 0
    use_yolo = not args.no_yolo
    paused = False
    step_one_frame = False
    
    # Cache to store the latest raw frames for re-rendering when toggling options in pause mode
    current_raw_frames = [None] * len(caps)
    
    try:
        while True:
            annotated_frames = []
            detections_counts = []
            
            # Determine if we should read a new frame
            should_read = (not paused) or step_one_frame
            step_one_frame = False # Reset single-step trigger
            
            for i, cap in enumerate(caps):
                if should_read:
                    ret, frame = cap.read()
                    
                    # Auto-loop video if it ends
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            print(f"Error: Failed to read frame from {cam_names[i]} even after loop reset.")
                            frame = np.zeros((stream_h, stream_w, 3), dtype=np.uint8)
                    
                    current_raw_frames[i] = frame.copy()
                else:
                    # Retrieve the cached frame if we are paused to avoid blank screens
                    frame = current_raw_frames[i].copy() if current_raw_frames[i] is not None else np.zeros((stream_h, stream_w, 3), dtype=np.uint8)
                
                # Run YOLO prediction if enabled
                count = 0
                if use_yolo:
                    results = detector.predict(frame)
                    if len(results) > 0 and results[0].boxes is not None:
                        count = len(results[0].boxes)
                    annotated = detector.draw_results(frame, results)
                else:
                    annotated = frame.copy()
                    
                detections_counts.append(count)
                
                # Resize to target display dimensions
                annotated_resized = cv2.resize(annotated, (stream_w, stream_h))
                
                # Apply high-fidelity camera stream overlay
                # Create semi-transparent top overlay for camera info
                overlay = annotated_resized.copy()
                cv2.rectangle(overlay, (0, 0), (stream_w, 35), (10, 10, 12), -1)
                cv2.addWeighted(overlay, 0.65, annotated_resized, 0.35, 0, annotated_resized)
                
                # Draw camera name
                cv2.putText(
                    annotated_resized, 
                    cam_names[i], 
                    (12, 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (240, 240, 245), 
                    1, 
                    lineType=cv2.LINE_AA
                )
                
                # Draw detection counter
                if use_yolo:
                    det_text = f"TRACKS: {count:02d}"
                    det_color = (0, 200, 255) if count > 0 else (140, 140, 145)
                else:
                    det_text = "YOLO: OFF"
                    det_color = (120, 120, 125)
                    
                cv2.putText(
                    annotated_resized,
                    det_text,
                    (stream_w - 180, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    det_color,
                    1,
                    lineType=cv2.LINE_AA
                )
                
                # Draw camera status indicators (LIVE / PAUSED)
                if paused:
                    status_text = "PAUSED"
                    status_color = (0, 165, 255) # Orange/Amber
                    indicator_color = (0, 165, 255)
                else:
                    status_text = "LIVE"
                    status_color = (100, 255, 100) # Green
                    # Blinking LIVE indicator (pulsing every 15 frames)
                    blink_on = (frame_count // 15) % 2 == 0
                    indicator_color = (100, 255, 100) if blink_on else (40, 120, 40)
                
                # Draw small status indicator circle and label
                cv2.circle(annotated_resized, (stream_w - 85, 17), 4, indicator_color, -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    annotated_resized,
                    status_text,
                    (stream_w - 73, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    status_color,
                    1,
                    lineType=cv2.LINE_AA
                )
                
                annotated_frames.append(annotated_resized)
                
            # Construct unified dashboard
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            
            # 1. Draw Dashboard Header (Top 45 pixels)
            header = canvas[0:header_h, 0:canvas_w]
            header[:] = (18, 18, 20)  # Sleek dark background
            
            # Bottom accent line to separate header
            accent_color = (255, 200, 0) if use_yolo else (120, 120, 125)
            cv2.line(header, (0, header_h - 1), (canvas_w, header_h - 1), accent_color, 1, lineType=cv2.LINE_AA)
            
            # System Title
            title_text = "M.E.N.T.O.R. MULTI-CAMERA RAIL TRACKING SYSTEM // CORE SURVEILLANCE MATRIX"
            cv2.putText(
                header,
                title_text,
                (20, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                accent_color,
                1,
                lineType=cv2.LINE_AA
            )
            
            # Global Metrics & Timestamp
            total_active_tracks = sum(detections_counts) if use_yolo else 0
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            yolo_status_str = "YOLO_ACTIVE" if use_yolo else "YOLO_BYPASS"
            sys_metrics = f"GLOBAL TRACKS: {total_active_tracks:02d}  |  {time_str}  |  {yolo_status_str}"
            cv2.putText(
                header,
                sys_metrics,
                (canvas_w - 460, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 185),
                1,
                lineType=cv2.LINE_AA
            )
            
            # 2. Concatenate individual camera frames and place onto canvas
            combined_streams = cv2.hconcat(annotated_frames)
            canvas[header_h:canvas_h, 0:canvas_w] = combined_streams
            
            # Display dashboard
            cv2.imshow("M.E.N.T.O.R. Multi-Cam Dashboard", canvas)
            
            if not paused:
                frame_count += 1
                
            # Key event handling
            # If paused, cv2.waitKey(0) blocks until a key is pressed, enabling step-by-frame
            wait_time = 0 if paused else playback_delay
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print(f" -> System state changed: {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord("y"):
                use_yolo = not use_yolo
                print(f" -> YOLO target detector: {'ENABLED' if use_yolo else 'DISABLED'}")
            elif paused and key != 0xFF:
                # Any other key advances exactly 1 frame when paused
                step_one_frame = True
                frame_count += 1
                
    finally:
        # Guaranteed resource cleanup
        print("Cleaning up resources...")
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("Dashboard stopped cleanly.")

if __name__ == "__main__":
    main()

