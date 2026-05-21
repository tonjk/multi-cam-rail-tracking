import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from src.detector import Detector

def no_op(x):
    pass

def draw_premium_text(img, text, position, color, bg_color=(0, 0, 0), alpha=0.6, scale=0.6, thickness=1):
    """Draws text with a semi-transparent background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    
    # Draw background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

def draw_polygon_interactive(window_name, frame):
    """
    Shows the frame and lets the user click points to draw a polygon.
    Press ENTER to finish drawing.
    """
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = frame.copy()
        
        if len(points) > 0:
            for p in points:
                cv2.circle(display, p, 4, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        if len(points) > 2:
            overlay = display.copy()
            cv2.fillPoly(overlay, [np.array(points)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
            
        cv2.putText(display, "Click to define ROI points.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(display, "Press ENTER when done.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord(' '):  # Enter or Space
            break
            
    cv2.destroyWindow(window_name)
    return np.array(points, np.int32)

def is_in_polygon(box, polygon):
    """Checks if the bottom-center of the bounding box is inside the polygon."""
    if len(polygon) < 3:
        return False
    x_center = (box[0] + box[2]) / 2.0
    y_bottom = float(box[3])
    return cv2.pointPolygonTest(polygon, (x_center, y_bottom), False) >= 0

def main():
    print("Initializing M.E.N.T.O.R. Demo Application...")
    
    # Setup Window and Controls
    window_name = "M.E.N.T.O.R. Multi-Cam Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    ctrl_win = "Controls"
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ctrl_win, 400, 200)
    
    cv2.createTrackbar("Cam1->2", ctrl_win, 10, 30, no_op)
    cv2.createTrackbar("Cam2->3", ctrl_win, 10, 30, no_op)
    cv2.createTrackbar("Alarm", ctrl_win, 1, 1, no_op)
    cv2.createTrackbar("Alert", ctrl_win, 15, 60, no_op)
    
    # Initialize Detector
    detector = Detector()
    
    # Video Sources
    paths = [
        "videos/long_train_1.mp4", 
        "videos/long_train_2.mp4", 
        "videos/long_train_3.mp4"
    ]
    caps = []
    for p in paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"Error opening {p}")
            return
        caps.append(cap)
        
    cam_names = ["CAM 01 // OVERVIEW", "CAM 02 // EAST RAIL", "CAM 03 // CROSSING"]
    
    # Dimensions
    stream_w, stream_h = 640, 360
    header_h = 80
    canvas_w = stream_w * len(paths)
    canvas_h = stream_h + header_h
    fps = 30.0  # Standard virtual framerate
    
    # ROI Definition with Config File
    roi_config_file = "roi_config.json"
    rois = []
    if os.path.exists(roi_config_file):
        print(f"Loading restricted areas from {roi_config_file}...")
        try:
            with open(roi_config_file, "r") as f:
                loaded_rois = json.load(f)
            rois = [np.array(poly, np.int32) for poly in loaded_rois]
        except Exception as e:
            print(f"Error loading {roi_config_file}: {e}")
            rois = []
            
    if not rois or len(rois) != len(caps):
        print("Please define the restricted area (ROI) for each camera.")
        rois = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (stream_w, stream_h))
                poly = draw_polygon_interactive(f"Define ROI for {cam_names[i]}", frame_resized)
                if len(poly) < 3:
                    poly = np.array([], np.int32)
                rois.append(poly)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                rois.append(np.array([], np.int32))
        try:
            with open(roi_config_file, "w") as f:
                json.dump([poly.tolist() for poly in rois], f)
            print(f"Saved ROI configuration to {roi_config_file}")
        except Exception as e:
            print(f"Error saving {roi_config_file}: {e}")
    
    # State tracking
    frame_count = 0
    paused = False
    start_time = time.time()
    
    # Virtual timers
    cam1_last_train_time = -100
    cam2_arrival_time = None
    
    cam2_last_train_time = -100
    cam3_arrival_time = None
    
    # Constants
    # PRE_ARRIVAL_WINDOW is now controlled via trackbar
    
    print("Application Ready. Controls: [Space] Pause/Play, [Q] Quit.")
    
    last_frames = [None, None, None]
    cached_results = [None, None, None]  # Cache for YOLO results to speed up FPS
    
    while True:
        # Determine delay from trackbars
        delay_1_to_2 = cv2.getTrackbarPos("Cam1->2", ctrl_win)
        delay_2_to_3 = cv2.getTrackbarPos("Cam2->3", ctrl_win)
        alarm_enabled = cv2.getTrackbarPos("Alarm", ctrl_win) == 1
        pre_arrival_window = cv2.getTrackbarPos("Alert", ctrl_win)
        
        if not paused:
            # Sync with real time to avoid slow-motion playback
            elapsed_time = time.time() - start_time
            expected_frame = int(elapsed_time * fps)
            
            # If we are significantly behind (e.g., > 2 frames), skip frames to catch up
            if expected_frame > frame_count + 2:
                for cap in caps:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame % total_frames)
                frame_count = expected_frame
                
            raw_frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((stream_h, stream_w, 3), dtype=np.uint8)
                frame = cv2.resize(frame, (stream_w, stream_h))
                raw_frames.append(frame)
            last_frames = raw_frames
            
            # Use virtual time based on frame count (which is now synced to real time)
            virtual_time = frame_count / fps
            
            # --- Detection & Logic ---
            annotated_frames = []
            
            # Throttle YOLO inference: run only every 3 frames (~10 fps) to drastically improve performance
            if frame_count % 3 == 0 or any(r is None for r in cached_results):
                cached_results = [detector.predict(f) for f in raw_frames]
            
            results_list = cached_results
            
            # Analyze Cam 1
            res1 = results_list[0]
            train_cam1 = False
            if res1 and res1[0].boxes:
                for cls_id in res1[0].boxes.cls:
                    if int(cls_id) == 6: # train
                        train_cam1 = True
                        break
            
            # Trigger arrival to Cam 2
            if train_cam1 and (virtual_time - cam1_last_train_time > delay_1_to_2 + 10):
                cam1_last_train_time = virtual_time
                cam2_arrival_time = virtual_time + delay_1_to_2
                
            # Analyze Cam 2
            res2 = results_list[1]
            train_cam2 = False
            obstacles_cam2 = False
            if res2 and res2[0].boxes:
                for i, cls_id in enumerate(res2[0].boxes.cls):
                    cls_id = int(cls_id)
                    box = list(map(int, res2[0].boxes.xyxy[i]))
                    if cls_id == 6: # found train
                        train_cam2 = True
                    elif cls_id in [0, 2, 3, 5, 7]: # person, car, motorcycle, bus, truck
                        if is_in_polygon(box, rois[1]): # found obstacle in restricted area
                            obstacles_cam2 = True
                            
            # Trigger arrival to Cam 3
            if train_cam2 and (virtual_time - cam2_last_train_time > delay_2_to_3 + 10):
                cam2_last_train_time = virtual_time
                cam3_arrival_time = virtual_time + delay_2_to_3
                
            # Analyze Cam 3
            res3 = results_list[2]
            obstacles_cam3 = False
            if res3 and res3[0].boxes:
                for i, cls_id in enumerate(res3[0].boxes.cls):
                    cls_id = int(cls_id)
                    box = list(map(int, res3[0].boxes.xyxy[i]))
                    if cls_id in [0, 2, 3, 5, 7]:
                        if is_in_polygon(box, rois[2]):
                            obstacles_cam3 = True
            
            # --- Rendering ---
            for i, (frame, res) in enumerate(zip(raw_frames, results_list)):
                ann = detector.draw_results(frame.copy(), res)
                
                # Default overlay info
                draw_premium_text(ann, cam_names[i], (15, 25), (240, 240, 240), scale=0.5)
                
                # Logic specific rendering
                if i == 0:
                    if train_cam1:
                        draw_premium_text(ann, "TRAIN DETECTED - SIGNALING CAM 02", (15, 60), (0, 255, 0), scale=0.5)
                        
                elif i == 1:
                    # Draw ROI
                    roi_color = (0, 200, 255) # Default yellow/amber
                    alert_active = False
                    
                    if cam2_arrival_time is not None:
                        time_left = cam2_arrival_time - virtual_time
                        if 0 <= time_left <= pre_arrival_window:
                            # Countdown active
                            draw_premium_text(ann, f"TRAIN IN: {time_left:.1f}s", (15, 60), (0, 255, 255), scale=0.6)
                            if obstacles_cam2 and alarm_enabled:
                                alert_active = True
                        elif time_left < 0 and time_left > -10:
                            draw_premium_text(ann, "TRAIN IN SECTOR", (15, 60), (0, 255, 0), scale=0.6)
                    
                    # Flash red if alert
                    if alert_active and (frame_count % 10 < 5):
                        roi_color = (0, 0, 255)
                        cv2.rectangle(ann, (0, 0), (stream_w-1, stream_h-1), (0, 0, 255), 4)
                        draw_premium_text(ann, "CRITICAL: OBSTACLE IN RESTRICTED AREA!", (15, stream_h - 30), (0, 0, 255), bg_color=(0,0,50), scale=0.6, thickness=2)
                        
                    # Draw semi-transparent Polygon ROI
                    if len(rois[1]) >= 3:
                        roi_overlay = ann.copy()
                        cv2.polylines(roi_overlay, [rois[1]], isClosed=True, color=roi_color, thickness=2)
                        cv2.fillPoly(roi_overlay, [rois[1]], roi_color)
                        cv2.addWeighted(roi_overlay, 0.15, ann, 0.85, 0, ann)
                    
                elif i == 2:
                    # Draw ROI
                    roi_color = (0, 200, 255)
                    alert_active = False
                    
                    if cam3_arrival_time is not None:
                        time_left = cam3_arrival_time - virtual_time
                        if 0 <= time_left <= pre_arrival_window:
                            draw_premium_text(ann, f"TRAIN IN: {time_left:.1f}s", (15, 60), (0, 255, 255), scale=0.6)
                            if obstacles_cam3 and alarm_enabled:
                                alert_active = True
                        elif time_left < 0 and time_left > -10:
                            draw_premium_text(ann, "TRAIN IN SECTOR", (15, 60), (0, 255, 0), scale=0.6)
                            
                    if alert_active and (frame_count % 10 < 5):
                        roi_color = (0, 0, 255)
                        cv2.rectangle(ann, (0, 0), (stream_w-1, stream_h-1), (0, 0, 255), 4)
                        draw_premium_text(ann, "CRITICAL: OBSTACLE ON CROSSING!", (15, stream_h - 30), (0, 0, 255), bg_color=(0,0,50), scale=0.6, thickness=2)
                        
                    # Draw semi-transparent Polygon ROI
                    if len(rois[2]) >= 3:
                        roi_overlay = ann.copy()
                        cv2.polylines(roi_overlay, [rois[2]], isClosed=True, color=roi_color, thickness=2)
                        cv2.fillPoly(roi_overlay, [rois[2]], roi_color)
                        cv2.addWeighted(roi_overlay, 0.15, ann, 0.85, 0, ann)
                
                annotated_frames.append(ann)
                
            frame_count += 1
            
        else:
            # Paused logic: keep rendering the last frames but update UI
            annotated_frames = last_frames
            pass # Keep simple for now during pause
            
        # Assemble Dashboard
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Header
        header = canvas[0:header_h, 0:canvas_w]
        header[:] = (20, 20, 22)
        cv2.line(header, (0, header_h - 1), (canvas_w, header_h - 1), (255, 200, 0), 1)
        
        title = "M.E.N.T.O.R. SYNC // EARLY WARNING TRACKING DEMO"
        draw_premium_text(header, title, (20, 25), (255, 200, 0), scale=0.7, bg_color=(20, 20, 22))
        
        status_text = f"ALARM SYSTEM: {'ACTIVE' if alarm_enabled else 'MUTED'}  |  STATE: {'PAUSED' if paused else 'LIVE'}  |  V-TIME: {virtual_time:.1f}s"
        status_color = (0, 255, 0) if alarm_enabled else (100, 100, 100)
        draw_premium_text(header, status_text, (canvas_w - 600, 25), status_color, scale=0.55, bg_color=(20, 20, 22))

        # --- Route Animation ---
        track_y = header_h - 20
        # Draw base track line
        cv2.line(header, (50, track_y), (canvas_w - 50, track_y), (100, 100, 100), 2)
        
        node_x = [int(stream_w / 2), int(stream_w + stream_w / 2), int(2 * stream_w + stream_w / 2)]
        node_names = ["CAM 01", "CAM 02", "CAM 03"]
        
        for i, nx in enumerate(node_x):
            cv2.circle(header, (nx, track_y), 6, (255, 200, 0), -1)
            draw_premium_text(header, node_names[i], (nx - 25, track_y - 12), (200, 200, 200), scale=0.4, bg_color=(20, 20, 22))
            
        # Draw moving train if in transit
        train_x = None
        if cam2_arrival_time is not None and virtual_time >= cam1_last_train_time and virtual_time <= cam2_arrival_time:
            transit_duration = cam2_arrival_time - cam1_last_train_time
            if transit_duration > 0:
                progress = (virtual_time - cam1_last_train_time) / transit_duration
                progress = max(0.0, min(1.0, progress))
                train_x = int(node_x[0] + progress * (node_x[1] - node_x[0]))
        elif cam3_arrival_time is not None and virtual_time >= cam2_last_train_time and virtual_time <= cam3_arrival_time:
            transit_duration = cam3_arrival_time - cam2_last_train_time
            if transit_duration > 0:
                progress = (virtual_time - cam2_last_train_time) / transit_duration
                progress = max(0.0, min(1.0, progress))
                train_x = int(node_x[1] + progress * (node_x[2] - node_x[1]))
                
        if train_x is not None:
            # Draw train symbol
            cv2.rectangle(header, (train_x - 20, track_y - 8), (train_x + 20, track_y + 8), (0, 255, 0), -1)
            cv2.putText(header, "TRAIN", (train_x - 18, track_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Stream combined
        if len(annotated_frames) == 3:
            combined = cv2.hconcat(annotated_frames)
            canvas[header_h:canvas_h, 0:canvas_w] = combined
        
        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            # When resuming, reset start time to align with current frame
            if not paused:
                start_time = time.time() - (frame_count / fps)
                
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
