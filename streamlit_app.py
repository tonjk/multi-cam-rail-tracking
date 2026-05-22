import streamlit as st
import cv2
import numpy as np
import time
import os
import json
from src.detector import Detector

st.set_page_config(page_title="M.E.N.T.O.R. SYNC", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_detector():
    return Detector()

detector = load_detector()

def draw_premium_text(img, text, position, color, bg_color=(0, 0, 0), alpha=0.6, scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - th - 5), (x + tw + 5, y + baseline + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, lineType=cv2.LINE_AA)

def is_in_polygon(box, polygon):
    if len(polygon) < 3:
        return False
    x_center = (box[0] + box[2]) / 2.0
    y_bottom = float(box[3])
    return cv2.pointPolygonTest(polygon, (x_center, y_bottom), False) >= 0

st.sidebar.title("M.E.N.T.O.R. SYNC")
st.sidebar.markdown("---")
playing = st.sidebar.toggle("Play Stream", value=True)
st.sidebar.markdown("### Route Configuration")
delay_1_to_2 = st.sidebar.slider("Transit Time: Cam 1 to 2 (s)", 0, 60, 25)
delay_2_to_3 = st.sidebar.slider("Transit Time: Cam 2 to 3 (s)", 0, 60, 25)
st.sidebar.markdown("### Alert Settings")
alarm_enabled = st.sidebar.toggle("Alarm Enabled", value=True)
pre_arrival_window = st.sidebar.slider("Alert Window (s)", 5, 60, 10)

header_placeholder = st.empty()
st.markdown("---")
cols = st.columns(3)
cam_names = ["CAM 01 - OVERVIEW", "CAM 02 - EAST RAIL", "CAM 03 - CROSSING"]

title_cols = [cols[i].columns([3, 2]) for i in range(3)]
for i in range(3):
    title_cols[i][0].markdown(f"#### 📷 CAM {i+1}")
    title_cols[i][0].caption(cam_names[i])

duty_placeholders = [title_cols[i][1].empty() for i in range(3)]
img_placeholders = [cols[i].empty() for i in range(3)]
status_placeholders = [cols[i].empty() for i in range(3)]

fps = 30.0
stream_w, stream_h = 640, 360
header_h = 80
canvas_w = stream_w * 3

try:
    paths = [
        st.secrets["CAM01_STREAM"],
        st.secrets["CAM02_STREAM"],
        st.secrets["CAM03_STREAM"]
    ]
except Exception:
    paths = [
        "videos/long_train_1.mp4",
        "videos/long_train_2.mp4",
        "videos/long_train_3.mp4"
    ]

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.frame_count = 0
    st.session_state.start_time = time.time()
    st.session_state.cam1_last_train_time = -100
    st.session_state.cam2_arrival_time = None
    st.session_state.cam2_last_train_time = -100
    st.session_state.last_alert_states = [None, None, None]
    st.session_state.cam3_arrival_time = None
    st.session_state.cached_results = [None, None, None]
        
    roi_config_file = "roi_config.json"
    rois = []
    if os.path.exists(roi_config_file):
        try:
            with open(roi_config_file, "r") as f:
                rois = [np.array(poly, np.int32) for poly in json.load(f)]
        except Exception:
            pass
    while len(rois) < len(paths):
        rois.append(np.array([], np.int32))
    st.session_state.rois = rois
    
    st.session_state.last_rendered_frames = [np.zeros((stream_h, stream_w, 3), dtype=np.uint8)] * 3
    st.session_state.last_header = np.zeros((header_h, canvas_w, 3), dtype=np.uint8)

# Create fresh VideoCapture objects in the current Streamlit thread to prevent FFmpeg crashes
caps = []
for p in paths:
    cap = cv2.VideoCapture(p)
    if st.session_state.frame_count > 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_count % total_frames)
    caps.append(cap)

if not playing:
    # Just render the last known frames if paused
    header_placeholder.image(cv2.cvtColor(st.session_state.last_header, cv2.COLOR_BGR2RGB), width="stretch")
    for i in range(3):
        img_placeholders[i].image(cv2.cvtColor(st.session_state.last_rendered_frames[i], cv2.COLOR_BGR2RGB), width="stretch")
    st.stop()

st.session_state.start_time = time.time() - (st.session_state.frame_count / fps)

while True:
    elapsed_time = time.time() - st.session_state.start_time
    expected_frame = int(elapsed_time * fps)
    
    if expected_frame > st.session_state.frame_count + 2:
        for cap in caps:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame % total_frames)
        st.session_state.frame_count = expected_frame
        
    raw_frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((stream_h, stream_w, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (stream_w, stream_h))
        raw_frames.append(frame)
        
    virtual_time = st.session_state.frame_count / fps
    
    if st.session_state.frame_count % 3 == 0 or any(r is None for r in st.session_state.cached_results):
        st.session_state.cached_results = [detector.predict(f) for f in raw_frames]
        
    results_list = st.session_state.cached_results
    
    # Logic
    res1 = results_list[0]
    train_cam1 = False
    if res1 and res1[0].boxes:
        for cls_id in res1[0].boxes.cls:
            if int(cls_id) == 6:
                train_cam1 = True
                break
                
    if train_cam1 and (virtual_time - st.session_state.cam1_last_train_time > delay_1_to_2 + 10):
        st.session_state.cam1_last_train_time = virtual_time
        st.session_state.cam2_arrival_time = virtual_time + delay_1_to_2
        
    res2 = results_list[1]
    train_cam2 = False
    obstacles_cam2 = False
    if res2 and res2[0].boxes:
        for i, cls_id in enumerate(res2[0].boxes.cls):
            cls_id = int(cls_id)
            box = list(map(int, res2[0].boxes.xyxy[i]))
            if cls_id == 6:
                train_cam2 = True
            elif cls_id in [0, 2, 3, 5, 7]:
                if is_in_polygon(box, st.session_state.rois[1]):
                    obstacles_cam2 = True
                    
    if train_cam2 and (virtual_time - st.session_state.cam2_last_train_time > delay_2_to_3 + 10):
        st.session_state.cam2_last_train_time = virtual_time
        st.session_state.cam3_arrival_time = virtual_time + delay_2_to_3
        
    res3 = results_list[2]
    obstacles_cam3 = False
    if res3 and res3[0].boxes:
        for i, cls_id in enumerate(res3[0].boxes.cls):
            cls_id = int(cls_id)
            box = list(map(int, res3[0].boxes.xyxy[i]))
            if cls_id in [0, 2, 3, 5, 7]:
                if is_in_polygon(box, st.session_state.rois[2]):
                    obstacles_cam3 = True

    cam_statuses = [0, 0, 0]

    annotated_frames = []
    for i, (frame, res) in enumerate(zip(raw_frames, results_list)):
        ann = detector.draw_results(frame.copy(), res)
        draw_premium_text(ann, cam_names[i], (15, 25), (240, 240, 240), scale=0.5)
        
        if i == 0:
            if train_cam1:
                draw_premium_text(ann, "TRAIN DETECTED - SIGNALING CAM 02", (15, 60), (0, 255, 0), scale=0.5)
                cam_statuses[0] = 1
        elif i == 1:
            roi_color = (0, 200, 255)
            alert_active = False
            if st.session_state.cam2_arrival_time is not None:
                time_left = st.session_state.cam2_arrival_time - virtual_time
                if 0 <= time_left <= pre_arrival_window:
                    draw_premium_text(ann, f"TRAIN IN: {time_left:.1f}s", (15, 60), (0, 255, 255), scale=0.6)
                    cam_statuses[1] = 1
                    if obstacles_cam2 and alarm_enabled:
                        alert_active = True
                        cam_statuses[1] = 2
                elif time_left < 0 and time_left > -10:
                    draw_premium_text(ann, "TRAIN IN SECTOR", (15, 60), (0, 255, 0), scale=0.6)
                    cam_statuses[1] = 1

            if alert_active and (st.session_state.frame_count % 10 < 5):
                roi_color = (0, 0, 255)
                cv2.rectangle(ann, (0, 0), (stream_w-1, stream_h-1), (0, 0, 255), 4)
                draw_premium_text(ann, "CRITICAL: OBSTACLE IN RESTRICTED AREA!", (15, stream_h - 30), (0, 0, 255), bg_color=(0,0,50), scale=0.6, thickness=2)
                
            if len(st.session_state.rois[1]) >= 3:
                roi_overlay = ann.copy()
                cv2.polylines(roi_overlay, [st.session_state.rois[1]], isClosed=True, color=roi_color, thickness=2)
                cv2.fillPoly(roi_overlay, [st.session_state.rois[1]], roi_color)
                cv2.addWeighted(roi_overlay, 0.15, ann, 0.85, 0, ann)
                
        elif i == 2:
            roi_color = (0, 200, 255)
            alert_active = False
            if st.session_state.cam3_arrival_time is not None:
                time_left = st.session_state.cam3_arrival_time - virtual_time
                if 0 <= time_left <= pre_arrival_window:
                    draw_premium_text(ann, f"TRAIN IN: {time_left:.1f}s", (15, 60), (0, 255, 255), scale=0.6)
                    cam_statuses[2] = 1
                    if obstacles_cam3 and alarm_enabled:
                        alert_active = True
                        cam_statuses[2] = 2
                elif time_left < 0 and time_left > -10:
                    draw_premium_text(ann, "TRAIN IN SECTOR", (15, 60), (0, 255, 0), scale=0.6)
                    cam_statuses[2] = 1

            if alert_active and (st.session_state.frame_count % 10 < 5):
                roi_color = (0, 0, 255)
                cv2.rectangle(ann, (0, 0), (stream_w-1, stream_h-1), (0, 0, 255), 4)
                draw_premium_text(ann, "CRITICAL: OBSTACLE ON CROSSING!", (15, stream_h - 30), (0, 0, 255), bg_color=(0,0,50), scale=0.6, thickness=2)
                
            if len(st.session_state.rois[2]) >= 3:
                roi_overlay = ann.copy()
                cv2.polylines(roi_overlay, [st.session_state.rois[2]], isClosed=True, color=roi_color, thickness=2)
                cv2.fillPoly(roi_overlay, [st.session_state.rois[2]], roi_color)
                cv2.addWeighted(roi_overlay, 0.15, ann, 0.85, 0, ann)
                
        annotated_frames.append(ann)

    st.session_state.frame_count += 1
    
    # Header logic
    header = np.zeros((header_h, canvas_w, 3), dtype=np.uint8)
    header[:] = (20, 20, 22)
    cv2.line(header, (0, header_h - 1), (canvas_w, header_h - 1), (255, 200, 0), 1)
    
    title = "M.E.N.T.O.R. SYNC // EARLY WARNING TRACKING DEMO"
    draw_premium_text(header, title, (20, 25), (255, 200, 0), scale=0.7, bg_color=(20, 20, 22))
    
    status_text = f"ALARM SYSTEM: {'ACTIVE' if alarm_enabled else 'MUTED'}  |  STATE: LIVE  |  V-TIME: {virtual_time:.1f}s"
    status_color = (0, 255, 0) if alarm_enabled else (100, 100, 100)
    draw_premium_text(header, status_text, (canvas_w - 600, 25), status_color, scale=0.55, bg_color=(20, 20, 22))

    track_y = header_h - 20
    cv2.line(header, (50, track_y), (canvas_w - 50, track_y), (100, 100, 100), 2)
    
    node_x = [int(stream_w / 2), int(stream_w + stream_w / 2), int(2 * stream_w + stream_w / 2)]
    node_names = ["CAM 01", "CAM 02", "CAM 03"]
    
    for i, nx in enumerate(node_x):
        cv2.circle(header, (nx, track_y), 6, (255, 200, 0), -1)
        draw_premium_text(header, node_names[i], (nx - 25, track_y - 12), (200, 200, 200), scale=0.4, bg_color=(20, 20, 22))
        
    train_x = None
    if st.session_state.cam2_arrival_time is not None and virtual_time >= st.session_state.cam1_last_train_time and virtual_time <= st.session_state.cam2_arrival_time:
        transit_duration = st.session_state.cam2_arrival_time - st.session_state.cam1_last_train_time
        if transit_duration > 0:
            progress = (virtual_time - st.session_state.cam1_last_train_time) / transit_duration
            progress = max(0.0, min(1.0, progress))
            train_x = int(node_x[0] + progress * (node_x[1] - node_x[0]))
    elif st.session_state.cam3_arrival_time is not None and virtual_time >= st.session_state.cam2_last_train_time and virtual_time <= st.session_state.cam3_arrival_time:
        transit_duration = st.session_state.cam3_arrival_time - st.session_state.cam2_last_train_time
        if transit_duration > 0:
            progress = (virtual_time - st.session_state.cam2_last_train_time) / transit_duration
            progress = max(0.0, min(1.0, progress))
            train_x = int(node_x[1] + progress * (node_x[2] - node_x[1]))
            
    if train_x is not None:
        cv2.rectangle(header, (train_x - 20, track_y - 8), (train_x + 20, track_y + 8), (0, 255, 0), -1)
        cv2.putText(header, "TRAIN", (train_x - 18, track_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Save to session state
    st.session_state.last_header = header
    st.session_state.last_rendered_frames = annotated_frames
    
    # Render Streamlit Alerts
    for i in range(3):
        if cam_statuses[i] != st.session_state.last_alert_states[i]:
            if cam_statuses[i] == 2:
                status_placeholders[i].error(f"🚨 **ALARM TRIGGERED**")
                duty_placeholders[i].markdown("<div style='text-align: right; padding-top: 15px;'><span style='color: #00ff00; font-weight: bold;'>🟢 DETECTING</span></div>", unsafe_allow_html=True)
            elif cam_statuses[i] == 1:
                status_placeholders[i].empty()
                duty_placeholders[i].markdown("<div style='text-align: right; padding-top: 15px;'><span style='color: #00ff00; font-weight: bold;'>🟢 DETECTING</span></div>", unsafe_allow_html=True)
            else:
                status_placeholders[i].empty()
                duty_placeholders[i].empty()
            st.session_state.last_alert_states[i] = cam_statuses[i]

    # Render to Streamlit UI (Optimized for Network/Ngrok Tunneling)
    if st.session_state.frame_count % 2 == 0:
        _, header_buffer = cv2.imencode('.jpg', header, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        header_placeholder.image(header_buffer.tobytes(), output_format='JPEG', width="stretch")
        
        for i in range(3):
            _, frame_buffer = cv2.imencode('.jpg', annotated_frames[i], [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            img_placeholders[i].image(frame_buffer.tobytes(), output_format='JPEG', width="stretch")
        
    time.sleep(0.01) # Small sleep to prevent CPU hogging
