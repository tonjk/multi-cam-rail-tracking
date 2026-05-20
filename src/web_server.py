import asyncio
import cv2
import threading
import time
import numpy as np
from typing import Set
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ultralytics import YOLO

# Import our custom components
from video_generator import generate_sample_videos
from video_reader import LoopingVideoCap
from cv_processor import TrainDetector, IntrusionDetector
from safety_coordinator import SafetyCoordinator

app = FastAPI(title="M.E.N.T.O.R. YOLO26n Rail Tracking System")

# Templates location
templates = Jinja2Templates(directory="src/templates")

# Define physical danger zone coordinates (Cam 2 crossing)
restricted_zone_polygon = np.array([
    [400, 570],  # Bottom Left
    [560, 300],  # Top Left
    [880, 300],  # Top Right
    [1040, 570]  # Bottom Right
], dtype=np.int32)

# Global variables (initialized in lifespan)
shared_model = None
train_detector = None
intrusion_detector = None
coordinator = None

cap1 = None
cap2 = None

# Thread-safe image buffers for MJPEG streaming
frame1_buffer = None
frame2_buffer = None
buffer_lock = threading.Lock()

# Background thread control
running = True
bg_thread = None

# Active WebSocket connections
active_connections: Set[WebSocket] = set()

# Keep track of simulated button toggle state
intruder_active = False

# Configuration update model
class ConfigUpdate(BaseModel):
    distance: float
    speed: float


def background_cv_loop():
    """
    Background processing loop that grabs frames from OpenCV video files,
    runs actual real-time multi-class YOLO26n object detection,
    verifies boundary crossing safety conditions, and updates telemetry.
    """
    global frame1_buffer, frame2_buffer, running
    
    print("[SYSTEM] Starting YOLO26n background Computer Vision processing loop...")
    coordinator.add_log("INFO", "YOLO26n model and real video pipelines initialized.")
    
    frame_duration = 1.0 / 30.0 # Match 30 FPS playback rate
    
    while running:
        start_time = time.time()
        
        try:
            # 1. Grab frames from looping video reader feeds
            grabbed1, frame1 = cap1.read()
            grabbed2, frame2 = cap2.read()
            
            # 2. Process Cam 1 Approach Feed
            if grabbed1 and frame1 is not None:
                train_triggered, annotated_frame1 = train_detector.process_frame(frame1)
                
                # If YOLO detects train crossing the sensor line, trigger Safety Coordinator
                if train_triggered:
                    coordinator.register_train_trip()
                    
                # Cache annotated JPEG
                _, jpeg1 = cv2.imencode('.jpg', annotated_frame1, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with buffer_lock:
                    frame1_buffer = jpeg1.tobytes()
            
            # 3. Get current train proximity status from safety engine
            train_approaching = (coordinator.state != "IDLE")
            
            # 4. Process Cam 2 Restricted Zone Feed
            if grabbed2 and frame2 is not None:
                # Run YOLO boundary crossing detection
                is_intruded, annotated_frame2 = intrusion_detector.process_frame(frame2, train_approaching, enabled=intruder_active)
                
                # Sync safety coordinator
                coordinator.register_intrusion_status(is_intruded)
                coordinator.tick()
                
                # Cache annotated JPEG
                _, jpeg2 = cv2.imencode('.jpg', annotated_frame2, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with buffer_lock:
                    frame2_buffer = jpeg2.tobytes()
            else:
                coordinator.tick()
                
        except Exception as e:
            print(f"[ERROR] Exception in Background CV loop: {e}")
            
        # Maintain constant processing framerate
        elapsed = time.time() - start_time
        sleep_time = max(0.001, frame_duration - elapsed)
        time.sleep(sleep_time)

    print("[SYSTEM] Background CV processing loop shut down safely.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for startup/shutdown synchronization."""
    global shared_model, train_detector, intrusion_detector, coordinator
    global cap1, cap2, bg_thread, running
    
    # --- Startup Logic ---
    print("[SYSTEM] Starting M.E.N.T.O.R. Safety Engine...")
    
    
    # 2. Instantiate YOLO26n (Nano architecture, optimized for CPU real-time edge processing)
    print("[SYSTEM] Loading Ultralytics YOLO26n neural network model...")
    shared_model = YOLO("yolo26n.pt")
    
    # 3. Instantiate safety and CV processing systems
    train_detector = TrainDetector(model=shared_model, sensor_line_y=450)
    intrusion_detector = IntrusionDetector(zone_polygon=restricted_zone_polygon, model=shared_model)
    coordinator = SafetyCoordinator()
    
    # 4. Load Loop Video Captures
    cap1 = LoopingVideoCap("src/videos/Thai_train_passing.mp4", name="ApproachFeed")
    cap2 = LoopingVideoCap("src/videos/Cars_driving_railway_crossing.mp4", name="CrossingFeed")
    
    # 5. Start loop reader threads
    cap1.start()
    cap2.start()
    
    # 6. Spawn Background CV loop thread
    running = True
    bg_thread = threading.Thread(target=background_cv_loop, daemon=True)
    bg_thread.start()
    
    yield
    
    # --- Shutdown Logic ---
    print("[SYSTEM] Initiating safe shutdown sequence...")
    running = False
    
    if cap1:
        cap1.stop()
    if cap2:
        cap2.stop()
        
    if bg_thread:
        bg_thread.join(timeout=2.0)
    print("[SYSTEM] Clean shutdown completed.")


app = FastAPI(title="M.E.N.T.O.R. YOLO26n Rail Tracking System", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serves the main web dashboard dashboard interface."""
    return templates.TemplateResponse(request=request, name="index.html")


def generate_video_stream(camera_id: int):
    """Generates an MJPEG multipart response stream for browsers."""
    global frame1_buffer, frame2_buffer
    
    while running:
        frame = None
        with buffer_lock:
            if camera_id == 1:
                frame = frame1_buffer
            elif camera_id == 2:
                frame = frame2_buffer
                
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033) # rate limit feed output


@app.get("/video_feed_1")
async def video_feed_1():
    """Streaming endpoint for Camera 1 Approach view."""
    return StreamingResponse(generate_video_stream(1), 
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed_2")
async def video_feed_2():
    """Streaming endpoint for Camera 2 Crossing restricted zone."""
    return StreamingResponse(generate_video_stream(2), 
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/trigger_train")
async def trigger_train():
    """Interactive trigger: seeks Camera 1 video playhead to the train event."""
    if cap1:
        # Seek to frame 0 (train arrives immediately)
        cap1.seek_to_frame(0)
        coordinator.add_log("SIMULATOR", "Interactive Train Approach event injected into stream.")
    return {"status": "success", "train_active": True}


@app.post("/toggle_intruder")
async def toggle_intruder():
    """Interactive trigger: toggles test state for crossing trespasser detection."""
    global intruder_active
    intruder_active = not intruder_active
    if intruder_active:
        coordinator.add_log("SIMULATOR", "Interactive Obstruction checking enabled.")
    else:
        coordinator.add_log("SIMULATOR", "Interactive Obstruction checking bypassed (Safe Mode).")
            
    return {"status": "success", "intruder_active": intruder_active}


@app.post("/update_config")
async def update_config(config: ConfigUpdate):
    """Updates physical inter-camera parameters on-the-fly."""
    coordinator.update_config(config.distance, config.speed)
    return {"status": "success", "distance": coordinator.distance_meters, "speed": coordinator.speed_kmh}


# --- WebSocket Broadcast Manager ---

async def websocket_broadcaster():
    """
    Periodically fetches telemetry data from SafetyCoordinator
    and broadcasts it to all active clients over WebSockets at 15 Hz.
    """
    while running:
        if active_connections:
            telemetry = coordinator.get_telemetry()
            for ws in list(active_connections):
                try:
                    await ws.send_json(telemetry)
                except Exception:
                    if ws in active_connections:
                        active_connections.remove(ws)
        await asyncio.sleep(0.066)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Establishes real-time connection with client dashboard."""
    await websocket.accept()
    active_connections.add(websocket)
    
    if len(active_connections) == 1:
        asyncio.create_task(websocket_broadcaster())
        
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    # Start uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
