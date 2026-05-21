# M.E.N.T.O.R. - Multi-Camera Intelligent Monitoring & Alert System

A real-time, multi-camera system designed to monitor train movements and critical zones using AI.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Environment
Create a `.streamlit/secrets.toml` file or set environment variables for your camera streams:

```toml
# .streamlit/secrets.toml
CAM01_STREAM=rtsp://[IP_ADDRESS]/cam/realmonitor?channel=1&subtype=0
CAM02_STREAM=rtsp://[IP_ADDRESS]/cam/realmonitor?channel=1&subtype=0
CAM03_STREAM=rtsp://[IP_ADDRESS]/cam/realmonitor?channel=1&subtype=0

# Optional: High-quality H.264 streams
# CAM01_STREAM=rtsp://[IP_ADDRESS]/cam/realmonitor?channel=1&subtype=1
# CAM02_STREAM=rtsp://[IP_ADDRESS]/cam/realmonitor?channel=1&subtype=1
# CAM03_STREAM=rtsp://[IP_ADDRESS]/cam/realmonitor?channel=1&subtype=1
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

## 🔧 Usage

### Adjusting Delays
- **Transit Time**: Set the estimated time it takes for a train to travel between cameras.
- **Alert Window**: Defines how early you want to be notified before the train reaches the next camera.

### Defining Restricted Areas (ROI)
1. Run the app.
2. Click the **"Define ROI"** button for each camera.
3. Click on the camera view to mark points around the restricted area.
4. Press **Enter** to save.

## 📁 File Structure

- `streamlit_app.py`: The main Streamlit application interface.
- `src/detector.py`: Core logic for object detection using YOLO.
- `utils/camera_config.py`: Manages camera connection and stream health.
- `data/class_list.txt`: YOLO class labels.
- `.streamlit/secrets.toml`: Stores sensitive camera stream URLs.
