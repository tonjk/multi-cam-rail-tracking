import cv2
import threading
import time

class LoopingVideoCap:
    """
    Non-blocking, thread-safe video reader that plays an MP4 file
    on loop continuously, maintaining the original video FPS.
    Complies with high-performance OpenCV guidelines.
    """
    def __init__(self, filepath, name="Stream"):
        self.filepath = filepath
        self.name = name
        self.cap = cv2.VideoCapture(filepath)
        self.frame = None
        self.grabbed = False
        self.started = False
        self.lock = threading.Lock()
        self.thread = None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Get dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(
            target=self._update, name=f"ReadThread-{self.name}", daemon=True
        )
        self.thread.start()
        return self

    def _update(self):
        frame_delay = 1.0 / self.fps
        while self.started:
            start_time = time.time()
            if not self.cap.isOpened():
                self._reconnect()
                continue

            grabbed, frame = self.cap.read()

            if not grabbed:
                # Video has reached end, loop back to start frame 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                grabbed, frame = self.cap.read()
                if not grabbed:
                    print(f"[{self.name}] Failed to read frame even after looping. Reconnecting...")
                    self._reconnect()
                    continue

            with self.lock:
                self.frame = frame
                self.grabbed = True
                
            # Rate limit reading to match original video FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)

    def _reconnect(self):
        self.cap.release()
        with self.lock:
            self.grabbed = False
        time.sleep(1.5)
        self.cap = cv2.VideoCapture(self.filepath)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def seek_to_frame(self, frame_number):
        """Seeks the video capture to a specific frame number immediately."""
        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                grabbed, frame = self.cap.read()
                if grabbed:
                    self.frame = frame
                    self.grabbed = True

    def stop(self):
        self.started = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
