import time
import threading

class SafetyCoordinator:
    """
    Coordinates state transitions, calculates dynamic arrival times,
    monitors safety conditions, and generates alerts.
    Designed with a thread-safe lock for asynchronous reading/writing.
    """
    def __init__(self, distance_meters=500.0, speed_kmh=90.0):
        self.lock = threading.Lock()
        
        # Configuration parameters
        self.distance_meters = distance_meters
        self.speed_kmh = speed_kmh
        self.train_length_seconds = 8.0 # time train takes to fully pass Cam 2
        
        # State Machine: "IDLE", "TRAIN_APPROACHING", "TRAIN_ARRIVED"
        self.state = "IDLE"
        self.train_detected_time = 0.0
        self.expected_arrival_time = 0.0
        
        # Track active alert states
        self.active_alert = None # Dict containing level, message, timestamp
        self.alert_history = []
        
        # Real-time state metrics
        self.cam1_status = "MONITORING"
        self.cam2_status = "ZONE CLEAR"
        self.intrusion_active = False

    def update_config(self, distance_meters: float, speed_kmh: float):
        """Thread-safe update of physical parameters."""
        with self.lock:
            self.distance_meters = max(10.0, distance_meters)
            self.speed_kmh = max(5.0, speed_kmh)

    def get_arrival_duration(self) -> float:
        """Calculates expected time in seconds for train to arrive at Cam 2."""
        # Speed in m/s = Speed in km/h / 3.6
        speed_mps = self.speed_kmh / 3.6
        return self.distance_meters / speed_mps

    def register_train_trip(self):
        """Triggered by Camera 1 CV detector when a train is spotted."""
        with self.lock:
            if self.state == "IDLE":
                self.state = "TRAIN_APPROACHING"
                self.train_detected_time = time.time()
                self.expected_arrival_time = self.train_detected_time + self.get_arrival_duration()
                self.add_log("INFO", f"Train detected on Camera 1. Speed: {self.speed_kmh} km/h. Expected travel time: {self.get_arrival_duration():.1f}s.")

    def register_intrusion_status(self, is_intruded: bool):
        """Updates the current intrusion status from Camera 2 CV processing."""
        with self.lock:
            self.intrusion_active = is_intruded
            self.cam2_status = "BLOCKED ZONE!" if is_intruded else "ZONE CLEAR"

    def tick(self):
        """
        Periodic state-machine updates.
        Should be called in the main processing loop or server background thread.
        """
        with self.lock:
            now = time.time()
            
            # 1. State Transitions
            if self.state == "TRAIN_APPROACHING":
                countdown = self.expected_arrival_time - now
                if countdown <= 0:
                    self.state = "TRAIN_ARRIVED"
                    self.add_log("INFO", "Train has arrived at Camera 2 crossing.")
                    
            elif self.state == "TRAIN_ARRIVED":
                # Train passes for a fixed duration
                passed_time = now - self.expected_arrival_time
                if passed_time >= self.train_length_seconds:
                    self.state = "IDLE"
                    self.add_log("INFO", "Train has safely cleared Camera 2 crossing. System reset to idle.")
            
            # 2. Alert Escalation Engine
            if self.intrusion_active:
                if self.state == "TRAIN_APPROACHING":
                    countdown = max(0.0, self.expected_arrival_time - now)
                    self.active_alert = {
                        "level": "CRITICAL",
                        "message": f"CRITICAL OBSTRUCTION: Object detected in restricted zone! Train arriving in {countdown:.1f}s!",
                        "timestamp": now
                    }
                elif self.state == "TRAIN_ARRIVED":
                    self.active_alert = {
                        "level": "CRITICAL",
                        "message": "CRITICAL EMERGENCY: Active collision risk! Object in restricted zone while train is passing!",
                        "timestamp": now
                    }
                else: # IDLE
                    self.active_alert = {
                        "level": "WARNING",
                        "message": "WARNING: Unauthorized movement detected inside crossing restricted zone.",
                        "timestamp": now
                    }
            else:
                # Clear active alerts if no intrusion is present
                self.active_alert = None

    def add_log(self, level: str, message: str):
        """Logs system event logs to history."""
        log_entry = {
            "time": time.strftime("%H:%M:%S", time.localtime()),
            "level": level,
            "message": message
        }
        self.alert_history.append(log_entry)
        # Keep logs history capped at 100 entries
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)
        print(f"[{log_entry['time']}] [{level}] {message}")

    def get_telemetry(self) -> dict:
        """Returns thread-safe system telemetry for web broadcast."""
        with self.lock:
            now = time.time()
            countdown = 0.0
            if self.state == "TRAIN_APPROACHING":
                countdown = max(0.0, self.expected_arrival_time - now)
                
            return {
                "state": self.state,
                "distance": self.distance_meters,
                "speed": self.speed_kmh,
                "arrival_duration": self.get_arrival_duration(),
                "countdown": countdown,
                "intrusion_active": self.intrusion_active,
                "active_alert": self.active_alert,
                "history": self.alert_history[-15:] # Return last 15 logs
            }
