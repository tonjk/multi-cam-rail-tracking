import cv2
import numpy as np
import time
import random

class CameraSimulator:
    """
    Simulates physical CCTV cameras for rail monitoring.
    Generates synthetic frames for Camera 1 (Approach) and Camera 2 (Crossing/Danger Zone)
    with realistic rail rendering, perspective, and interactive elements.
    """
    def __init__(self, fps=30):
        self.fps = fps
        self.width = 640
        self.height = 480
        
        # Camera 1 (Train Approach) State
        self.train_active = False
        self.train_position = 0.0  # 0.0 (horizon) to 1.0 (exited screen)
        self.train_speed = 0.02   # normalized speed per frame
        self.train_detected_callback = None
        self.train_detector_tripped = False
        
        # Camera 2 (Crossing/Restricted Zone) State
        self.intruder_active = False
        self.intruder_pos = [320, 240]  # x, y in screen coordinates
        self.intruder_target = [320, 240]
        self.intruder_type = "pedestrian" # "pedestrian" or "vehicle"
        
        # Danger zone polygon in Camera 2 (Bird's eye skew / perspective)
        # Quad defining the intersection of road and rails
        self.restricted_zone_polygon = np.array([
            [200, 380],  # Bottom Left
            [280, 200],  # Top Left
            [440, 200],  # Top Right
            [520, 380]   # Bottom Right
        ], dtype=np.int32)
        
        # Static background elements for consistency
        self._init_backgrounds()

    def _init_backgrounds(self):
        # Create static base landscapes
        # Cam 1 base (Approaching track in rural/industrial area)
        self.cam1_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Grass background
        self.cam1_bg[:, :] = [25, 45, 20]  # dark green HSL-ish
        # Ballast (gravel roadbed)
        ballast_poly = np.array([[220, 0], [420, 0], [550, 480], [90, 480]], dtype=np.int32)
        cv2.fillPoly(self.cam1_bg, [ballast_poly], [50, 50, 50])  # dark gray ballast
        # Draw sleepers (ties) in perspective
        for i in range(15):
            y = int(480 * (i / 14)**2)  # Quadratic spacing for perspective
            w = int(60 + 260 * (i / 14))
            x1 = 320 - w // 2
            x2 = 320 + w // 2
            h = max(2, int(15 * (i / 14)))
            cv2.rectangle(self.cam1_bg, (x1, y), (x2, y + h), [35, 55, 75], -1)  # Wood sleepers
            cv2.rectangle(self.cam1_bg, (x1, y), (x2, y + h), [20, 30, 40], 1)
        # Steel Rails
        # Convergence points: Horizon (320, 0)
        # Left Rail: (240, 0) to (140, 480)
        # Right Rail: (400, 0) to (500, 480)
        cv2.line(self.cam1_bg, (320 - 40, 0), (320 - 150, 480), [160, 160, 160], 3, cv2.LINE_AA)
        cv2.line(self.cam1_bg, (320 + 40, 0), (320 + 150, 480), [160, 160, 160], 3, cv2.LINE_AA)

        # Cam 2 base (Crossing track)
        self.cam2_bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.cam2_bg[:, :] = [30, 40, 30]  # dark gray-green
        # Railway running diagonally from top-left to bottom-right
        ballast_poly_2 = np.array([[100, 0], [250, 0], [580, 480], [430, 480]], dtype=np.int32)
        cv2.fillPoly(self.cam2_bg, [ballast_poly_2], [45, 45, 45])
        # Crossing road running from bottom-left to top-right
        road_poly = np.array([[0, 380], [0, 300], [640, 180], [640, 260]], dtype=np.int32)
        cv2.fillPoly(self.cam2_bg, [road_poly], [60, 65, 70])  # asphalt road
        # Road yellow markings
        cv2.line(self.cam2_bg, (0, 340), (640, 220), [20, 180, 220], 2, cv2.LINE_AA) # Double yellow (BGR)
        cv2.line(self.cam2_bg, (0, 344), (640, 224), [20, 180, 220], 2, cv2.LINE_AA)
        # Rails diagonal
        cv2.line(self.cam2_bg, (150, 0), (480, 480), [180, 180, 180], 4, cv2.LINE_AA)
        cv2.line(self.cam2_bg, (200, 0), (530, 480), [180, 180, 180], 4, cv2.LINE_AA)

    def trigger_train(self, speed_mps=25):
        """Spawns a train on Camera 1."""
        if not self.train_active:
            self.train_active = True
            self.train_position = 0.0
            self.train_detector_tripped = False
            # Map speed in meters/second to normalized step rate
            # Let's say the simulated field of view is 150 meters long.
            # Train needs to travel 150m. Total time = 150 / speed_mps
            # Total frames = time * FPS
            # Step per frame = 1 / total_frames
            fov_length_meters = 150.0
            travel_time = fov_length_meters / speed_mps
            total_frames = max(30, travel_time * self.fps)
            self.train_speed = 1.0 / total_frames

    def toggle_intruder(self, state=None):
        """Spawns or despawns an intruder on Camera 2."""
        if state is not None:
            self.intruder_active = state
        else:
            self.intruder_active = not self.intruder_active
            
        if self.intruder_active:
            # Place intruder near danger zone boundary
            self.intruder_pos = [320 + random.randint(-50, 50), 300 + random.randint(-20, 20)]
            self.intruder_target = self.intruder_pos.copy()
            self.intruder_type = random.choice(["pedestrian", "vehicle"])
        
    def set_train_detected_callback(self, callback):
        self.train_detected_callback = callback

    def update(self):
        """Advances physics/animation state by one frame."""
        # Update train position
        if self.train_active:
            self.train_position += self.train_speed
            # Check detection point on Cam 1 (say at 65% down the screen, y=312)
            if self.train_position >= 0.55 and not self.train_detector_tripped:
                self.train_detector_tripped = True
                if self.train_detected_callback:
                    self.train_detected_callback()
            
            if self.train_position >= 1.2:  # Train has fully exited
                self.train_active = False
                self.train_position = 0.0
                self.train_detector_tripped = False

        # Update intruder motion (slight wandering/pacing inside/around restricted zone to trigger motion detector)
        if self.intruder_active:
            # Walk towards a random point inside danger zone
            dx = self.intruder_target[0] - self.intruder_pos[0]
            dy = self.intruder_target[1] - self.intruder_pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 5:
                # Pick a new nearby target inside or very close to the restricted zone
                # Polygon bounding box roughly: X [200, 520], Y [200, 380]
                self.intruder_target = [
                    random.randint(240, 480),
                    random.randint(220, 360)
                ]
            else:
                # Step towards target
                self.intruder_pos[0] += int(np.clip(dx / dist * 3, -3, 3))
                self.intruder_pos[1] += int(np.clip(dy / dist * 3, -3, 3))

    def render_cam1(self) -> np.ndarray:
        """Renders Camera 1 (Train Approach) frame."""
        frame = self.cam1_bg.copy()
        
        # Render train if active
        if self.train_active:
            pos = self.train_position
            # Perspective math: train width and height scale with position
            w = int(20 + 200 * pos)
            h = int(10 + 120 * pos)
            
            # Train center tracks convergence line (320, 0) -> (320, 480)
            center_x = 320
            # Track quadratic descent for acceleration feel
            center_y = int(480 * pos**2)
            
            x1 = center_x - w // 2
            y1 = center_y - h
            x2 = center_x + w // 2
            y2 = center_y
            
            if y2 > 0 and y1 < 480:
                # Draw shadows
                cv2.rectangle(frame, (max(0, x1 - 10), max(0, y1 + h - 5)), (min(640, x2 + 10), min(480, y2 + 15)), [10, 15, 10], -1)
                
                # Draw train body (sleek silver-red modern bullet train)
                cv2.rectangle(frame, (x1, max(0, y1)), (x2, min(480, y2)), [70, 70, 90], -1)  # Silver body
                # Red nose stripe
                nose_h = max(2, h // 4)
                cv2.rectangle(frame, (x1, min(480, y2 - nose_h)), (x2, min(480, y2)), [40, 40, 200], -1) # Red accent (BGR)
                
                # Front windshield
                shield_w = int(w * 0.7)
                shield_h = max(2, h // 5)
                shield_x1 = center_x - shield_w // 2
                shield_y1 = y2 - nose_h - shield_h - max(1, h // 10)
                cv2.rectangle(frame, (shield_x1, max(0, shield_y1)), (shield_x1 + shield_w, min(480, shield_y1 + shield_h)), [80, 50, 20], -1) # Dark glass
                
                # Draw headlights if train is approaching
                if pos < 0.8:
                    headlight_r = max(2, int(8 * pos))
                    cv2.circle(frame, (center_x - w // 3, y2 - nose_h // 2), headlight_r, [180, 255, 255], -1)
                    cv2.circle(frame, (center_x + w // 3, y2 - nose_h // 2), headlight_r, [180, 255, 255], -1)
                    # Glow
                    overlay = frame.copy()
                    cv2.circle(overlay, (center_x - w // 3, y2 - nose_h // 2), headlight_r * 4, [100, 220, 220], -1)
                    cv2.circle(overlay, (center_x + w // 3, y2 - nose_h // 2), headlight_r * 4, [100, 220, 220], -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Draw Sensor Line (Cam 1)
        # Let's draw it in high contrast neon cyan at y=312 (approx pos = 0.55)
        sensor_y = 312
        sensor_color = [240, 240, 40] if not self.train_detector_tripped else [40, 40, 255] # Cyan vs Red BGR
        cv2.line(frame, (80, sensor_y), (560, sensor_y), sensor_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "TRAIN DETECTION SENSOR LINE", (90, sensor_y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, sensor_color, 1, cv2.LINE_AA)
                    
        return frame

    def render_cam2(self) -> np.ndarray:
        """Renders Camera 2 (Restricted Crossing) frame."""
        frame = self.cam2_bg.copy()
        
        # Render danger zone boundary on the frame
        # Draw translucent polygon overlay for restricted area
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.restricted_zone_polygon], [40, 100, 240]) # Soft amber/orange BGR
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        
        # Draw dashed outline for restricted zone
        cv2.polylines(frame, [self.restricted_zone_polygon], True, [40, 120, 240], 2, cv2.LINE_AA)

        # Render Intruder if active
        if self.intruder_active:
            x, y = self.intruder_pos
            if self.intruder_type == "pedestrian":
                # Draw shadow
                cv2.circle(frame, (x, y + 2), 7, [15, 15, 15], -1)
                # Draw torso (dark clothing)
                cv2.rectangle(frame, (x - 6, y - 12), (x + 6, y), [80, 40, 40], -1)
                # Draw head (skin tone)
                cv2.circle(frame, (x, y - 16), 5, [140, 200, 240], -1)
                # Draw active bounding box indicators
                cv2.circle(frame, (x, y), 3, [30, 240, 50], -1) # tracker point
            else: # vehicle
                # Draw shadow
                cv2.rectangle(frame, (x - 22, y - 12), (x + 22, y + 16), [15, 15, 15], -1)
                # Car body (deep metallic blue)
                cv2.rectangle(frame, (x - 20, y - 10), (x + 20, y + 12), [150, 80, 50], -1)
                # Windshield
                cv2.rectangle(frame, (x - 14, y - 8), (x + 14, y - 2), [80, 60, 40], -1)
                # Wheels
                cv2.circle(frame, (x - 14, y + 12), 4, [30, 30, 30], -1)
                cv2.circle(frame, (x + 14, y + 12), 4, [30, 30, 30], -1)

        # Draw a simulated train passing diagonally on Cam 2 if train position on Cam 1 is very advanced (meaning it crossed Cam 1, and now simulated on Cam 2 after brief delay)
        # Note: In our system, the coordinate coordinator calculates arrival time. 
        # But to make the visualization fun, let's simulate the train passing Cam 2 under two conditions:
        # 1. 2 seconds after it clears Cam 1 (if no controller is driving it directly) OR
        # 2. When the coordinator reports train is passing (TRAIN_ARRIVED)
        # For pure frame generation, let's just make it possible to render a passing train if triggered.
        # (We will keep it separate: Cam 2 only focuses on restricted zone detection, and train visual is optional).
        
        return frame
