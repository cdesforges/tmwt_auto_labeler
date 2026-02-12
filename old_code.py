class GaitTimer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Calibration points (Start/Far and End/Near)
        self.string_start = None  # Point A (Far - 0m)
        self.string_end = None  # Point B (Near - 10m)
        self.calibrating = True

        # Timing state
        self.start_time = None
        self.end_time = None
        self.is_running = False
        self.finished = False

    def click_event(self, event, x, y, flags, params):
        """Handle mouse clicks to set up the string endpoints."""
        if event == cv2.EVENT_LBUTTONDOWN and self.calibrating:
            if self.string_start is None:
                self.string_start = (x, y)
                print(f"Set Start Point (Far/0m): {self.string_start}")
            elif self.string_end is None:
                self.string_end = (x, y)
                print(f"Set End Point (Near/10m): {self.string_end}")
                self.calibrating = False  # Calibration complete

    def get_progress_along_string(self, foot_point):
        """
        Projects the foot position onto the vector defined by the string using linear algebra.
        Returns a scalar: 0.0 = at start, 1.0 = at end.
        """
        if not self.string_start or not self.string_end:
            return 0.0

        # Convert to numpy vectors
        A = np.array(self.string_start)
        B = np.array(self.string_end)
        P = np.array(foot_point)

        # Vector AB (The string) and Vector AP (Start to Foot)
        AB = B - A
        AP = P - A

        # Project AP onto AB:  (AP . AB) / |AB|^2
        norm_AB_sq = np.dot(AB, AB)
        if norm_AB_sq == 0: return 0  # Avoid division by zero

        scalar = np.dot(AP, AB) / norm_AB_sq
        return scalar

    def process(self):
        cv2.namedWindow('Gait Analysis')
        cv2.setMouseCallback('Gait Analysis', self.click_event)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            h, w, c = frame.shape

            # --- Calibration Phase ---
            if self.calibrating:
                cv2.putText(frame, "CLICK: 1. Far end of string (Start)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "CLICK: 2. Near end of string (Stop)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.string_start:
                    cv2.circle(frame, self.string_start, 5, (0, 255, 0), -1)

                cv2.imshow('Gait Analysis', frame)
                if cv2.waitKey(0) == 27: break  # Wait indefinitely for clicks
                continue  # Skip processing until calibrated

            # --- Tracking Phase ---
            # Recalculate frame to clear text
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            current_progress = 0.0

            # Draw the virtual string (The Ground Truth Vector)
            cv2.line(frame, self.string_start, self.string_end, (255, 0, 0), 2)
            cv2.circle(frame, self.string_start, 5, (0, 255, 0), -1)
            cv2.circle(frame, self.string_end, 5, (0, 0, 255), -1)

            if results.pose_landmarks:
                # Extract Ankle Coordinates
                landmarks = results.pose_landmarks.landmark

                # We use the midpoint between left and right ankle for better accuracy
                l_ankle = (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x * w),
                           int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y * h))
                r_ankle = (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x * w),
                           int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y * h))

                mid_ankle = ((l_ankle[0] + r_ankle[0]) // 2, (l_ankle[1] + r_ankle[1]) // 2)

                # Visualize feet
                cv2.circle(frame, mid_ankle, 8, (0, 255, 255), -1)

                # --- The 3D Logic ---
                # Calculate progress (0.0 to 1.0)
                current_progress = self.get_progress_along_string(mid_ankle)

                # Draw perpendicular line at foot position to visualize "Plane"
                # This helps visualize the "Start/Stop" line crossing
                # Simple visualization: Line perpendicular to string at progress point
                if 0 <= current_progress <= 1.2:
                    vector = np.array(self.string_end) - np.array(self.string_start)
                    proj_point = np.array(self.string_start) + vector * current_progress
                    proj_point = (int(proj_point[0]), int(proj_point[1]))
                    cv2.line(frame, mid_ankle, proj_point, (255, 255, 0), 1)

                # --- Timing Trigger Logic ---
                # Buffer: We use 0.0 and 1.0.
                # Note: Progress > 0.0 means they passed the start.
                # Progress > 1.0 means they passed the end.

                if not self.is_running and not self.finished:
                    # Waiting to start
                    if current_progress > 0.0:
                        self.start_time = time.time()
                        self.is_running = True
                        print("TIMER STARTED")

                elif self.is_running:
                    # Currently walking
                    if current_progress >= 1.0:
                        self.end_time = time.time()
                        self.is_running = False
                        self.finished = True
                        print("TIMER FINISHED")

            # --- UI Display ---
            elapsed = 0
            if self.is_running:
                elapsed = time.time() - self.start_time
            elif self.finished:
                elapsed = self.end_time - self.start_time

            color = (0, 255, 0) if self.is_running else (0, 255, 255)
            cv2.putText(frame, f"Time: {elapsed:.2f} s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Distance approximation (Linear assumption)
            dist_str = f"Dist: {current_progress * 10:.1f}m"
            cv2.putText(frame, dist_str, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Gait Analysis', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Run the script
# Replace 'walking.mp4' with your video file or 0 for webcam
if __name__ == "__main__":
    # Ensure you have a video file named 'walking.mp4' or change this line
    # app = GaitTimer('walking.mp4')
    print("Please instantiate GaitTimer with a video path in the code.")