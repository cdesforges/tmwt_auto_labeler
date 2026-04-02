"""
Data export for per-frame tracking data.

Saves frame-by-frame position data to CSV for later analysis.
Each row contains the frame index, timestamp, body position,
rope endpoint positions, the normalized t-value along the rope,
and all 33 MediaPipe pose landmarks (normalized x, y, z).

Landmark coordinates are stored in normalized [0-1] form (as MediaPipe
returns them). To convert back to pixels, multiply x by frame width
and y by frame height. z represents depth relative to the hip midpoint.
"""

import csv
import os

# Number of MediaPipe pose landmarks
NUM_LANDMARKS = 33

# Build CSV column headers:
#   Core columns + 33 landmarks * 3 values each (x, y, z)
HEADERS = [
    "frame",            # Frame index (0-based)
    "time_s",           # Timestamp in seconds
    "frame_w",          # Frame width in pixels (for denormalizing landmarks)
    "frame_h",          # Frame height in pixels (for denormalizing landmarks)
    "body_x",           # Ankle midpoint X (pixels)
    "body_y",           # Ankle midpoint Y (pixels)
    "far_ep_x",         # Far rope endpoint X (pixels)
    "far_ep_y",         # Far rope endpoint Y (pixels)
    "near_ep_x",        # Near rope endpoint X (pixels)
    "near_ep_y",        # Near rope endpoint Y (pixels)
    "t_along",          # Normalized position along rope (0=far, 1=near)
]

# Add landmark columns: lm_00_x, lm_00_y, lm_00_z, lm_01_x, ...
for i in range(NUM_LANDMARKS):
    HEADERS.append(f"lm_{i:02d}_x")
    HEADERS.append(f"lm_{i:02d}_y")
    HEADERS.append(f"lm_{i:02d}_z")


class FrameDataRecorder:
    """
    Collects per-frame tracking data and writes it to a CSV file.

    Usage:
        recorder = FrameDataRecorder(frame_w=1920, frame_h=1080)
        recorder.add_frame(frame_idx=0, time_s=0.0, body_px=(100, 200), ...)
        recorder.save("output.csv")
    """

    def __init__(self, frame_w, frame_h):
        """
        Args:
            frame_w: Video frame width in pixels.
            frame_h: Video frame height in pixels.
        """
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.rows = []

    def add_frame(self, frame_idx, time_s, body_px, far_ep, near_ep, t_along, pose_landmarks=None):
        """
        Record data for a single frame.

        Args:
            frame_idx: Frame number (0-based).
            time_s: Timestamp in seconds.
            body_px: (x, y) ankle midpoint in pixels, or None if no pose.
            far_ep: (x, y) far rope endpoint in pixels.
            near_ep: (x, y) near rope endpoint in pixels.
            t_along: Normalized position along the rope (0=far, 1=near), or None.
            pose_landmarks: List of 33 MediaPipe landmarks with .x, .y, .z, or None.
        """
        row = {
            "frame": frame_idx,
            "time_s": round(time_s, 4),
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "body_x": body_px[0] if body_px else "",
            "body_y": body_px[1] if body_px else "",
            "far_ep_x": far_ep[0],
            "far_ep_y": far_ep[1],
            "near_ep_x": near_ep[0],
            "near_ep_y": near_ep[1],
            "t_along": round(t_along, 6) if t_along is not None else "",
        }

        # Add all 33 landmark positions (normalized)
        for i in range(NUM_LANDMARKS):
            if pose_landmarks and i < len(pose_landmarks):
                lm = pose_landmarks[i]
                row[f"lm_{i:02d}_x"] = round(lm.x, 6)
                row[f"lm_{i:02d}_y"] = round(lm.y, 6)
                row[f"lm_{i:02d}_z"] = round(lm.z, 6)
            else:
                row[f"lm_{i:02d}_x"] = ""
                row[f"lm_{i:02d}_y"] = ""
                row[f"lm_{i:02d}_z"] = ""

        self.rows.append(row)

    def save(self, output_path):
        """
        Write all recorded data to a CSV file.

        Args:
            output_path: Path to the output .csv file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"  Saved {len(self.rows)} frames to {output_path}")
