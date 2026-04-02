"""
TMWT Skeleton Viewer (view.py)

Plays back de-identified CSV data as a skeleton-only visualization.
No video frames are shown — just the pose skeleton and rope endpoints
on a black background, with the same timing side panel as the labeler.

Usage:
    python view.py <csv_directory>

Controls:
    q       = quit / next file
    space   = pause / resume
    left    = step back 1 frame (while paused)
    right   = step forward 1 frame (while paused)
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

from pose import POSE_CONNECTIONS, FACE_IDXS, NOSE_IDX, LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX

# Number of MediaPipe landmarks
NUM_LANDMARKS = 33


def load_csv(csv_path):
    """
    Load a CSV file exported by label.py.

    Returns:
        List of dicts (one per frame), with numeric fields parsed to float.
        Empty list if the file can't be read.
    """
    frames = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, val in row.items():
                if val == "":
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(val)
                    except ValueError:
                        parsed[key] = val
            frames.append(parsed)
    return frames


def get_landmarks_px(frame_data, frame_w, frame_h):
    """
    Extract all 33 landmarks from a frame row as pixel coordinates.

    Args:
        frame_data: Dict for one frame row.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        List of 33 (x, y) tuples in pixel coords, or None for missing landmarks.
    """
    landmarks = []
    for i in range(NUM_LANDMARKS):
        lx = frame_data.get(f"lm_{i:02d}_x")
        ly = frame_data.get(f"lm_{i:02d}_y")
        if lx is not None and ly is not None:
            landmarks.append((int(lx * frame_w), int(ly * frame_h)))
        else:
            landmarks.append(None)
    return landmarks


def draw_skeleton(canvas, landmarks, color=(0, 255, 0), point_radius=4, line_thickness=2):
    """
    Draw the pose skeleton on a canvas using pixel-coordinate landmarks.

    Args:
        canvas: Image to draw on (modified in-place).
        landmarks: List of 33 (x, y) tuples or None for missing landmarks.
        color: BGR color for the skeleton.
        point_radius: Radius of landmark dots.
        line_thickness: Thickness of skeleton lines.
    """
    # Draw skeleton lines
    for a, b in POSE_CONNECTIONS:
        if landmarks[a] is not None and landmarks[b] is not None:
            cv2.line(canvas, landmarks[a], landmarks[b], color, line_thickness)

    # Draw body landmarks (skip face except nose)
    for i, pt in enumerate(landmarks):
        if pt is None:
            continue
        if i in FACE_IDXS and i != NOSE_IDX:
            continue
        cv2.circle(canvas, pt, point_radius, color, -1)


def draw_panel(panel_h, panel_w, frame_data, walk_start_time, walk_end_time, walk_duration, file_name):
    """
    Draw the info side panel (same layout as label.py).

    Returns:
        Panel image (numpy array).
    """
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    x0 = 15
    y_pos = 40

    # Title
    cv2.putText(panel, "TMWT Viewer", (x0, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 20

    # File name
    cv2.putText(panel, file_name, (x0, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    y_pos += 5

    # Separator
    cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
    y_pos += 30

    time_s = frame_data.get("time_s", 0) or 0
    t_along = frame_data.get("t_along")
    frame_idx = int(frame_data.get("frame", 0) or 0)

    # Walk status
    if walk_duration is not None:
        status = "FINISHED"
        status_color = (0, 200, 0)
    elif walk_start_time is not None:
        status = "WALKING"
        status_color = (0, 255, 255)
    else:
        status = "WAITING"
        status_color = (150, 150, 150)

    cv2.putText(panel, "Status:", (x0, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(panel, status, (x0 + 80, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    y_pos += 40

    # Timer
    if walk_duration is not None:
        cv2.putText(panel, "Walk Time", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 35
        cv2.putText(panel, f"{walk_duration:.3f}s", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        y_pos += 30
        speed = 10.0 / walk_duration
        cv2.putText(panel, f"{speed:.2f} m/s", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    elif walk_start_time is not None:
        elapsed = time_s - walk_start_time
        cv2.putText(panel, "Elapsed", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 35
        cv2.putText(panel, f"{elapsed:.2f}s", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    else:
        cv2.putText(panel, "Waiting for person", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 25
        cv2.putText(panel, "to cross start line...", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    y_pos += 50
    cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
    y_pos += 25

    # Frame info
    cv2.putText(panel, f"Frame: {frame_idx}", (x0, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    y_pos += 22
    cv2.putText(panel, f"Time:  {time_s:.2f}s", (x0, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    y_pos += 22
    if t_along is not None:
        cv2.putText(panel, f"t_along: {t_along:.3f}", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    # Controls at bottom
    cv2.putText(panel, "q=quit  space=pause  <-/-> step", (x0, panel_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    return panel


def compute_walk_timing(frames):
    """
    Pre-compute walk start/end times from the frame data using the same
    smoothed t_along threshold logic as label.py.

    Returns:
        (walk_start_time, walk_end_time, walk_duration) — any may be None.
    """
    SMOOTH_ALPHA = 0.7
    FAR_T = 0.0
    NEAR_T = 1.0

    prev_t_smooth = None
    prev_time_s = None
    walk_start_time = None
    walk_end_time = None

    for fd in frames:
        t_along = fd.get("t_along")
        time_s = fd.get("time_s")
        if t_along is None or time_s is None:
            continue

        if prev_t_smooth is None:
            t_smooth = t_along
        else:
            t_smooth = SMOOTH_ALPHA * t_along + (1.0 - SMOOTH_ALPHA) * prev_t_smooth

        if walk_start_time is None:
            if prev_t_smooth is not None and prev_t_smooth < FAR_T and t_smooth >= FAR_T:
                frac = (FAR_T - prev_t_smooth) / (t_smooth - prev_t_smooth) if t_smooth != prev_t_smooth else 0.0
                walk_start_time = prev_time_s + frac * (time_s - prev_time_s)
            elif prev_t_smooth is None and t_smooth >= FAR_T:
                walk_start_time = time_s

        if walk_start_time is not None and walk_end_time is None:
            if prev_t_smooth is not None and prev_t_smooth < NEAR_T and t_smooth >= NEAR_T:
                frac = (NEAR_T - prev_t_smooth) / (t_smooth - prev_t_smooth) if t_smooth != prev_t_smooth else 0.0
                walk_end_time = prev_time_s + frac * (time_s - prev_time_s)

        prev_t_smooth = t_smooth
        prev_time_s = time_s

    walk_duration = (walk_end_time - walk_start_time) if (walk_start_time is not None and walk_end_time is not None) else None
    return walk_start_time, walk_end_time, walk_duration


def get_walk_state_at_frame(frame_data, walk_start_time, walk_end_time, walk_duration):
    """
    Determine the walk state to display for a given frame.

    Returns:
        (display_start, display_end, display_duration) — values to pass
        to the panel drawer. display_start is set once the walk has started,
        display_end/duration once finished.
    """
    time_s = frame_data.get("time_s") or 0
    if walk_start_time is not None and time_s >= walk_start_time:
        if walk_end_time is not None and time_s >= walk_end_time:
            return walk_start_time, walk_end_time, walk_duration
        else:
            return walk_start_time, None, None
    return None, None, None


def play_csv(csv_path):
    """
    Play back a single CSV file as a skeleton-only visualization.

    Args:
        csv_path: Path to the CSV file from label.py.
    """
    file_name = os.path.basename(csv_path)
    print(f"\n{'='*60}")
    print(f"Viewing: {file_name}")
    print(f"{'='*60}")

    frames = load_csv(csv_path)
    if not frames:
        print("  No frame data found.")
        return

    # Get frame dimensions from the CSV
    frame_w = int(frames[0].get("frame_w", 720) or 720)
    frame_h = int(frames[0].get("frame_h", 1280) or 1280)

    # Compute timing from timestamps to get playback delay
    if len(frames) >= 2:
        t0 = frames[0].get("time_s", 0) or 0
        t1 = frames[1].get("time_s", 0) or 0
        dt = t1 - t0
        delay = max(int(dt * 1000), 1) if dt > 0 else 33
    else:
        delay = 33

    # Pre-compute walk timing
    walk_start_time, walk_end_time, walk_duration = compute_walk_timing(frames)
    if walk_duration is not None:
        print(f"  Walk time: {walk_duration:.3f}s ({10.0/walk_duration:.2f} m/s)")
    else:
        print("  Walk timing: incomplete")

    print(f"  Frames: {len(frames)}, Size: {frame_w}x{frame_h}, Delay: {delay}ms")

    panel_w = 300
    paused = False
    frame_i = 0

    while frame_i < len(frames):
        fd = frames[frame_i]

        # --- Draw skeleton on black canvas ---
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Draw rope
        far_x = fd.get("far_ep_x")
        far_y = fd.get("far_ep_y")
        near_x = fd.get("near_ep_x")
        near_y = fd.get("near_ep_y")

        if all(v is not None for v in [far_x, far_y, near_x, near_y]):
            far_pt = (int(far_x), int(far_y))
            near_pt = (int(near_x), int(near_y))
            cv2.line(canvas, far_pt, near_pt, (0, 255, 255), 2)
            cv2.circle(canvas, far_pt, 7, (255, 0, 0), -1)     # Blue = far
            cv2.circle(canvas, near_pt, 7, (0, 0, 255), -1)    # Red = near

        # Draw skeleton
        landmarks = get_landmarks_px(fd, frame_w, frame_h)
        has_pose = any(lm is not None for lm in landmarks)
        if has_pose:
            draw_skeleton(canvas, landmarks)

            # Draw ankle midpoint
            body_x = fd.get("body_x")
            body_y = fd.get("body_y")
            if body_x is not None and body_y is not None:
                cv2.circle(canvas, (int(body_x), int(body_y)), 5, (0, 0, 255), -1)

        # --- Side panel ---
        ds, de, dd = get_walk_state_at_frame(fd, walk_start_time, walk_end_time, walk_duration)
        panel = draw_panel(frame_h, panel_w, fd, ds, de, dd, file_name)

        # Combine canvas + panel
        combined = np.hstack([canvas, panel])
        cv2.imshow("TMWT Viewer", combined)

        # --- Input handling ---
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81 or key == 2:  # left arrow
            if paused and frame_i > 0:
                frame_i -= 1
                continue
        elif key == 83 or key == 3:  # right arrow
            if paused:
                frame_i += 1
                continue

        if not paused:
            frame_i += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="TMWT Skeleton Viewer — play back de-identified CSV data."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing CSV files from label.py.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a valid directory.")
        sys.exit(1)

    # Find all CSV files
    csvs = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".csv")
    ])

    if not csvs:
        print(f"No CSV files found in '{args.input_dir}'.")
        sys.exit(1)

    print(f"Found {len(csvs)} CSV file(s):")
    for c in csvs:
        print(f"  - {os.path.basename(c)}")

    for csv_path in csvs:
        play_csv(csv_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
