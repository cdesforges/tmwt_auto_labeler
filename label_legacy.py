"""
TMWT Manual Labeler (label.py)

Processes all video files in a given input directory:
  1. For each video, the user manually selects the rope endpoints.
  2. Runs pose estimation + ground-plane tracking frame by frame.
  3. Saves per-frame position data (skeleton + rope + timing) to CSV.

The output CSVs are de-identified — they contain only skeleton landmark
coordinates and rope positions, no video frames. Use view.py to play
them back as a skeleton-only visualization.

Usage:
    python label.py <input_directory> [--output_dir <output_directory>] [--model <model_path>]

Output:
    One CSV file per video, saved to <output_directory>/<video_name>.csv
    (defaults to an "output" folder next to the input directory).
"""

import argparse
import os
import sys

import cv2
import numpy as np

from pose import create_landmarker, detect_poses, draw_pose, get_ankle_midpoint
from tracking import GroundTracker
from manual_selection import select_rope_endpoints
from data_export import FrameDataRecorder

# Video file extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


def find_videos(input_dir):
    """
    Find all video files in the given directory (non-recursive).

    Args:
        input_dir: Path to the directory to scan.

    Returns:
        Sorted list of full paths to video files.
    """
    videos = []
    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(os.path.join(input_dir, fname))
    return sorted(videos)


def process_video(video_path, output_path, model_path):
    """
    Process a single video: manual selection, tracking, and data export.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the output CSV.
        model_path: Path to the MediaPipe pose model.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(400 / fps) if fps and fps > 0 else 30
    print(f"  FPS: {fps}, Total frames: {total_frames}")

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("  ERROR: Failed to read first frame.")
        cap.release()
        return

    # --- Step 1: Manual rope endpoint selection ---
    print("\n  [Step 1] Select rope endpoints on the first frame.")
    far_ep, near_ep = select_rope_endpoints(first_frame)
    if far_ep is None or near_ep is None:
        print("  Rope selection cancelled. Skipping video.")
        cap.release()
        return
    print(f"  Far endpoint:  {far_ep}")
    print(f"  Near endpoint: {near_ep}")

    # --- Step 2: Create pose landmarker (single person tracking) ---
    print("\n  [Step 2] Initializing pose detector (single person mode)...")
    landmarker = create_landmarker(model_path)

    # --- Step 3: Initialize ground tracking ---
    print("\n  [Step 3] Initializing ground-plane tracker...")
    try:
        tracker = GroundTracker(first_frame)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        landmarker.close()
        cap.release()
        return
    print(f"  Ground features found: {len(tracker.p0)}")

    # --- Step 4: Process all frames ---
    print("\n  [Step 4] Processing frames...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
    h_frame, w_frame = first_frame.shape[:2]
    recorder = FrameDataRecorder(frame_w=w_frame, frame_h=h_frame)
    frame_idx = 0

    # Walk timing state
    SMOOTH_ALPHA = 0.7  # Exponential smoothing for t_along
    FAR_T = 0.0         # t threshold at far (start) end
    NEAR_T = 1.0        # t threshold at near (finish) end

    prev_t_smooth = None
    prev_time_s = None
    walk_start_time = None   # Time when person crosses the far endpoint
    walk_end_time = None     # Time when person crosses the near endpoint
    walk_duration = None     # Final walk time in seconds

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Compute timestamp
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ts_ms or ts_ms < 0:
            ts_ms = (frame_idx / fps) * 1000.0 if fps and fps > 0 else frame_idx * delay
        time_s = ts_ms / 1000.0

        # Update ground-plane homography
        H = tracker.update(frame_bgr)

        # Transform rope endpoints to current frame
        far_ep_curr, near_ep_curr = GroundTracker.transform_points(
            H, [far_ep, near_ep]
        )

        # Run pose detection (single person mode)
        all_poses = detect_poses(landmarker, frame_bgr, ts_ms)

        body_px = None
        t_along = None
        t_smooth = None
        pose_lm = None

        if all_poses:
            pose_lm = all_poses[0]
            draw_pose(frame_bgr, pose_lm)
            body_px = get_ankle_midpoint(pose_lm, frame_bgr.shape)
            cv2.circle(frame_bgr, body_px, 5, (0, 0, 255), -1)

            # Compute t_along (normalized position along the rope line)
            Ax, Ay = far_ep_curr
            Bx, By = near_ep_curr
            Px, Py = body_px

            v = np.array([Bx - Ax, By - Ay], dtype=np.float32)
            w_vec = np.array([Px - Ax, Py - Ay], dtype=np.float32)
            vv = float(v.dot(v))
            if vv > 1e-6:
                t_along = float(v.dot(w_vec) / vv)

                # Smooth the t_along value
                if prev_t_smooth is None:
                    t_smooth = t_along
                else:
                    t_smooth = SMOOTH_ALPHA * t_along + (1.0 - SMOOTH_ALPHA) * prev_t_smooth

                # Detect walk start: crossing the far threshold (t >= 0)
                if walk_start_time is None:
                    if prev_t_smooth is not None and prev_t_smooth < FAR_T and t_smooth >= FAR_T:
                        # Interpolate exact crossing time
                        frac = (FAR_T - prev_t_smooth) / (t_smooth - prev_t_smooth) if t_smooth != prev_t_smooth else 0.0
                        walk_start_time = prev_time_s + frac * (time_s - prev_time_s)
                        print(f"  Walk STARTED at {walk_start_time:.3f}s")
                    elif prev_t_smooth is None and t_smooth >= FAR_T:
                        walk_start_time = time_s
                        print(f"  Walk STARTED at {walk_start_time:.3f}s (initial)")

                # Detect walk end: crossing the near threshold (t >= 1)
                if walk_start_time is not None and walk_end_time is None:
                    if prev_t_smooth is not None and prev_t_smooth < NEAR_T and t_smooth >= NEAR_T:
                        frac = (NEAR_T - prev_t_smooth) / (t_smooth - prev_t_smooth) if t_smooth != prev_t_smooth else 0.0
                        walk_end_time = prev_time_s + frac * (time_s - prev_time_s)
                        walk_duration = walk_end_time - walk_start_time
                        print(f"  Walk FINISHED at {walk_end_time:.3f}s — Duration: {walk_duration:.3f}s")

                prev_t_smooth = t_smooth
                prev_time_s = time_s

        # Draw rope endpoints and line
        cv2.circle(frame_bgr, far_ep_curr, 7, (255, 0, 0), -1)    # Blue = far
        cv2.circle(frame_bgr, near_ep_curr, 7, (0, 0, 255), -1)   # Red = near
        cv2.line(frame_bgr, far_ep_curr, near_ep_curr, (0, 255, 255), 2)

        # Record data for this frame
        recorder.add_frame(
            frame_idx=frame_idx,
            time_s=time_s,
            body_px=body_px,
            far_ep=far_ep_curr,
            near_ep=near_ep_curr,
            t_along=t_along,
            pose_landmarks=pose_lm,
        )

        # --- Build display with side panel ---
        h_frame, w_frame = frame_bgr.shape[:2]
        panel_w = 300
        panel = np.zeros((h_frame, panel_w, 3), dtype=np.uint8)

        # Panel content (white text on black background)
        x0 = 15  # left margin in panel
        y_pos = 40

        # Title
        cv2.putText(panel, "TMWT Labeler", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 15

        # Separator line
        cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
        y_pos += 30

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

        # Timer display
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
        cv2.putText(panel, "q = stop", (x0, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # Combine video + panel side by side
        combined = np.hstack([frame_bgr, panel])

        # Display
        cv2.imshow("TMWT Labeler", combined)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            print("  Stopped early by user.")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Save the data
    recorder.save(output_path)
    print(f"  Done. Processed {frame_idx} frames.")


def main():
    parser = argparse.ArgumentParser(
        description="TMWT Manual Labeler — label walking videos for timing analysis."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing video files to process.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save output CSVs (default: <input_dir>/output).",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
        help="Path to the MediaPipe pose landmarker model file.",
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a valid directory.")
        sys.exit(1)

    # Set up output directory
    output_dir = args.output_dir or os.path.join(args.input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Find videos
    videos = find_videos(args.input_dir)
    if not videos:
        print(f"No video files found in '{args.input_dir}'.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in '{args.input_dir}':")
    for v in videos:
        print(f"  - {os.path.basename(v)}")

    # Process each video
    for video_path in videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}.csv")
        process_video(video_path, output_path, args.model)

    print(f"\nAll done! Output files are in '{output_dir}'.")


if __name__ == "__main__":
    main()
