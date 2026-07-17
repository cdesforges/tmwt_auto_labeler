"""
TMWT Labeler (label.py)

Processes all video files in a given input directory:
  1. For each video, tries to auto-detect an ArUco marker for the near endpoint
     and pose landmarks for the far endpoint. Falls back to manual selection
     (two clicks) if either is missing.
  2. Runs pose estimation + ground-plane tracking frame by frame.
  3. Saves per-frame position data (skeleton + rope + timing) to CSV,
     plus annotated and de-identified skeleton videos.

The output CSVs are de-identified — they contain only skeleton landmark
coordinates and rope positions, no video frames. Use view.py to play
them back as a skeleton-only visualization.

Usage:
    python label.py --input_dir <dir> [--output_dir <dir>]
                    [--backend {mediapipe,mmpose}] [--model <path_or_alias>]

Output (per video, in <output_dir>):
    <basename>.csv              — per-frame landmark + rope + timing data
    <basename>_annotated.mp4    — original frames with skeleton/rope + side panel
    <basename>_skeleton.mp4     — de-identified: black canvas + skeleton/rope + panel
"""

import argparse
import os
import sys

import cv2
import numpy as np

from pose_backend import get_backend
from tracking import GroundTracker
from manual_selection import detect_endpoints
from data_export import FrameDataRecorder

# Video file extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}

# Width (px) of the side panel composed next to each frame in the display
# and annotated output video.
PANEL_W = 300

# set up aruco stuff
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)


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


def process_video(video_path, output_path, model_path, backend):
    """
    Process a single video: manual selection, tracking, and data export.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the output CSV.
        model_path: Path/alias for the pose model (backend-specific).
        backend: Pose backend module (see pose_backend.get_backend).
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video {video_path}")
        return

    # find first non-black frame
    first_frame_idx = find_first_frame(cap)
    if first_frame_idx is None:
        print(f"The whole video appeared to be black frames... exiting")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(400 / fps) if fps and fps > 0 else 30
    print(f"  FPS: {fps}, Total frames: {total_frames}")

    # [Step 1] Manual selection — uses a one-shot (IMAGE-mode) landmarker.
    print("\n  [Step 1] Selecting endpoints...")
    image_landmarker = backend.create_image_landmarker(model_path)
    try:
        far_ep, near_ep, manual_start_mode, pose_placed_far_ep = detect_endpoints(
            cap, detector, image_landmarker, backend
        )
    finally:
        image_landmarker.close()
    if near_ep is None:
        print("  ERROR: Endpoint selection failed or was cancelled.")
        cap.release()
        return
    if manual_start_mode:
        print("  Manual start mode: spacebar drives timing.")

    # Read first frame once for GroundTracker init / retry prompt
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
    ret, first_frame_bgr = cap.read()
    if not ret:
        print("  ERROR: Failed to read first frame.")
        cap.release()
        return
    h_frame, w_frame = first_frame_bgr.shape[:2]

    # Video outputs sit next to the CSV:
    #   <basename>_annotated.mp4  → original frames + annotations + panel
    #   <basename>_skeleton.mp4   → black canvas + annotations + panel (de-identified)
    annotated_video_path = os.path.splitext(output_path)[0] + "_annotated.mp4"
    skeleton_video_path = os.path.splitext(output_path)[0] + "_skeleton.mp4"
    video_fps = fps if fps and fps > 0 else 30.0

    # Outer retry loop: pass 1 uses pose-based detection (with optional manual start);
    # pass 2 (if no end was detected) is full manual: spacebar drives start AND stop.
    full_manual = False
    final_recorder = None
    final_frame_idx = 0

    while True:
        # Fresh landmarker each pass — MediaPipe VIDEO mode requires monotonically
        # increasing timestamps, and this keeps mmpose parity simple.
        landmarker = backend.create_landmarker(model_path)
        try:
            tracker = GroundTracker(first_frame_bgr)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            landmarker.close()
            cap.release()
            return
        recorder = FrameDataRecorder(frame_w=w_frame, frame_h=h_frame)

        # Fresh writers each pass — overwrites prior attempts so the final files
        # reflect the final successful (or final-attempted) run.
        # Output size = frame width + side panel width.
        os.makedirs(os.path.dirname(annotated_video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = (w_frame + PANEL_W, h_frame)
        video_writer = cv2.VideoWriter(annotated_video_path, fourcc, video_fps, out_size)
        if not video_writer.isOpened():
            print(f"  WARNING: Could not open annotated video writer at {annotated_video_path}")
            video_writer = None
        skeleton_writer = cv2.VideoWriter(skeleton_video_path, fourcc, video_fps, out_size)
        if not skeleton_writer.isOpened():
            print(f"  WARNING: Could not open skeleton video writer at {skeleton_video_path}")
            skeleton_writer = None

        mode_label = "FULL MANUAL" if full_manual else ("MANUAL START" if manual_start_mode else "AUTO")
        print(f"\n  Processing frames... ({mode_label})")

        walk_start_time, walk_end_time, walk_duration, frame_idx = run_walk_pass(
            cap=cap,
            first_frame_idx=first_frame_idx,
            fps=fps,
            delay=delay,
            tracker=tracker,
            landmarker=landmarker,
            recorder=recorder,
            far_ep=far_ep,
            near_ep=near_ep,
            manual_start_mode=manual_start_mode,
            full_manual=full_manual,
            video_writer=video_writer,
            skeleton_writer=skeleton_writer,
            backend=backend,
            pose_placed_far_ep=pose_placed_far_ep,
        )
        landmarker.close()
        if video_writer is not None:
            video_writer.release()
        if skeleton_writer is not None:
            skeleton_writer.release()
        final_recorder = recorder
        final_frame_idx = frame_idx

        # Done if we got an end, already retried, or there's nothing meaningful to retry
        if walk_end_time is not None or full_manual:
            break

        if not prompt_full_manual_retry(first_frame_bgr):
            break
        full_manual = True

    cap.release()
    cv2.destroyAllWindows()

    if final_recorder is not None:
        final_recorder.save(output_path)
    print(f"  Annotated video: {annotated_video_path}")
    print(f"  Skeleton video:  {skeleton_video_path}")
    print(f"  Done. Processed {final_frame_idx} frames.")


def run_walk_pass(cap, first_frame_idx, fps, delay,
                  tracker, landmarker, recorder,
                  far_ep, near_ep,
                  manual_start_mode, full_manual,
                  video_writer=None, skeleton_writer=None,
                  backend=None,
                  pose_placed_far_ep=False):
    """
    One pass through the video. Returns (walk_start_time, walk_end_time, walk_duration, frame_idx).

    Modes:
      auto             : pose-based start AND end (FAR_T / NEAR_T crossings of t_along).
      manual_start_mode: spacebar = start; pose-based end is still active.
      full_manual      : spacebar = start AND stop; pose-based crossings disabled.
    """
    SMOOTH_ALPHA = 0.7
    FAR_T = 0.0
    NEAR_T = 1.0
    # Fraction of the rope the subject must move forward from their standstill
    # position to be considered "walking". 0.5% ≈ 5cm on a 10m course — small
    # enough to fire quickly, may occasionally trigger on ankle-keypoint noise.
    MOTION_THRESHOLD = 0.005

    prev_t_smooth = None
    prev_time_s = None
    walk_start_time = None
    walk_end_time = None
    walk_duration = None
    # Baseline for motion detection: smallest t_smooth observed so far,
    # and the time it was observed at.
    baseline_t = None
    baseline_time = None
    frame_idx = 0

    spacebar_active = manual_start_mode or full_manual

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ts_ms or ts_ms < 0:
            ts_ms = (frame_idx / fps) * 1000.0 if fps and fps > 0 else frame_idx * delay
        time_s = ts_ms / 1000.0

        H = tracker.update(frame_bgr)
        far_ep_curr, near_ep_curr = GroundTracker.transform_points(H, [far_ep, near_ep])

        all_poses = backend.detect_poses(landmarker, frame_bgr, ts_ms)

        body_px = None
        t_along = None
        t_smooth = None
        pose_lm = None

        if all_poses:
            pose_lm = all_poses[0]
            backend.draw_pose(frame_bgr, pose_lm)
            body_px = backend.get_ankle_midpoint(pose_lm, frame_bgr.shape)

            if body_px is not None:
                cv2.circle(frame_bgr, body_px, 5, (0, 0, 255), -1)

                Ax, Ay = far_ep_curr
                Bx, By = near_ep_curr
                Px, Py = body_px

                v = np.array([Bx - Ax, By - Ay], dtype=np.float32)
                w_vec = np.array([Px - Ax, Py - Ay], dtype=np.float32)
                vv = float(v.dot(v))
                if vv > 1e-6:
                    t_along = float(v.dot(w_vec) / vv)

                    if prev_t_smooth is None:
                        t_smooth = t_along
                    else:
                        t_smooth = SMOOTH_ALPHA * t_along + (1.0 - SMOOTH_ALPHA) * prev_t_smooth

                    # Track the standstill baseline so we can catch the moment
                    # the person begins to move (see MOTION_THRESHOLD below).
                    if walk_start_time is None and (baseline_t is None or t_smooth < baseline_t):
                        baseline_t = t_smooth
                        baseline_time = time_s

                    # Pose-based START: only in pure auto mode
                    if not (manual_start_mode or full_manual) and walk_start_time is None:
                        # Case A: person was behind the user-specified start line and
                        # crossed it — interpolate the sub-frame crossing moment.
                        # DISABLED when far_ep came from the pose detector itself:
                        # ankle-keypoint noise wobbles t_smooth across 0 while the
                        # subject is still standing and fires this branch prematurely.
                        if (not pose_placed_far_ep
                                and prev_t_smooth is not None
                                and prev_t_smooth < FAR_T
                                and t_smooth >= FAR_T):
                            frac = (FAR_T - prev_t_smooth) / (t_smooth - prev_t_smooth) if t_smooth != prev_t_smooth else 0.0
                            walk_start_time = prev_time_s + frac * (time_s - prev_time_s)
                            print(f"  Walk STARTED at {walk_start_time:.3f}s (crossing)")
                        # Case B: motion-threshold — fire once t_smooth has moved
                        # forward from the standstill baseline by more than
                        # MOTION_THRESHOLD. Report the baseline timestamp (when the
                        # subject was closest to standstill just before moving).
                        elif baseline_t is not None and t_smooth - baseline_t >= MOTION_THRESHOLD:
                            walk_start_time = baseline_time
                            print(f"  Walk STARTED at {walk_start_time:.3f}s (motion)")

                    # Pose-based END: any mode except full_manual
                    if not full_manual and walk_start_time is not None and walk_end_time is None:
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

        # De-identified skeleton canvas: black background + same annotations.
        # Only built if we're writing the skeleton video — saves work otherwise.
        skeleton_canvas = None
        if skeleton_writer is not None:
            skeleton_canvas = np.zeros_like(frame_bgr)
            if pose_lm is not None:
                backend.draw_pose(skeleton_canvas, pose_lm)
                if body_px is not None:
                    cv2.circle(skeleton_canvas, body_px, 5, (0, 0, 255), -1)
            cv2.circle(skeleton_canvas, far_ep_curr, 7, (255, 0, 0), -1)
            cv2.circle(skeleton_canvas, near_ep_curr, 7, (0, 0, 255), -1)
            cv2.line(skeleton_canvas, far_ep_curr, near_ep_curr, (0, 255, 255), 2)

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
        panel_w = PANEL_W
        panel = np.zeros((h_frame, panel_w, 3), dtype=np.uint8)
        x0 = 15
        y_pos = 40

        cv2.putText(panel, "TMWT Labeler", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 15
        cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
        y_pos += 30

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
            wait_msg = "Press SPACE when" if spacebar_active else "Waiting for person"
            line2 = "person starts walking" if spacebar_active else "to cross start line..."
            cv2.putText(panel, wait_msg, (x0, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_pos += 25
            cv2.putText(panel, line2, (x0, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        y_pos += 50
        cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
        y_pos += 25

        cv2.putText(panel, f"Frame: {frame_idx}", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        y_pos += 22
        cv2.putText(panel, f"Time:  {time_s:.2f}s", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        y_pos += 22
        if t_along is not None:
            cv2.putText(panel, f"t_along:  {t_along:+.3f}", (x0, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            y_pos += 22
            # Distance from camera, assuming 10m course (near_ep is camera-side).
            dist_from_cam = (1.0 - t_along) * 10.0
            cv2.putText(panel, f"Dist cam: {dist_from_cam:5.2f}m", (x0, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 120), 1)
            y_pos += 22
            # Motion score = how far past the standstill baseline we currently are.
            # Only meaningful before the timer starts; hidden after.
            if walk_start_time is None and baseline_t is not None and t_smooth is not None:
                motion = t_smooth - baseline_t
                motion_m = motion * 10.0
                color = (0, 255, 0) if motion >= MOTION_THRESHOLD else (120, 120, 120)
                cv2.putText(panel, f"Motion:  {motion_m:+.2f}m", (x0, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if full_manual:
            controls = "space = start/stop | q = quit"
        elif manual_start_mode:
            controls = "space = start | q = quit"
        else:
            controls = "q = stop"
        cv2.putText(panel, controls, (x0, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        combined = np.hstack([frame_bgr, panel])
        if video_writer is not None:
            video_writer.write(combined)
        if skeleton_writer is not None and skeleton_canvas is not None:
            skeleton_writer.write(np.hstack([skeleton_canvas, panel]))
        cv2.imshow("TMWT Labeler", combined)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            print("  Stopped early by user.")
            break
        if spacebar_active and key == ord(" "):
            if walk_start_time is None:
                walk_start_time = time_s
                print(f"  Walk STARTED (manual) at {walk_start_time:.3f}s")
            elif full_manual and walk_end_time is None:
                walk_end_time = time_s
                walk_duration = walk_end_time - walk_start_time
                print(f"  Walk FINISHED (manual) at {walk_end_time:.3f}s — Duration: {walk_duration:.3f}s")

        frame_idx += 1

    return walk_start_time, walk_end_time, walk_duration, frame_idx


def prompt_full_manual_retry(frame_bgr):
    """Show a popup asking whether to retry in full manual mode. Returns True if 'r'."""
    window = "No walk end detected"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    display = frame_bgr.copy()
    cv2.putText(display, "No walk end was detected.",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(display, "Press 'r' to retry in full manual mode (spacebar = start AND stop).",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.putText(display, "Any other key to skip and save what we have.",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    cv2.imshow(window, display)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window)
    return key == ord("r")


def main():
    parser = argparse.ArgumentParser(
        description="TMWT Manual Labeler — label walking videos for timing analysis."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing video files to process.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save output CSVs (default: <input_dir>/output).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Pose model. Backend-specific — mediapipe: .task file path "
             "(default: models/pose_landmarker_full.task); mmpose: config path "
             "or alias (default: 'human'); rtmlib: mode name "
             "'balanced' | 'performance' | 'lightweight' (default: 'balanced'). "
             "Falls back to the backend default if unset.",
    )
    parser.add_argument(
        "--backend",
        choices=["mediapipe", "mmpose", "rtmlib"],
        default="mediapipe",
        help="Pose backend to use (default: mediapipe).",
    )
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a valid directory.")
        sys.exit(1)

    # Load backend + resolve model path
    backend = get_backend(args.backend)
    model_path = args.model or backend.DEFAULT_MODEL_PATH
    print(f"Backend: {args.backend}  |  Model: {model_path}")

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
        process_video(video_path, output_path, model_path, backend)

    print(f"\nAll done! Output files are in '{output_dir}'.")

def is_black_frame(frame_bgr, threshold=10):
    return frame_bgr.mean() < threshold

def find_first_frame(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  ERROR: Reached end of video without finding a non-black frame.")
            return None
        if not is_black_frame(frame):
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # rewind to that frame
            print(f"First frame idx: {idx}")
            return idx


if __name__ == "__main__":
    main()
