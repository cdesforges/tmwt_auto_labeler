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

import metric
from pose_backend import get_backend
from tracking import GroundTracker
from manual_selection import detect_endpoints
from data_export import FrameDataRecorder

# Video file extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}

# Width (px) of the side panel composed next to each frame in the display
# and annotated output video.
PANEL_W = 300

# Real length of the walking course, in metres. far_ep is 0 m, near_ep is COURSE_M.
COURSE_M = 10.0

# Forward displacement a foot must make from its standstill position before the
# walk is considered started. This is a TRUE metric distance — it is applied by
# the post-pass refinement in metric.py, which perspective-corrects the track.
MOTION_THRESHOLD_M = 0.05

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


def _to_reference_frame(H, points):
    """
    Map current-frame pixel points back into first-frame (reference) coordinates.

    The ground tracker gives H mapping reference -> current, so we apply its
    inverse. Doing this keeps the accumulated walk track in one consistent frame
    even when the camera drifts, so the vanishing-point fit stays valid.
    """
    if H is None:
        return list(points)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return list(points)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_inv)
    return [(float(p[0][0]), float(p[0][1])) for p in out]


def _landmark_px(lm, frame_w, frame_h):
    """Landmark to (x, y) pixels, or None if the backend dropped it."""
    if lm is None:
        return None
    return (lm.x * frame_w, lm.y * frame_h)


def _project_ankle(lm, frame_w, frame_h, far_ep_curr, near_ep_curr):
    """
    Project one ankle landmark onto the rope direction and return its t_along,
    or None if the landmark is missing or the rope has zero length.
    """
    if lm is None:
        return None
    Ax, Ay = far_ep_curr
    Bx, By = near_ep_curr
    px = lm.x * frame_w
    py = lm.y * frame_h
    vx, vy = float(Bx - Ax), float(By - Ay)
    vv = vx * vx + vy * vy
    if vv <= 1e-6:
        return None
    return ((px - Ax) * vx + (py - Ay) * vy) / vv


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
    # PROVISIONAL live trigger, in image-space t_along units. This is NOT a fixed
    # real distance: because of perspective its true size varies ~8x across the
    # course (0.005 is ~15 cm at the far end but ~2 cm near the camera). It only
    # drives the live on-screen timer; the returned start time is corrected
    # afterwards by the metric refinement below, which works in real metres.
    MOTION_THRESHOLD = 0.005

    prev_t_smooth = None
    prev_time_s = None
    walk_start_time = None
    walk_end_time = None
    walk_duration = None
    # Per-foot baselines: minimum t_along observed for each ankle so far, and
    # the timestamp it was seen at. Motion detection fires when EITHER ankle has
    # moved MOTION_THRESHOLD past its own baseline — catches the lifting foot on
    # the very first step instead of waiting for the midpoint to advance.
    left_baseline_t = None
    left_baseline_time = None
    right_baseline_t = None
    right_baseline_time = None
    # Latest per-foot motion values, exposed for the debug panel.
    left_motion = None
    right_motion = None
    # Walk track in REFERENCE-frame coords, accumulated for the post-pass
    # vanishing-point fit and metric start refinement.
    track = []
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

                    # Project each ankle individually onto the rope direction
                    # so we can catch the lifting foot at the first step.
                    h_f, w_f = frame_bgr.shape[:2]
                    left_lm = pose_lm[backend.LEFT_ANKLE_IDX]
                    right_lm = pose_lm[backend.RIGHT_ANKLE_IDX]
                    left_t = _project_ankle(left_lm, w_f, h_f, far_ep_curr, near_ep_curr)
                    right_t = _project_ankle(right_lm, w_f, h_f, far_ep_curr, near_ep_curr)

                    # Accumulate the track for the post-pass metric refinement.
                    # Everything is back-projected into reference-frame coords so
                    # camera drift doesn't corrupt the vanishing-point fit.
                    head_px = _landmark_px(pose_lm[backend.NOSE_IDX], w_f, h_f)
                    left_px = _landmark_px(left_lm, w_f, h_f)
                    right_px = _landmark_px(right_lm, w_f, h_f)
                    if head_px is not None:
                        ref_pts = _to_reference_frame(
                            H, [body_px, head_px,
                                left_px or body_px, right_px or body_px]
                        )
                        track.append({
                            "time_s": time_s,
                            "foot": ref_pts[0],
                            "head": ref_pts[1],
                            "left_ankle": ref_pts[2] if left_px else None,
                            "right_ankle": ref_pts[3] if right_px else None,
                        })

                    # Track the standstill baseline for each foot.
                    if walk_start_time is None:
                        if left_t is not None and (left_baseline_t is None or left_t < left_baseline_t):
                            left_baseline_t = left_t
                            left_baseline_time = time_s
                        if right_t is not None and (right_baseline_t is None or right_t < right_baseline_t):
                            right_baseline_t = right_t
                            right_baseline_time = time_s

                    left_motion = (left_t - left_baseline_t) if (left_t is not None and left_baseline_t is not None) else None
                    right_motion = (right_t - right_baseline_t) if (right_t is not None and right_baseline_t is not None) else None

                    # Pose-based START: only in pure auto mode. Case A and Case B
                    # are mutually exclusive by branch:
                    #   - User-clicked far_ep (line semantics)   → Case A only
                    #   - Pose-placed far_ep (subject position)  → Case B only
                    if not (manual_start_mode or full_manual) and walk_start_time is None:
                        if not pose_placed_far_ep:
                            # Case A: user picked far_ep as the start LINE. Fire the
                            # moment the ankle-midpoint's smoothed t_along crosses it
                            # from below; interpolate the sub-frame moment.
                            if (prev_t_smooth is not None
                                    and prev_t_smooth < FAR_T
                                    and t_smooth >= FAR_T):
                                frac = (FAR_T - prev_t_smooth) / (t_smooth - prev_t_smooth) if t_smooth != prev_t_smooth else 0.0
                                walk_start_time = prev_time_s + frac * (time_s - prev_time_s)
                                print(f"  Walk STARTED at {walk_start_time:.3f}s (crossing)")
                        else:
                            # Case B: far_ep IS the subject's standstill position, so
                            # crossing it is ill-defined. Fire when EITHER foot has
                            # moved MOTION_THRESHOLD past its own baseline. Report the
                            # baseline timestamp of the foot that lifted first.
                            if left_motion is not None and left_motion >= MOTION_THRESHOLD:
                                walk_start_time = left_baseline_time
                                print(f"  Walk STARTED at {walk_start_time:.3f}s (motion: left foot)")
                            elif right_motion is not None and right_motion >= MOTION_THRESHOLD:
                                walk_start_time = right_baseline_time
                                print(f"  Walk STARTED at {walk_start_time:.3f}s (motion: right foot)")

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
            speed = COURSE_M / walk_duration
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
            # Image-space distance estimate. Marked "~" because it is NOT
            # perspective-corrected — it can be several metres optimistic mid-course.
            # The perspective-correct values are produced by the post-pass refinement.
            dist_from_cam = (1.0 - t_along) * COURSE_M
            cv2.putText(panel, f"~Dist cam:{dist_from_cam:5.2f}m", (x0, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 120), 1)
            y_pos += 22
            # Per-foot motion above baseline, in raw t_along units (not metres —
            # see MOTION_THRESHOLD note above). Only meaningful before start.
            if walk_start_time is None:
                for label, motion in (("L foot:", left_motion), ("R foot:", right_motion)):
                    if motion is None:
                        continue
                    color = (0, 255, 0) if motion >= MOTION_THRESHOLD else (120, 120, 120)
                    cv2.putText(panel, f"{label} {motion:+.4f}t", (x0, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    y_pos += 22

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

    # --- Post-pass metric refinement of the start time ---
    # The live trigger fires on an image-space threshold whose real size varies
    # with distance. Now that the whole track is available we can recover the
    # walking direction's vanishing point and redo the start detection against a
    # true metric threshold. Only applies to the pose-auto path — the manual
    # modes take their start from the user's spacebar, which needs no correction.
    if (walk_start_time is not None
            and pose_placed_far_ep
            and not (manual_start_mode or full_manual)):
        walk_start_time, walk_duration = _refine_start_metric(
            track, far_ep, near_ep, walk_start_time, walk_end_time
        )

    return walk_start_time, walk_end_time, walk_duration, frame_idx


def _refine_start_metric(track, far_ep, near_ep, provisional_start, walk_end_time):
    """
    Recompute the walk start using a true metric threshold.

    Returns (start_time, duration). Falls back to the provisional start — and
    says why — whenever the geometry can't be recovered reliably.
    """
    foot_pts = [s["foot"] for s in track if s.get("foot")]
    head_pts = [s["head"] for s in track if s.get("head")]

    V, info = metric.estimate_vanishing_point(foot_pts, head_pts)
    if V is None:
        print(f"  Metric refinement skipped ({info.get('reason', 'unknown')}); "
              f"keeping image-space start {provisional_start:.3f}s")
        return provisional_start, (walk_end_time - provisional_start
                                   if walk_end_time is not None else None)

    refined, which_foot = metric.refine_start_time(
        track, far_ep, near_ep, V, MOTION_THRESHOLD_M, COURSE_M
    )
    if refined is None:
        print(f"  Metric refinement found no {MOTION_THRESHOLD_M*100:.0f}cm displacement; "
              f"keeping image-space start {provisional_start:.3f}s")
        return provisional_start, (walk_end_time - provisional_start
                                   if walk_end_time is not None else None)

    delta = refined - provisional_start
    print(f"  Walk START refined to {refined:.3f}s "
          f"({MOTION_THRESHOLD_M*100:.0f}cm true displacement, {which_foot} foot, "
          f"{delta:+.3f}s vs image-space estimate)")
    duration = walk_end_time - refined if walk_end_time is not None else None
    if duration is not None:
        print(f"  Corrected duration: {duration:.3f}s "
              f"({COURSE_M/duration:.2f} m/s)")
    return refined, duration


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
