"""
TMWT Labeler (label.py)

Processes every video in an input directory in three phases:

  1. Analyse (unattended): for each video, auto-detect the rope endpoints (pose
     for the far end, ArUco for the near end), run pose estimation + ground-plane
     tracking over every frame, and decide the walk start and end in hindsight
     from the whole track.
  2. Review: each video is played back in real time with its detection overlaid,
     then pauses for the user to approve it or flag a problem:
       - rope endpoints inaccurate → click them manually; timing is recomputed
       - walk start/stop inaccurate → replay with spacebar start/stop timing
       - body not detected         → skip the file (no outputs written)
     Videos whose endpoints couldn't be auto-detected ask for clicks here.
  3. Report: labeling_report.csv / .md summarise every video's outcome.

One window shows analysis progress, playback and the review prompt, with every
file listed and colour-coded in its sidebar (see labeler_ui.py). --no_display
skips the window and review and saves the automatic results unreviewed.

The output CSVs are de-identified — they contain only skeleton landmark
coordinates and rope positions, no video frames. Use view.py to play
them back as a skeleton-only visualization.

Usage:
    python label.py --input_dir <dir> [--output_dir <dir>]
                    [--backend {mediapipe,mmpose,rtmlib}] [--model <path_or_alias>]
                    [--no_matte_crop] [--no_display]

Output (in <output_dir>):
    <basename>.csv              — per-frame landmark + rope + timing data
    <basename>_annotated.mp4    — original frames with skeleton/rope + side panel
    <basename>_skeleton.mp4     — de-identified: black canvas + skeleton/rope + panel
    labeling_report.csv / .md   — per-video outcome of the whole run
"""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np

import matte
import metric
from pose_backend import get_backend
from tracking import GroundTracker
from manual_selection import auto_detect_endpoints, select_rope_endpoints
from data_export import FrameDataRecorder
from labeler_ui import LabelerUI, WHITE, GREY, GREEN, RED, ORANGE
from report import write_report, job_result

# Video file extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}

# Width (px) of the side panel composed next to each frame in the display
# and annotated output video.
PANEL_W = 300

# Real length of the walking course, in metres. far_ep is 0 m, near_ep is COURSE_M.
COURSE_M = 10.0

# Smoothing for t_along (EMA weight on the newest sample) and the t_along values
# of the start line (far_ep) and finish line (near_ep).
SMOOTH_ALPHA = 0.7
FAR_T = 0.0
NEAR_T = 1.0

ESC = 27
ENTER_KEYS = (13, 10)

# set up aruco stuff
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)


class QuitRequested(Exception):
    """Raised when the user presses Esc during analysis."""


@dataclass
class VideoJob:
    """Everything known about one video as it moves through analysis and review."""
    path: str
    output_path: str                    # CSV path; the videos sit next to it
    name: str = ""
    crop: object = None                 # (x, y, w, h) matte crop, or None
    first_frame_idx: int = 0
    fps: float = 30.0
    first_frame: object = None          # first real frame (cropped), for clicks
    frames: list = field(default_factory=list)
    track: list = field(default_factory=list)
    far_ep: object = None
    near_ep: object = None
    pose_placed_far_ep: bool = False
    endpoint_source: str = ""           # "auto" | "manual"
    endpoint_problem: str = ""
    walk_start: object = None
    walk_end: object = None
    timing_source: str = ""             # "auto" | "manual"
    timing_detail: str = ""
    status: str = "pending"             # after analysis: ok | incomplete | needs_input | failed
    error: str = ""
    review: str = "unreviewed"          # approved | rejected | unreviewed
    review_note: str = ""
    saved: bool = False

    @property
    def annotated_path(self):
        return os.path.splitext(self.output_path)[0] + "_annotated.mp4"

    @property
    def skeleton_path(self):
        return os.path.splitext(self.output_path)[0] + "_skeleton.mp4"

    @property
    def duration(self):
        if self.walk_start is None or self.walk_end is None:
            return None
        return self.walk_end - self.walk_start

    def open_capture(self):
        """Reopen the video with the same matte crop, positioned at the first real frame."""
        cap = cv2.VideoCapture(self.path)
        if self.crop is not None:
            cap = matte.CroppingCapture(cap, self.crop)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.first_frame_idx)
        return cap


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


# --- Phase 1: analysis ---------------------------------------------------------

def analyze_job(job, model_path, backend, matte_crop=True, on_progress=None):
    """
    Analyse one video: endpoints, pose + tracking over every frame, and
    automatic walk timing. Sets job.status to ok / incomplete / needs_input / failed.

    Args:
        job: the VideoJob to fill in.
        model_path: Path/alias for the pose model (backend-specific).
        backend: Pose backend module (see pose_backend.get_backend).
        matte_crop: If True, auto-detect and crop solid-color mattes
            (letterbox / pillarbox bars) around the active picture.
        on_progress: optional callback(fraction, frame_bgr, pose_lm), called per frame.
    """
    cap = cv2.VideoCapture(job.path)
    if not cap.isOpened():
        job.status, job.error = "failed", "cannot open video"
        return
    try:
        # find first non-black frame
        first_frame_idx = find_first_frame(cap)
        if first_frame_idx is None:
            job.status, job.error = "failed", "all frames are black"
            return
        job.first_frame_idx = first_frame_idx

        # Detect and crop solid-color mattes from the first real frame.
        # Wrapping the capture makes every subsequent read() return the cropped
        # picture, so all downstream coordinates share one consistent space.
        if matte_crop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
            ret, probe = cap.read()
            if ret:
                crop = matte.detect_content_crop(probe)
                if matte.is_full_frame(crop, probe.shape):
                    print("  No matte detected — using full frame.")
                else:
                    x, y, cw, ch = crop
                    ph, pw = probe.shape[:2]
                    print(f"  Matte detected — cropping to {cw}x{ch} at ({x},{y}) "
                          f"from {pw}x{ph}.")
                    job.crop = crop
                    cap = matte.CroppingCapture(cap, crop)

        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
        ret, first_frame = cap.read()
        if not ret:
            job.status, job.error = "failed", "cannot read first frame"
            return
        job.first_frame = first_frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        job.fps = fps if fps and fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  FPS: {fps}, Total frames: {total_frames}")

        # Endpoints from the first frame — never asks for input here; videos
        # that need clicks are handled at review.
        image_landmarker = backend.create_image_landmarker(model_path)
        try:
            far_ep, near_ep, problem = auto_detect_endpoints(
                first_frame, detector, image_landmarker, backend
            )
        finally:
            image_landmarker.close()

        landmarker = backend.create_landmarker(model_path)
        try:
            tracker = GroundTracker(first_frame)
            job.frames, job.track = analyze_video(
                cap, first_frame_idx, total_frames, tracker, landmarker,
                backend, on_progress,
            )
        finally:
            landmarker.close()
    finally:
        cap.release()

    if not any(f["pose_lm"] is not None for f in job.frames):
        job.status, job.error = "failed", "no body detected in any frame"
        return
    if problem:
        job.status, job.endpoint_problem = "needs_input", problem
        print(f"  Rope endpoints need manual input at review: {problem}")
        return

    job.far_ep, job.near_ep = far_ep, near_ep
    job.endpoint_source = "auto"
    job.pose_placed_far_ep = True
    update_timing(job)
    job.status = "ok" if job.duration is not None else "incomplete"


def analyze_video(cap, first_frame_idx, total_frames, tracker, landmarker,
                  backend, on_progress=None):
    """
    Pose + ground tracking for every frame, with no display.

    Nothing here depends on the rope endpoints, so they can be changed later
    (see apply_endpoints) without re-running pose.

    Returns:
        (frames, track). `frames` has one dict per frame with keys frame_idx,
        time_s, H (reference -> current homography), pose_lm and body_px.
        `track` is the walk track in REFERENCE-frame coords for the
        vanishing-point fit and hindsight start detection.
    """
    frames = []
    track = []
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ts_ms or ts_ms < 0:
            ts_ms = (frame_idx / fps) * 1000.0 if fps and fps > 0 else frame_idx * 33.0
        time_s = ts_ms / 1000.0

        H = tracker.update(frame_bgr)
        all_poses = backend.detect_poses(landmarker, frame_bgr, ts_ms)
        pose_lm = all_poses[0] if all_poses else None
        body_px = backend.get_ankle_midpoint(pose_lm, frame_bgr.shape) if pose_lm is not None else None

        if body_px is not None:
            # Accumulate the track for the hindsight start detection.
            # Everything is back-projected into reference-frame coords so
            # camera drift doesn't corrupt the vanishing-point fit.
            h_f, w_f = frame_bgr.shape[:2]
            head_px = _landmark_px(pose_lm[backend.NOSE_IDX], w_f, h_f)
            left_px = _landmark_px(pose_lm[backend.LEFT_ANKLE_IDX], w_f, h_f)
            right_px = _landmark_px(pose_lm[backend.RIGHT_ANKLE_IDX], w_f, h_f)
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

        frames.append({
            "frame_idx": frame_idx,
            "time_s": time_s,
            "H": H,
            "pose_lm": pose_lm,
            "body_px": body_px,
        })

        frame_idx += 1
        if on_progress is not None:
            on_progress(frame_idx / total_frames if total_frames > 0 else 0.0,
                        frame_bgr, pose_lm)

    return frames, track


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


# --- Walk timing ---------------------------------------------------------------

def apply_endpoints(job):
    """
    Project the job's rope endpoints into every frame and recompute each
    frame's far_ep / near_ep (current-frame coords), t_along and t_smooth.
    """
    prev_t_smooth = None
    for f in job.frames:
        far_c, near_c = GroundTracker.transform_points(f["H"], [job.far_ep, job.near_ep])
        f["far_ep"], f["near_ep"] = far_c, near_c
        t_along = _t_along(f["body_px"], far_c, near_c) if f["body_px"] is not None else None
        f["t_along"] = t_along
        if t_along is None:
            f["t_smooth"] = None
            continue
        if prev_t_smooth is None:
            prev_t_smooth = t_along
        else:
            prev_t_smooth = SMOOTH_ALPHA * t_along + (1.0 - SMOOTH_ALPHA) * prev_t_smooth
        f["t_smooth"] = prev_t_smooth


def update_timing(job):
    """Recompute the automatic walk timing from the job's current endpoints."""
    apply_endpoints(job)
    job.walk_start, job.walk_end, job.timing_detail = detect_walk_times(
        job.frames, job.track, job.far_ep, job.near_ep, job.pose_placed_far_ep
    )
    job.timing_source = "auto"


def _t_along(point, far_ep, near_ep):
    """Image-space fraction of `point` along far_ep -> near_ep, or None if degenerate."""
    Ax, Ay = far_ep
    Bx, By = near_ep
    vx, vy = float(Bx - Ax), float(By - Ay)
    vv = vx * vx + vy * vy
    if vv <= 1e-6:
        return None
    return ((point[0] - Ax) * vx + (point[1] - Ay) * vy) / vv


def _find_crossing(frames, level, after=None):
    """
    Time at which the smoothed t_along first crosses `level` from below,
    interpolated between frames. Only crossings at or after `after` count.
    Returns None if it never crosses.
    """
    prev = None
    for f in frames:
        if f.get("t_smooth") is None:
            continue
        if (prev is not None
                and prev["t_smooth"] < level <= f["t_smooth"]
                and (after is None or f["time_s"] >= after)):
            span = f["t_smooth"] - prev["t_smooth"]
            frac = (level - prev["t_smooth"]) / span if span else 0.0
            return prev["time_s"] + frac * (f["time_s"] - prev["time_s"])
        prev = f
    return None


def _t_smooth_at(frames, time_s):
    """Smoothed t_along of the last frame at or before `time_s`, or None."""
    value = None
    for f in frames:
        if f["time_s"] > time_s:
            break
        if f.get("t_smooth") is not None:
            value = f["t_smooth"]
    return value


def _hindsight_onset(track, far_ep, near_ep, end_time):
    """
    Run metric.find_walk_onset on the track. Returns (start, info, units).
    """
    foot_pts = [s["foot"] for s in track if s.get("foot")]
    head_pts = [s["head"] for s in track if s.get("head")]
    V, info = metric.estimate_vanishing_point(foot_pts, head_pts)
    if V is not None:
        dist_fn = lambda p: metric.metric_along(p, far_ep, near_ep, V, COURSE_M)
        units = "perspective-corrected"
    else:
        # Without the vanishing point, fall back to image-space distance. Its
        # real scale varies with distance from the camera, so thresholds are
        # only approximate.
        print(f"  Vanishing-point fit failed ({info.get('reason', 'unknown')}); "
              f"using image-space distances for start detection.")
        dist_fn = lambda p: (lambda t: None if t is None else t * COURSE_M)(
            _t_along(p, far_ep, near_ep))
        units = "image-space"

    times, left, right, mid = metric.track_distances(track, dist_fn)
    start, onset = metric.find_walk_onset(times, left, right, mid, end_time=end_time)
    return start, onset, units


def detect_walk_times(frames, track, far_ep, near_ep, pose_placed_far_ep):
    """
    Decide the walk start and end from the completed analysis.

    The end is when the ankle midpoint crosses near_ep. The start is found in
    hindsight from the walk that leads into that end (metric.find_walk_onset):
    the first frame a foot leaves its standstill before the walk.

    When far_ep was clicked by the user it is a start LINE, so if the subject
    stood behind it the start is when they cross it; if they were already on
    or past it when they started moving, it is the first foot movement.

    Returns:
        (walk_start_time, walk_end_time, detail); times may be None.
    """
    first_end = _find_crossing(frames, NEAR_T)
    onset, info, units = _hindsight_onset(track, far_ep, near_ep, first_end)

    if onset is not None:
        side = info["foot"]
        print(f"  Walk confirmed at {info['walk_confirmed_at']:.3f}s "
              f"({info['walk_covered_m']:.1f} m sustained advance, {units})")
        print(f"  First foot movement at {onset:.3f}s ({side} foot; "
              f"threshold {info[side + '_threshold_m'] * 100:.0f} cm, "
              f"noise {info[side + '_sigma_m'] * 100:.1f} cm)")
    else:
        print(f"  No walk onset found: {info.get('reason', 'unknown')}")

    if pose_placed_far_ep:
        start = onset
        detail = f"first {info['foot']} foot movement" if onset is not None else ""
    else:
        t_at_onset = _t_smooth_at(frames, onset) if onset is not None else None
        if onset is not None and t_at_onset is not None and t_at_onset >= FAR_T:
            start, detail = onset, f"first {info['foot']} foot movement (on/past start line)"
        else:
            start = _find_crossing(frames, FAR_T, after=onset)
            detail = "start-line crossing"
            if start is None and onset is not None:
                start, detail = onset, f"first {info['foot']} foot movement"

    end = _find_crossing(frames, NEAR_T, after=start) if start is not None else None
    if start is not None:
        print(f"  Walk STARTED at {start:.3f}s ({detail})")
    if end is not None:
        print(f"  Walk FINISHED at {end:.3f}s")
    return start, end, detail


# --- Drawing -------------------------------------------------------------------

def draw_annotated(frame_bgr, f, backend, walk_start, walk_end, spacebar_active,
                   controls, with_skeleton=False):
    """
    Draw pose, rope and the info panel for one frame.

    Returns:
        (annotated, skeleton): frame + panel, and the de-identified black-canvas
        version (only if with_skeleton, else None).
    """
    pose_lm, body_px = f["pose_lm"], f["body_px"]
    far_c, near_c = f.get("far_ep"), f.get("near_ep")

    def annotate(img):
        if pose_lm is not None:
            backend.draw_pose(img, pose_lm)
            if body_px is not None:
                cv2.circle(img, body_px, 5, (0, 0, 255), -1)
        if far_c is not None and near_c is not None:
            cv2.circle(img, far_c, 7, (255, 0, 0), -1)    # Blue = far
            cv2.circle(img, near_c, 7, (0, 0, 255), -1)   # Red = near
            cv2.line(img, far_c, near_c, (0, 255, 255), 2)

    # Walk state as of this frame.
    time_s = f["time_s"]
    started = walk_start is not None and time_s >= walk_start
    finished = started and walk_end is not None and time_s >= walk_end
    panel = _draw_panel(frame_bgr.shape[0], f,
                        walk_start if started else None,
                        walk_end if finished else None,
                        spacebar_active, controls)

    skeleton = None
    if with_skeleton:
        canvas = np.zeros_like(frame_bgr)
        annotate(canvas)
        skeleton = np.hstack([canvas, panel])
    annotate(frame_bgr)
    return np.hstack([frame_bgr, panel]), skeleton


def _draw_panel(h_frame, f, walk_start_time, walk_end_time, spacebar_active, controls):
    """
    Build the info side panel for one frame. `walk_start_time` / `walk_end_time`
    are passed only once the walk has started / finished as of this frame.
    """
    panel_w = PANEL_W
    panel = np.zeros((h_frame, panel_w, 3), dtype=np.uint8)
    x0 = 15
    y_pos = 40
    time_s = f["time_s"]
    t_along = f.get("t_along")

    cv2.putText(panel, "TMWT Labeler", (x0, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 15
    cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
    y_pos += 30

    walk_duration = (walk_end_time - walk_start_time) if walk_end_time is not None else None
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
        line2 = "person starts walking" if spacebar_active else "to start walking..."
        cv2.putText(panel, wait_msg, (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 25
        cv2.putText(panel, line2, (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    y_pos += 50
    cv2.line(panel, (x0, y_pos), (panel_w - x0, y_pos), (80, 80, 80), 1)
    y_pos += 25

    cv2.putText(panel, f"Frame: {f['frame_idx']}", (x0, y_pos),
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
        dist_from_cam = (1.0 - t_along) * COURSE_M
        cv2.putText(panel, f"~Dist cam:{dist_from_cam:5.2f}m", (x0, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 200, 120), 1)

    cv2.putText(panel, controls, (x0, h_frame - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    return panel


# --- Output --------------------------------------------------------------------

def save_job(job, backend, on_progress=None):
    """
    Write the job's CSV and annotated + skeleton videos from the cached analysis.

    on_progress(fraction) is called per frame.
    """
    h_frame, w_frame = job.first_frame.shape[:2]
    recorder = FrameDataRecorder(frame_w=w_frame, frame_h=h_frame)

    os.makedirs(os.path.dirname(job.annotated_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_size = (w_frame + PANEL_W, h_frame)
    writers = []
    for path in (job.annotated_path, job.skeleton_path):
        writer = cv2.VideoWriter(path, fourcc, job.fps, out_size)
        if not writer.isOpened():
            print(f"  WARNING: Could not open video writer at {path}")
            writer = None
        writers.append(writer)
    video_writer, skeleton_writer = writers

    cap = job.open_capture()
    n = len(job.frames)
    for k, f in enumerate(job.frames):
        recorder.add_frame(
            frame_idx=f["frame_idx"],
            time_s=f["time_s"],
            body_px=f["body_px"],
            far_ep=f.get("far_ep"),
            near_ep=f.get("near_ep"),
            t_along=f.get("t_along"),
            pose_landmarks=f["pose_lm"],
        )
        ret, frame_bgr = cap.read()
        if ret and (video_writer is not None or skeleton_writer is not None):
            annotated, skeleton = draw_annotated(
                frame_bgr, f, backend, job.walk_start, job.walk_end,
                spacebar_active=False, controls="", with_skeleton=True,
            )
            if video_writer is not None:
                video_writer.write(annotated)
            if skeleton_writer is not None:
                skeleton_writer.write(skeleton)
        if on_progress is not None:
            on_progress((k + 1) / n)
    cap.release()
    for writer in writers:
        if writer is not None:
            writer.release()

    recorder.save(job.output_path)
    print(f"  Annotated video: {job.annotated_path}")
    print(f"  Skeleton video:  {job.skeleton_path}")
    job.saved = True


# --- Phase 2: review -----------------------------------------------------------

def playback(job, ui, backend, header, manual_timing=False):
    """
    Play the job back in real time in the viewer with its detection overlaid.

    Normal playback: SPACE pauses, ENTER skips to the review prompt.
    Manual timing: SPACE marks the start and then the stop, P pauses.
    Esc quits in both.

    Returns:
        (result, start, end, last_frame) — result is "done" or "quit"; start/end
        are the displayed timing (the spacebar marks, in manual timing).
    """
    if manual_timing:
        start = end = None
        controls = "space = mark start/stop | p = pause"
        pause_key = ord("p")
    else:
        start, end = job.walk_start, job.walk_end
        controls = "space = pause | enter = review"
        pause_key = ord(" ")

    cap = job.open_capture()
    last = None
    video_t0 = wall_t0 = None
    result = "done"
    for f in job.frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        last, _ = draw_annotated(frame_bgr, f, backend, start, end, manual_timing, controls)

        # Pace to the video's own timestamps so playback is real time.
        if video_t0 is None:
            video_t0, wall_t0 = f["time_s"], time.perf_counter()
        wait_ms = (wall_t0 + (f["time_s"] - video_t0) - time.perf_counter()) * 1000.0
        key = ui.show_frame(last, wait_ms, header)

        if key == pause_key:
            while True:
                key = ui.show_frame(last, 50, header + "   [PAUSED]")
                if key in (pause_key, ESC):
                    break
            video_t0, wall_t0 = f["time_s"], time.perf_counter()
        if key == ESC:
            result = "quit"
            break
        if not manual_timing and key in ENTER_KEYS:
            break
        if manual_timing and key == ord(" "):
            if start is None:
                start = f["time_s"]
                print(f"  Walk STARTED (manual) at {start:.3f}s")
            elif end is None and f["time_s"] > start:
                end = f["time_s"]
                print(f"  Walk FINISHED (manual) at {end:.3f}s")
    cap.release()
    return result, start, end, last


def _summary_lines(job):
    lines = [f"Endpoints: {job.endpoint_source or 'none'}    "
             f"Timing: {job.timing_source or 'none'}"
             + (f" ({job.timing_detail})" if job.timing_detail else "")]
    if job.duration is not None:
        lines.append(f"Start {job.walk_start:.2f}s    End {job.walk_end:.2f}s    "
                     f"Duration {job.duration:.2f}s ({COURSE_M / job.duration:.2f} m/s)")
    else:
        start = f"{job.walk_start:.2f}s" if job.walk_start is not None else "not found"
        end = f"{job.walk_end:.2f}s" if job.walk_end is not None else "not found"
        lines.append(f"Timing incomplete: start {start}, end {end}")
    return lines


def _set_manual_endpoints(job):
    """Ask the user to click both rope endpoints, then recompute timing. False if cancelled."""
    far_ep, near_ep = select_rope_endpoints(job.first_frame)
    if far_ep is None or near_ep is None:
        return False
    job.far_ep, job.near_ep = far_ep, near_ep
    job.endpoint_source = "manual"
    job.pose_placed_far_ep = False
    print(f"  Manual endpoints: far {far_ep}, near {near_ep}")
    update_timing(job)
    return True


def _save_with_progress(job, ui, i, backend):
    ui.set_status(i, "saving")
    save_job(job, backend, on_progress=lambda frac: ui.show_progress(
        f"Saving {job.name}", "Writing CSV and annotated videos", frac))


def review_job(job, ui, i, n, backend):
    """
    Play one analysed video back in real time and ask the user to confirm it.
    Returns "quit" if the user asked to stop reviewing, else None.
    """
    print(f"\n  Reviewing {job.name}")
    header = f"Reviewing {job.name} ({i + 1} of {n})"
    ui.active = i
    ui.set_status(i, "reviewing")

    if job.far_ep is None:
        ui.show_message([
            (f"{job.name}: rope endpoints needed", WHITE),
            (f"Automatic detection failed: {job.endpoint_problem}.", ORANGE),
            ("Press any key, then click the FAR end (start) and NEAR end (finish).", GREY),
        ], background=job.first_frame)
        if not _set_manual_endpoints(job):
            job.review, job.review_note = "rejected", "rope endpoints not set"
            ui.set_status(i, "rejected", "rejected: endpoints not set")
            return None

    last = job.first_frame
    note = None
    play = True
    while True:
        if play:
            result, _, _, frame = playback(job, ui, backend, header)
            last = frame if frame is not None else last
            if result == "quit":
                return "quit"
        play = True

        choice = ui.ask_review(last, _summary_lines(job), note)
        note = None
        if choice == "approve":
            if job.duration is None:
                note = "Timing is incomplete: press 3 to time it manually, or 4 to skip."
                play = False
                continue
            job.review = "approved"
            _save_with_progress(job, ui, i, backend)
            ui.set_status(i, "approved", f"approved  {job.duration:.2f}s")
            return None
        if choice == "endpoints":
            if not _set_manual_endpoints(job):
                play = False
        elif choice == "timing":
            ui.show_message([
                ("Manual timing", WHITE),
                ("The video will replay in real time.", GREY),
                ("Press SPACE when the walk starts, and again when it ends.", GREY),
                ("Press any key to begin.", GREY),
            ], background=last)
            result, start, end, frame = playback(
                job, ui, backend, header + "  -  MANUAL TIMING", manual_timing=True)
            last = frame if frame is not None else last
            if result == "quit":
                return "quit"
            if start is not None and end is not None:
                job.walk_start, job.walk_end = start, end
                job.timing_source, job.timing_detail = "manual", "spacebar"
            else:
                note = "Manual timing needs both a start and a stop press. Timing unchanged."
            play = False
        elif choice == "body":
            job.review, job.review_note = "rejected", "body not detected"
            ui.set_status(i, "rejected", "rejected: body not detected")
            print("  Rejected: body not detected — file skipped.")
            return None
        elif choice == "quit":
            return "quit"
        # "replay" falls through to play again.


# --- Batch driver --------------------------------------------------------------

def _analysis_status(job):
    """(ui status, sidebar note) for a job after analysis."""
    if job.status == "ok":
        return "ok", f"auto  {job.duration:.2f}s"
    if job.status == "needs_input":
        return "needs_input", f"needs endpoints: {job.endpoint_problem}"
    if job.status == "incomplete":
        return "failed", _summary_lines(job)[1]
    return "failed", job.error


def run_analysis(jobs, ui, model_path, backend, matte_crop):
    """Phase 1 over every job. Returns True if the user quit part-way."""
    n = len(jobs)
    for i, job in enumerate(jobs):
        print(f"\n{'=' * 60}\nAnalysing ({i + 1}/{n}): {job.name}\n{'=' * 60}")
        next_report = [0.1]

        def on_progress(frac, frame_bgr, pose_lm):
            if ui is None:
                if frac >= next_report[0]:
                    print(f"    {int(next_report[0] * 100):3d}%")
                    next_report[0] += 0.1
                return
            ui.notes[i] = f"analysing {frac * 100:.0f}%"

            def preview():
                img = frame_bgr.copy()
                if pose_lm is not None:
                    backend.draw_pose(img, pose_lm)
                return img
            key = ui.show_progress(f"Analysing {job.name}", f"{i + 1} of {n}", frac, preview)
            if key == ESC:
                raise QuitRequested()

        if ui is not None:
            ui.active = i
            ui.set_status(i, "analysing", "starting...")
            ui.show_progress(f"Analysing {job.name}", f"{i + 1} of {n}", 0.0, force=True)
        try:
            analyze_job(job, model_path, backend, matte_crop, on_progress)
        except QuitRequested:
            print("  Analysis cancelled by user.")
            for rest in jobs[i:]:
                rest.status, rest.error = "failed", "not analysed (run cancelled)"
                rest.frames = []
                if ui is not None:
                    ui.set_status(jobs.index(rest), "failed", rest.error)
            return True
        except Exception as e:
            traceback.print_exc()
            job.status, job.error = "failed", f"error: {e}"
        if ui is not None:
            ui.set_status(i, *_analysis_status(job))
    return False


def run_review(jobs, ui, backend):
    """Phase 2: review every analysable job in turn."""
    n = len(jobs)
    for i, job in enumerate(jobs):
        if job.status == "failed":
            continue
        if review_job(job, ui, i, n, backend) == "quit":
            print("  Review stopped by user.")
            return


def save_unreviewed(jobs, ui, backend):
    """Save the automatic results of every job that wasn't approved or rejected."""
    for i, job in enumerate(jobs):
        if job.saved or job.review == "rejected" or not job.frames or job.far_ep is None:
            continue
        print(f"\n  Saving unreviewed: {job.name}")
        if ui is not None:
            _save_with_progress(job, ui, i, backend)
            ui.set_status(i, "unreviewed", "saved (not reviewed)")
        else:
            save_job(job, backend)


def main():
    parser = argparse.ArgumentParser(
        description="TMWT Labeler — time 10 m walk test videos from pose tracking."
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
    parser.add_argument(
        "--no_matte_crop",
        action="store_true",
        help="Disable automatic cropping of solid-color mattes "
             "(letterbox / pillarbox bars) around the active picture.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Run unattended: no window and no review. Automatic results are "
             "saved unreviewed; videos that need manual endpoints are reported "
             "as failed.",
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

    jobs = []
    for video_path in videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        jobs.append(VideoJob(
            path=video_path,
            output_path=os.path.join(output_dir, f"{video_name}.csv"),
            name=os.path.basename(video_path),
        ))

    ui = None if args.no_display else LabelerUI([job.name for job in jobs])
    quit_early = run_analysis(jobs, ui, model_path, backend, not args.no_matte_crop)
    if ui is not None and not quit_early:
        run_review(jobs, ui, backend)
    save_unreviewed(jobs, ui, backend)
    csv_path, _ = write_report(jobs, output_dir, COURSE_M, args.backend)

    if ui is not None:
        counts = {}
        for job in jobs:
            result = job_result(job)[0]
            counts[result] = counts.get(result, 0) + 1
        ui.active = None
        ui.show_message(
            [("All done", GREEN)]
            + [(f"{result}: {count}", RED if result in ("rejected", "failed") else WHITE)
               for result, count in counts.items()]
            + [(f"Report: {os.path.basename(csv_path)} / .md in the output folder", GREY),
               ("Press any key to close.", GREY)]
        )
        ui.close()

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
