"""
Per-video outputs, and reading them back (view.py).

For each saved video, next to <basename>.csv:

  <basename>.csv            one row per frame: timestamp, body point, rope
                            endpoints, t_along and all 33 pose landmarks. It
                            holds no image data, so it is de-identified.
  <basename>_timing.json    the walk timing decided by the labeler (start, end,
                            duration, how it was found, review outcome).
  <basename>_annotated.mp4  the video with skeleton, rope and info panel.
  <basename>_skeleton.mp4   the same annotations on a black canvas (de-identified).

Landmark coordinates are normalized (0-1): multiply x by frame_w and y by
frame_h for pixels. z is backend-specific depth (0 for COCO-17 backends).
"""

import csv
import json
import os

import cv2

import annotate
import pose_common
from job import COURSE_M

# Suffixes of the output files, appended to the CSV's basename.
TIMING_SUFFIX = "_timing.json"
ANNOTATED_SUFFIX = "_annotated.mp4"
SKELETON_SUFFIX = "_skeleton.mp4"

CORE_COLUMNS = [
    "frame",            # frame index (0 = first content frame)
    "time_s",           # timestamp in seconds
    "frame_w",          # frame width in pixels (for denormalizing landmarks)
    "frame_h",          # frame height in pixels
    "body_x",           # ankle midpoint X (pixels)
    "body_y",           # ankle midpoint Y (pixels)
    "far_ep_x",         # far rope endpoint X (pixels, this frame)
    "far_ep_y",         # far rope endpoint Y
    "near_ep_x",        # near rope endpoint X
    "near_ep_y",        # near rope endpoint Y
    "t_along",          # position along the rope (0 = far, 1 = near)
]
HEADERS = CORE_COLUMNS + [f"lm_{i:02d}_{axis}"
                          for i in range(pose_common.NUM_LANDMARKS)
                          for axis in ("x", "y", "z")]


def output_paths(csv_path):
    """(timing, annotated, skeleton) paths belonging to a CSV output path."""
    base = os.path.splitext(csv_path)[0]
    return base + TIMING_SUFFIX, base + ANNOTATED_SUFFIX, base + SKELETON_SUFFIX


# --- CSV -----------------------------------------------------------------------

def _csv_row(result, frame_w, frame_h):
    def xy(pt, i):
        return pt[i] if pt is not None else ""

    row = {
        "frame": result.frame_idx,
        "time_s": round(result.time_s, 4),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "body_x": xy(result.body_px, 0),
        "body_y": xy(result.body_px, 1),
        "far_ep_x": xy(result.far_ep, 0),
        "far_ep_y": xy(result.far_ep, 1),
        "near_ep_x": xy(result.near_ep, 0),
        "near_ep_y": xy(result.near_ep, 1),
        "t_along": round(result.t_along, 6) if result.t_along is not None else "",
    }
    pose = result.pose or [None] * pose_common.NUM_LANDMARKS
    for i in range(pose_common.NUM_LANDMARKS):
        lm = pose[i] if i < len(pose) else None
        for axis in ("x", "y", "z"):
            row[f"lm_{i:02d}_{axis}"] = round(getattr(lm, axis), 6) if lm is not None else ""
    return row


def write_frames_csv(path, frames, frame_w, frame_h):
    """Write one CSV row per job.FrameResult."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(_csv_row(r, frame_w, frame_h) for r in frames)
    print(f"  Saved {len(frames)} frames to {path}")


def read_frames_csv(path):
    """
    Read a CSV written by write_frames_csv.

    Returns:
        One dict per row: numeric fields as floats, empty fields as None.
    """
    def parse(value):
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return value

    with open(path, "r", newline="") as f:
        return [{k: parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def row_point(row, prefix):
    """Integer (x, y) from a row's <prefix>_x / <prefix>_y columns, or None."""
    x, y = row.get(f"{prefix}_x"), row.get(f"{prefix}_y")
    return None if x is None or y is None else (int(x), int(y))


def row_pose(row):
    """A row's landmarks as a 33-entry pose (pose_common layout), or None if empty."""
    pose = []
    for i in range(pose_common.NUM_LANDMARKS):
        x, y = row.get(f"lm_{i:02d}_x"), row.get(f"lm_{i:02d}_y")
        z = row.get(f"lm_{i:02d}_z") or 0.0
        pose.append(pose_common.Landmark(x, y, z) if x is not None and y is not None else None)
    return pose if any(lm is not None for lm in pose) else None


# --- Timing --------------------------------------------------------------------

def write_timing(job):
    """Write the job's walk timing and review outcome to <basename>_timing.json."""
    path = output_paths(job.output_path)[0]
    data = {
        "video": job.name,
        "course_m": COURSE_M,
        "walk_start_s": job.walk_start,
        "walk_end_s": job.walk_end,
        "duration_s": job.duration,
        "speed_mps": job.speed,
        "endpoints": job.endpoint_source,
        "timing": job.timing_source,
        "start_method": job.timing_detail,
        "end_rule": job.end_behavior,
        "review": job.review,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_timing(csv_path):
    """The timing saved next to a CSV, as a dict, or None if there isn't one."""
    path = output_paths(csv_path)[0]
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# --- Saving a job --------------------------------------------------------------

def save_job(job, on_progress=None):
    """
    Write all of a job's outputs (CSV, timing, annotated and skeleton videos)
    from its cached analysis. on_progress(fraction) is called per frame.
    """
    h, w = job.info.first_frame.shape[:2]
    _, annotated_path, skeleton_path = output_paths(job.output_path)
    os.makedirs(os.path.dirname(annotated_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (w + annotate.PANEL_W, h)
    writers = []
    for path in (annotated_path, skeleton_path):
        writer = cv2.VideoWriter(path, fourcc, job.info.fps, size)
        if writer.isOpened():
            writers.append(writer)
        else:
            print(f"  WARNING: could not open video writer at {path}")
            writers.append(None)

    cap = job.open_capture()
    try:
        for k, result in enumerate(job.frames):
            ret, frame_bgr = cap.read()
            if ret:
                annotated, skeleton = annotate.render_frame(
                    frame_bgr, result, job.walk_start, job.walk_end, with_skeleton=True)
                for writer, img in zip(writers, (annotated, skeleton)):
                    if writer is not None:
                        writer.write(img)
            if on_progress is not None:
                on_progress((k + 1) / len(job.frames))
    finally:
        cap.release()
        for writer in writers:
            if writer is not None:
                writer.release()

    write_frames_csv(job.output_path, job.frames, w, h)
    write_timing(job)
    print(f"  Videos: {annotated_path}\n          {skeleton_path}")
    job.saved = True
