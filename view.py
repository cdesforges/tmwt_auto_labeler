"""
TMWT Skeleton Viewer — plays back the labeler's de-identified CSVs.

No video frames are shown: the skeleton and rope are drawn on a black canvas
with the same info panel as the labeler. The walk timing comes from the
<basename>_timing.json that label.py saves next to each CSV, so the viewer
shows exactly what was approved at review. CSVs from before that file existed
fall back to the start-line / finish-line crossings of t_along.

Usage:
    python view.py --input_dir <output folder> [--no_content_crop]

(The folder can also be given without --input_dir, as before.)

Controls:
    q       = next file (quits after the last one)
    space   = pause / resume
    left    = step back 1 frame (while paused)
    right   = step forward 1 frame (while paused)
"""

import argparse
import os
import sys

import cv2
import numpy as np

import annotate
import data_export
import timing
import video_io
from job import COURSE_M

WINDOW = "TMWT Viewer"
# Arrow-key codes from cv2.waitKey (Linux/Windows, macOS).
KEYS_LEFT = (81, 2)
KEYS_RIGHT = (83, 3)
CONTROLS = "q=next  space=pause  <-/-> step"
# Padding kept around the drawn content when cropping the view.
CONTENT_MARGIN = 40
# Fallback canvas size for CSVs without frame_w / frame_h.
DEFAULT_FRAME_W, DEFAULT_FRAME_H = 720, 1280


def find_csvs(input_dir):
    """Frame CSVs in `input_dir` (the labeling report is skipped)."""
    return sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if f.lower().endswith(".csv") and not f.startswith("labeling_report"))


def walk_timing(csv_path, rows):
    """
    (start, end, source) for a CSV: from its _timing.json if present, otherwise
    from the t_along line crossings (older CSVs).
    """
    saved = data_export.read_timing(csv_path)
    if saved is not None:
        return saved["walk_start_s"], saved["walk_end_s"], f"saved ({saved.get('review', '?')})"

    times = [r.get("time_s") or 0.0 for r in rows]
    t_smooth = timing.smooth_t_along([r.get("t_along") for r in rows])
    start = timing.find_crossing(times, t_smooth, timing.FAR_T)
    end = timing.find_crossing(times, t_smooth, timing.NEAR_T, after=start) if start is not None else None
    return start, end, "line crossings (no _timing.json)"


def content_bounds(rows, frame_w, frame_h, margin=CONTENT_MARGIN):
    """
    Bounding box of everything drawn across the recording — every landmark,
    body point and rope endpoint — plus `margin`, clamped to the canvas.

    Playback is drawn on a generated canvas, so there's no matte to detect, but
    a recording often fills only part of the frame. This also tightens CSVs
    recorded before matte cropping existed.

    Returns:
        (x, y, w, h); the full canvas if nothing was drawn.
    """
    xs, ys = [], []
    for row in rows:
        for prefix in ("body", "far_ep", "near_ep"):
            pt = data_export.row_point(row, prefix)
            if pt is not None:
                xs.append(pt[0])
                ys.append(pt[1])
        for lm in data_export.row_pose(row) or []:
            if lm is not None:
                xs.append(lm.x * frame_w)
                ys.append(lm.y * frame_h)
    if not xs:
        return (0, 0, frame_w, frame_h)

    x0 = max(0, int(min(xs)) - margin)
    y0 = max(0, int(min(ys)) - margin)
    x1 = min(frame_w, int(max(xs)) + margin)
    y1 = min(frame_h, int(max(ys)) + margin)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return (0, 0, frame_w, frame_h)
    return (x0, y0, x1 - x0, y1 - y0)


def render_row(row, frame_w, frame_h, crop, walk_start, walk_end, file_name):
    """One frame of the viewer: the cropped skeleton canvas plus the info panel."""
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    annotate.draw_scene(canvas, data_export.row_pose(row),
                        data_export.row_point(row, "body"),
                        data_export.row_point(row, "far_ep"),
                        data_export.row_point(row, "near_ep"))
    # Drawing happens in full-canvas coordinates, so crop afterwards.
    x, y, w, h = crop
    canvas = canvas[y:y + h, x:x + w]
    panel = annotate.draw_info_panel(
        h, row.get("time_s") or 0.0, int(row.get("frame") or 0), row.get("t_along"),
        walk_start, walk_end, title="TMWT Viewer", subtitle=file_name, controls=CONTROLS)
    return np.hstack([canvas, panel])


def play_csv(csv_path, crop_to_content=True):
    """Play one CSV until it ends or the user presses q."""
    file_name = os.path.basename(csv_path)
    print(f"\n{'=' * 60}\nViewing: {file_name}\n{'=' * 60}")

    rows = data_export.read_frames_csv(csv_path)
    if not rows:
        print("  No frame data found.")
        return
    frame_w = int(rows[0].get("frame_w") or DEFAULT_FRAME_W)
    frame_h = int(rows[0].get("frame_h") or DEFAULT_FRAME_H)

    walk_start, walk_end, source = walk_timing(csv_path, rows)
    if walk_start is not None and walk_end is not None:
        duration = walk_end - walk_start
        print(f"  Walk time: {duration:.3f}s ({COURSE_M / duration:.2f} m/s) — {source}")
    else:
        print(f"  Walk timing incomplete — {source}")

    crop = (0, 0, frame_w, frame_h)
    if crop_to_content:
        crop = content_bounds(rows, frame_w, frame_h)
        if crop[2] < frame_w or crop[3] < frame_h:
            print(f"  Cropping view to content: {crop[2]}x{crop[3]} at ({crop[0]},{crop[1]})")
    print(f"  Frames: {len(rows)}, size: {frame_w}x{frame_h}")

    clock = video_io.PlaybackClock()
    paused = False
    i = 0
    while i < len(rows):
        img = render_row(rows[i], frame_w, frame_h, crop, walk_start, walk_end, file_name)
        cv2.imshow(WINDOW, img)
        wait = 0 if paused else max(1, int(clock.ms_until(rows[i].get("time_s") or 0.0)))
        key = cv2.waitKey(wait) & 0xFF

        if key == ord("q"):
            return
        if key == ord(" "):
            paused = not paused
            clock.restart()
            continue
        if paused:
            if key in KEYS_LEFT:
                i = max(0, i - 1)
            elif key in KEYS_RIGHT:
                i = min(len(rows) - 1, i + 1)
            continue
        i += 1


def main():
    parser = argparse.ArgumentParser(
        description="TMWT Skeleton Viewer — play back de-identified CSV data.")
    parser.add_argument("--input_dir", default=None,
                        help="Folder of CSVs written by the labeler (its output folder).")
    parser.add_argument("folder", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_content_crop", action="store_true",
                        help="Show the full recorded frame instead of cropping to the "
                             "region containing drawn content.")
    args = parser.parse_args()
    args.input_dir = args.input_dir or args.folder   # the older positional form still works
    if args.input_dir is None:
        parser.error("the following arguments are required: --input_dir")

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: '{args.input_dir}' is not a valid directory.")
    csvs = find_csvs(args.input_dir)
    if not csvs:
        sys.exit(f"No CSV files found in '{args.input_dir}'.")
    print(f"Found {len(csvs)} CSV file(s):")
    for c in csvs:
        print(f"  - {os.path.basename(c)}")

    for csv_path in csvs:
        play_csv(csv_path, crop_to_content=not args.no_content_crop)
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
