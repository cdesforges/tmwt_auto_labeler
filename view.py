"""
TMWT Skeleton Viewer — plays back the labeler's de-identified CSVs.

No video frames are shown: the skeleton and rope are drawn on a black canvas
with the same info panel as the labeler. The walk timing comes from the
<basename>_timing.json that label.py saves next to each CSV, so the viewer
shows exactly what was approved at review. CSVs from before that file existed
fall back to the start-line / finish-line crossings of t_along.

It uses the labeler's window: every recording is listed in the sidebar
(coloured by its saved result, with its walk time) and can be clicked to play
it, with previous-recording, frame-back, play / pause, frame-forward and
next-recording buttons below the picture, and a seek bar above them: drag it (or click on it) to scrub through
the recording; the walk's start and end are marked on it in green and red. At
the end of a recording the next one plays; after the last, playback
pauses on its final frame. Close the window (or press Esc) to quit.

Usage:
    python view.py --input_dir <output folder> [--no_content_crop]

(The folder can also be given without --input_dir, as before.)

Keys: Space = pause / play, Left / Right = step one frame, P / N = previous /
next recording, Esc = quit.
"""

import argparse
import os
import sys

import numpy as np

from tmwt.ui import annotate
from tmwt.core import data_export
from tmwt.measurement import timing
from tmwt.core.job import COURSE_M
from tmwt.ui.labeler_ui import JumpTo, LabelerUI, WindowClosed
from tmwt.ui.sidebar import DONE, FAILED, UNREVIEWED, WAITING
from tmwt.ui.top_bar import folder_title
from tmwt.ui.player import Player
from tmwt.ui.widgets import GREEN, GREY, KEY_ESC, RED, WHITE

# Sidebar legend for the viewer.
LEGEND = [("approved", GREEN), ("not reviewed", GREY), ("incomplete", RED),
          ("no timing file", WHITE)]
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
        walk_start, walk_end, title="TMWT Viewer", subtitle=file_name)
    return np.hstack([canvas, panel])


class Recording:
    """One CSV loaded for playback: its rows, canvas size, walk timing and view crop."""

    def __init__(self, csv_path, crop_to_content=True):
        self.name = os.path.basename(csv_path)
        self.rows = data_export.read_frames_csv(csv_path)
        first = self.rows[0] if self.rows else {}
        self.frame_w = int(first.get("frame_w") or DEFAULT_FRAME_W)
        self.frame_h = int(first.get("frame_h") or DEFAULT_FRAME_H)
        self.walk_start, self.walk_end, self.source = walk_timing(csv_path, self.rows)
        self.times = [row.get("time_s") or 0.0 for row in self.rows]
        self.crop = (0, 0, self.frame_w, self.frame_h)
        if crop_to_content and self.rows:
            self.crop = content_bounds(self.rows, self.frame_w, self.frame_h)

    def render(self, k):
        """Frame k of the recording, drawn for the viewer."""
        return render_row(self.rows[k], self.frame_w, self.frame_h, self.crop,
                          self.walk_start, self.walk_end, self.name)

    def time_at(self, k):
        return self.rows[k].get("time_s") or 0.0


    def describe(self):
        """Console summary of the recording."""
        print(f"\n{'=' * 60}\nViewing: {self.name}\n{'=' * 60}")
        if self.walk_start is not None and self.walk_end is not None:
            duration = self.walk_end - self.walk_start
            print(f"  Walk time: {duration:.3f}s ({COURSE_M / duration:.2f} m/s) — {self.source}")
        else:
            print(f"  Walk timing incomplete — {self.source}")
        print(f"  Frames: {len(self.rows)}, size: {self.frame_w}x{self.frame_h}")


def sidebar_state(csv_path):
    """(state, note) for a recording in the sidebar, from its saved timing file."""
    saved = data_export.read_timing(csv_path)
    if saved is None:
        return WAITING, "no timing file (older CSV)"
    duration = saved.get("duration_s")
    if duration is None:
        return FAILED, "timing incomplete"
    reviewed = saved.get("review") == "approved"
    return (DONE if reviewed else UNREVIEWED), f"{'approved' if reviewed else 'not reviewed'}  {duration:.2f}s"


def play(ui, recording, index, count):
    """
    Play one recording in the window until the user moves to another.

    Returns:
        The index of the recording to play next, or None to quit. (Clicking a
        recording in the sidebar raises JumpTo instead.)
    """
    recording.describe()
    if not recording.rows:
        print("  No frame data found.")
        return index + 1 if index + 1 < count else None
    ui.active = index
    ui.start_playback()
    player = Player(recording.times)
    while True:
        transport = player.transport()
        specs = ([("Previous recording", "previous", (ord("p"), ord("P")), "prev_video")] + transport
                 + [("Next recording", "next", (ord("n"), ord("N")), "next_video")])
        value, _ = player.show(ui, recording.render(player.k), specs,
                               marks=[(recording.walk_start, GREEN), (recording.walk_end, RED)],
                               hotkeys={KEY_ESC: "quit"})
        if value == "quit":
            return None
        if value == "previous":
            return max(0, index - 1)
        if value == "next":
            return index + 1 if index + 1 < count else index
        if not player.advance():
            if index + 1 < count:
                return index + 1              # on to the next recording
            player.paused = True              # the last one: hold on its final frame


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
    print(f"Found {len(csvs)} CSV file(s).")

    ui = LabelerUI([os.path.basename(c) for c in csvs], title="Recordings",
                   click_hint="click to view", legend=LEGEND,
                   heading=folder_title(args.input_dir))
    for i, csv_path in enumerate(csvs):
        ui.set_state(i, *sidebar_state(csv_path))
    ui.review_targets = set(range(len(csvs)))   # every recording can be clicked
    try:
        current = 0
        while current is not None:
            try:
                current = play(ui, Recording(csvs[current], not args.no_content_crop),
                               current, len(csvs))
            except JumpTo as jump:
                current = jump.index
    except WindowClosed:
        pass
    finally:
        ui.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
