"""
The review's playback (review.py): the video in real time with its detection
drawn on, until the reviewer chooses Confirm or the menu.

Below the picture: a seek bar with the walk's start and stop marks (drag their
tabs to move them), the mark buttons (green dot: walk start, red dot: walk
stop; M presses them in turn), frame-step and play buttons (player.py),
Confirm (Enter; shown on hover) and the menu button (☰, Esc). Flagged pose points are orange
and smoothed ones yellow, on the skeleton and along the timeline.
"""

from collections import namedtuple

from tmwt.core import video_io
from tmwt.pose.pose_common import FLAGGED_COLOR, SMOOTHED_COLOR
from tmwt.ui import annotate
from tmwt.ui.player import Player
from tmwt.ui.widgets import CONFIRM, GREEN, KEY_ESC, RED

# Marks: M marks the start, then the stop, then the start again, and so on (the
# tooltip and badge say which is next). Space always plays / pauses.
_MARK_KEYS = (ord("m"), ord("M"))

_STOP_BEFORE_START = "Stop is before start: mark or drag them again"

# Seek bar ids of the two marks, and the playback values that set them.
_MARKS = {"start": "mark_start", "stop": "mark_stop"}

# The playback's own buttons, right of the transport controls.

_MENU = ("Menu (Esc)", "menu", (KEY_ESC,), "menu")

# How the playback ended: "confirm" or "menu"; the marks then (start, stop);
# the last annotated frame shown (or None); and the frame index it was on.
PlaybackResult = namedtuple("PlaybackResult", "value start end last k")


def playback(job, ui, start_k=None, notice=None):
    """
    Play the job back in real time with its detection drawn on (see the module
    docs): the seek bar and marks, mark buttons, frame-back / play-pause /
    frame-forward (player.py), Confirm and the menu button. Plays until
    Confirm or the menu is chosen; the video pauses on its last frame.

    The walk's start and stop can be moved by clicking a mark button (the frame
    on screen when it was pressed) or by dragging a mark's tab along the seek
    bar. Changed marks are shown in a badge at the top left, and a stop before
    the start as an error until it's fixed (Confirm waits for that).

    Args:
        start_k: frame to resume at, paused (e.g. after the menu); None plays
            from the start.
        notice: text to show in orange above the seek bar (e.g. why the last
            action didn't work); the job's timing note is shown otherwise.

    Frames come from a video_io.FrameSource, so scrubbing and stepping back show
    exactly the frames the analysis used.

    Returns:
        A PlaybackResult. The marks are the job's own timing if they weren't
        changed; the job itself isn't changed.
    """
    marks = {"start": job.walk_start, "stop": job.walk_end}
    original = dict(marks)
    player = Player([f.time_s for f in job.frames], highlights=pose_highlights(job))
    if start_k is not None:
        player.k, player.paused = start_k, True
    if notice is None and job.timing_source == "auto" and job.timing_note:
        notice = job.timing_note
    source = video_io.FrameSource(job)
    m_next = "start"   # which mark M sets next
    ui.start_playback()
    last = None
    try:
        while True:
            frame = source.get(player.k)
            if frame is None:
                break
            start, end = marks["start"], marks["stop"]
            backwards = start is not None and end is not None and end <= start
            # A stop before the start isn't a walk: don't draw it as finished.
            last, _ = annotate.render_frame(frame, job.frames[player.k], start,
                                            None if backwards else end,
                                            model_strength=job.model_strength)
            specs = ([_mark_button(which, m_next) for which in _MARKS] + player.transport()
                     + [CONFIRM, _MENU])
            label = None
            if marks != original:
                label = f"EDITED  start {_mark_text(start)}  stop {_mark_text(end)}  (M: {m_next})"
            value, pressed_at = player.show(
                ui, last, specs, label, marks=[(start, GREEN, "start"), (end, RED, "stop")],
                alert=_STOP_BEFORE_START if backwards else None, notice=notice)

            if value == "menu" or (value == "confirm" and not backwards):
                return PlaybackResult(value, marks["start"], marks["stop"], last, player.k)
            if value in _MARKS.values():
                which = "start" if value == "mark_start" else "stop"
                marks[which] = player.time_on_screen(pressed_at)
                m_next = "stop" if which == "start" else "start"
                notice = None
                print(f"  Walk {which} marked at {marks[which]:.3f}s")
            elif isinstance(value, tuple) and value[0] == "mark":
                _, which, t = value
                marks[which] = t
                notice = None
                print(f"  Walk {which} dragged to {t:.3f}s")
            if not player.advance():
                player.paused = True   # stay on the last frame
    finally:
        source.close()
    return PlaybackResult("menu", marks["start"], marks["stop"], last, player.k)


def _mark_button(which, m_next):
    """The green (start) / red (stop) dot button; M works on the one it sets next."""
    m = which == m_next
    return (f"Mark walk {which}" + (" (M)" if m else ""), _MARKS[which],
            _MARK_KEYS if m else (), _MARKS[which])


def _mark_text(t):
    """A mark for the badge: its time, or a dash if not marked yet."""
    return "--" if t is None else f"{t:.2f}s"


def pose_highlights(job):
    """Seek-bar colours: orange for frames with flagged points left, yellow for smoothed ones."""
    return {FLAGGED_COLOR: [k for k, f in enumerate(job.frames) if set(f.pose_flags) - f.pose_smoothed],
            SMOOTHED_COLOR: [k for k, f in enumerate(job.frames) if f.pose_smoothed]}
