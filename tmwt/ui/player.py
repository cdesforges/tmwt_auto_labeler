"""
Playback shared by the review (session/review.py) and the viewer (view.py).

A Player keeps the position in a clip (frame index k), whether it's paused, and
real-time pacing, and handles the transport controls: frame back / play-pause /
frame forward (buttons and arrow keys) and dragging or clicking the seek bar
(playback pauses while dragging and resumes after if it was playing). It also
remembers which frame was on screen when, so a mark made with a button uses the
frame showing when the button was pressed, not when it was released.

Typical loop:

    player = Player(times)
    while True:
        image = draw frame player.k
        value, pressed_at = player.show(ui, image, specs + player.transport(), marks=...)
        ...handle the caller's own buttons (value)...
        if not player.advance():
            break          # reached the end while playing
"""

import time
from collections import deque

import numpy as np

from tmwt.ui.widgets import KEY_SPACE
from tmwt.ui.window import KEY_LEFT, KEY_RIGHT

# Recent frames remembered, to find what was on screen when a button was
# pressed (a few seconds' worth).
_SHOWN_HISTORY = 300
# Redraw interval while paused, and while the seek bar is dragged.
_PAUSED_MS = 50
_DRAGGING_MS = 20


class PlaybackClock:
    """Paces playback to the video's own timestamps so it runs in real time."""

    def __init__(self):
        self._video_t0 = None
        self._wall_t0 = None

    def ms_until(self, time_s):
        """Milliseconds to wait before showing the frame at video time `time_s`."""
        if self._video_t0 is None:
            self._video_t0, self._wall_t0 = time_s, time.perf_counter()
        return (self._wall_t0 + (time_s - self._video_t0) - time.perf_counter()) * 1000.0

    def restart(self):
        """Re-anchor after a pause or a jump, so the next frame plays immediately."""
        self._video_t0 = None


def seek_state(times, k, marks=()):
    """
    LabelerUI.show_frame's `seek` for frame k of a clip whose frames are at
    `times` (seconds): its position, [(time, colour)] `marks` as markers, and
    the time as elapsed / total.
    """
    t0, span = times[0], max(times[-1] - times[0], 1e-6)
    markers = [((t - t0) / span, color) for t, color in marks if t is not None and t0 <= t <= times[-1]]
    return (times[k] - t0) / span, markers, f"{times[k] - t0:5.2f} / {span:5.2f} s"


def seek_index(times, fraction):
    """The frame at `fraction` (0-1) of the way through a clip, by time."""
    target = times[0] + fraction * (times[-1] - times[0])
    return int(np.argmin(np.abs(np.asarray(times) - target)))


class Player:
    """Position, pause state and transport controls for one clip (see module docs)."""

    def __init__(self, times):
        """`times`: every frame's time in seconds. Space always plays / pauses."""
        self.times = times
        self.k = 0
        self.paused = False
        self._resume_after_seek = None      # set while the seek bar is dragged
        self._hold = False                  # stay on this frame for one more redraw
        self._clock = PlaybackClock()
        self._shown = deque(maxlen=_SHOWN_HISTORY)   # (perf_counter when shown, video time)

    @property
    def last(self):
        return len(self.times) - 1

    def transport(self):
        """Button specs for frame back, play / pause and frame forward."""
        return [("Back one frame", "back", (), "prev_frame"),
                ("Play" if self.paused else "Pause", "toggle", (KEY_SPACE,),
                 "play" if self.paused else "pause"),
                ("Forward one frame", "forward", (), "next_frame")]

    def show(self, ui, image, specs, label=None, marks=(), hotkeys=None, alert=None):
        """
        Show `image` (frame k) with `specs` and the seek bar, waiting as long as
        real-time pacing needs, and apply any transport control used.

        Args:
            marks: [(time, colour)] to mark on the seek bar.
            hotkeys: extra {key: value} for keys without buttons.
            alert: error text to show over the frame (see LabelerUI.show_frame).

        Returns:
            (value, pressed_at): a button value the caller has to handle (None
            if nothing, or a transport control that was applied here), and when
            it was pressed.
        """
        if self._resume_after_seek is not None:
            wait_ms = _DRAGGING_MS
        else:
            wait_ms = _PAUSED_MS if self.paused else self._clock.ms_until(self.times[self.k])
        self._shown.append((time.perf_counter(), self.times[self.k]))
        keys = {KEY_LEFT: "back", KEY_RIGHT: "forward", **(hotkeys or {})}
        dialogs = ui.dialogs_shown
        value, pressed_at = ui.show_frame(image, wait_ms, specs, label, hotkeys=keys,
                                          seek=seek_state(self.times, self.k, marks), alert=alert)
        if ui.dialogs_shown != dialogs:
            self._clock.restart()                    # a dialog paused everything: carry on from here
        if self._apply(value):
            self._hold = True                        # show the new position before moving on
            return None, pressed_at
        return value, pressed_at

    def _apply(self, value):
        """Apply a transport control; True if `value` was one."""
        if isinstance(value, tuple):                 # the seek bar: ("seek" | "seek_end", fraction)
            kind, fraction = value
            self.k = seek_index(self.times, fraction)
            if self._resume_after_seek is None:
                self._resume_after_seek = not self.paused
            self.paused = True                       # hold still while dragging
            if kind == "seek_end":
                self.paused = not self._resume_after_seek
                self._resume_after_seek = None
                self._clock.restart()
            return True
        if value == "toggle":
            self.paused = not self.paused
            self._clock.restart()
            return True
        if value in ("back", "forward"):
            self.paused = True                       # stepping pauses playback
            self.k = max(0, self.k - 1) if value == "back" else min(self.last, self.k + 1)
            return True
        return False

    def advance(self):
        """
        Move on one frame if playing. Returns False once playback reaches the
        end (the position stays on the last frame).
        """
        if self.paused or self._hold:
            self._hold = False
            return True
        if self.k < self.last:
            self.k += 1
            return True
        return False

    def time_on_screen(self, wall_time):
        """Video time of the frame that was on screen at perf_counter time `wall_time`."""
        t = self._shown[-1][1] if self._shown else self.times[self.k]
        for shown_at, video_t in reversed(self._shown):
            t = video_t
            if shown_at <= wall_time:
                break
        return t
