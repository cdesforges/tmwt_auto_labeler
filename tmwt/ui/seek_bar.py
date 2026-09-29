"""
The seek bar shown under playback (LabelerUI.show_frame):

    |=======O-----|------|-----------|   1.23 / 8.96 s
            ^     ^      ^
     playhead     start  end marks (green / red)
                  ▲      ▲  tabs: drag these to move a mark

  - Press on the track (anywhere but a tab) and drag to scrub; the playhead
    follows the pointer.
  - When the frame on screen is a mark's frame, the playhead is filled with the
    mark's colour (keeping its white outline).
  - Stretches of the clip whose pose has flagged points (pose_check.py) are
    orange along the track, however short (at least a few pixels wide).
  - Marks with an id are draggable by the tab under the track. While one is
    dragged, the playhead is hidden, so the mark's new place is clear. Marks
    are grabbed by their tabs rather than their lines because right after
    marking, the mark sits exactly under the playhead, and a press there
    should still scrub.

SeekBar only does geometry, hit-testing and drawing (in main-area pixels); the
drag state is kept by the caller (LabelerUI).
"""

from collections import namedtuple

import cv2
import numpy as np

from tmwt.ui.widgets import BAR_H, FONT, GREY, MAIN_H, MAIN_W, WHITE

# Height of the strip above the button bar that holds the seek bar.
SEEK_H = 34

# A mark on the bar: where (0-1), its colour, and an id if it can be dragged
# (None for a fixed mark).
Marker = namedtuple("Marker", "fraction color id")

# Everything the bar shows (LabelerUI.show_frame's `seek`): the playhead (0-1),
# [Marker], the text at the bar's right (e.g. the time), the colour to fill
# the playhead with (that of a mark on the frame on screen) or None, and
# [(from, to)] stretches (0-1) to colour orange (flagged pose points).
SeekState = namedtuple("SeekState", "fraction markers text playhead_fill flagged",
                       defaults=(None, ()))

# Colour of flagged stretches, and their least width in pixels.
FLAGGED_COLOR = (0, 140, 255)   # orange, as flagged points on the skeleton
_MIN_FLAGGED_W = 3

# Drag kinds (SeekBar.grab): scrubbing, or moving a mark.
SCRUB = "scrub"
MARK = "mark"


class SeekBar:
    """Geometry, hit-testing and drawing of the seek bar (see module docs)."""

    X0, X1 = 24, MAIN_W - 150          # track ends
    Y = MAIN_H - BAR_H - SEEK_H // 2   # track centre line
    GRAB = 12                          # how far above / below the track a press still scrubs
    TAB_TOP, TAB_BOTTOM = 6, 14        # a mark's tab, below the track (offsets from Y)
    TAB_HALF_W = 7

    def fraction_at(self, x):
        """Position along the bar (0-1) for x (clamped to the ends)."""
        return min(1.0, max(0.0, (x - self.X0) / (self.X1 - self.X0)))

    def x_at(self, fraction):
        return int(round(self.X0 + fraction * (self.X1 - self.X0)))

    def _on_track(self, pt):
        return (pt is not None and self.X0 - self.GRAB <= pt[0] <= self.X1 + self.GRAB
                and abs(pt[1] - self.Y) <= self.GRAB)

    def marker_at(self, pt, markers):
        """The id of the draggable mark whose tab is at `pt`, or None (the nearest if tabs overlap)."""
        if pt is None or not self.Y + self.TAB_TOP - 3 <= pt[1] <= self.Y + self.TAB_BOTTOM + 3:
            return None
        near = [(abs(pt[0] - self.x_at(m.fraction)), m.id) for m in markers
                if m.id is not None and abs(pt[0] - self.x_at(m.fraction)) <= self.TAB_HALF_W + 3]
        return min(near)[1] if near else None

    def grab(self, pt, markers):
        """
        What a press at `pt` would drag: (MARK, id) for a mark's tab, (SCRUB,
        None) for the track, or None if it's not on the bar.
        """
        mark = self.marker_at(pt, markers)
        if mark is not None:
            return MARK, mark
        if self._on_track(pt):
            return SCRUB, None
        return None

    def draw(self, img, state, hover=None, drag=None, drag_fraction=None):
        """
        Draw the bar.

        Args:
            state: a SeekState.
            hover: what the pointer is over (as grab() returns), for highlights.
            drag: what's being dragged (as grab() returns), or None.
            drag_fraction: where the pointer is along the bar while dragging.
        """
        fraction, markers, text, fill, flagged = state
        y = self.Y
        dragging_mark = drag is not None and drag[0] == MARK
        if drag is not None and drag[0] == SCRUB:
            fraction = drag_fraction
        x = self.x_at(fraction)
        cv2.line(img, (self.X0, y), (self.X1, y), (70, 70, 70), 4, cv2.LINE_AA)
        cv2.line(img, (self.X0, y), (x, y), (200, 200, 200), 4, cv2.LINE_AA)
        for a, b in flagged:
            xa, xb = self.x_at(a), self.x_at(b)
            if xb - xa < _MIN_FLAGGED_W:
                mid = (xa + xb) // 2
                xa, xb = mid - _MIN_FLAGGED_W // 2, mid + _MIN_FLAGGED_W // 2
            cv2.rectangle(img, (xa, y - 3), (xb, y + 3), FLAGGED_COLOR, -1)

        for m in markers:
            active = dragging_mark and m.id is not None and drag[1] == m.id
            f = drag_fraction if active else m.fraction
            mx = self.x_at(f)
            big = active or (hover == (MARK, m.id) and m.id is not None and drag is None)
            cv2.line(img, (mx, y - (11 if big else 8)), (mx, y + 8), m.color, 3 if big else 2, cv2.LINE_AA)
            if m.id is not None:
                self._draw_tab(img, mx, m.color, big)

        # The playhead, hidden while a mark is dragged so its drop point is clear.
        if not dragging_mark:
            big = drag is not None or hover == (SCRUB, None)
            r = 9 if big else 7
            cv2.circle(img, (x, y), r, WHITE, -1, cv2.LINE_AA)
            if fill is not None:
                cv2.circle(img, (x, y), r - 3, fill, -1, cv2.LINE_AA)   # a 3 px white rim stays
        cv2.putText(img, text, (self.X1 + 18, y + 5), FONT, 0.5, GREY, 1, cv2.LINE_AA)

    def _draw_tab(self, img, x, color, big):
        """A mark's grab tab: a small upward-pointing flag under the track."""
        w = self.TAB_HALF_W + (2 if big else 0)
        top, bottom = self.Y + self.TAB_TOP, self.Y + self.TAB_BOTTOM + (2 if big else 0)
        pts = np.array([(x, top), (x + w, top + 5), (x + w, bottom), (x - w, bottom), (x - w, top + 5)],
                       np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        if big:
            cv2.polylines(img, [pts], True, WHITE, 1, cv2.LINE_AA)
