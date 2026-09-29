"""
The sidebar: every video in the batch, colour-coded by state.

    white  = waiting
    yellow = being analysed / reviewed / saved
    green  = done (automatic timing found, or approved at review)
    orange = needs your input at review (e.g. endpoints must be clicked)
    red    = failed or rejected
    grey   = saved without review

Once any video has been reviewed the list splits into "Unreviewed" and
"Reviewed" sections; reviewed rows are faded, with a mark: a grey check
(approved, not saved yet), a green check (saved) or a red cross (skipped).

The list scrolls (mouse wheel / trackpad) when it's longer than the sidebar,
and otherwise follows the active video. Files in `review_targets` can be
clicked; labeler_ui.py turns a click into JumpTo.
"""

import cv2
import numpy as np

from tmwt.ui.panel import Panel
from tmwt.ui.widgets import DIM, FONT, GREEN, GREY, ORANGE, RED, WHITE, YELLOW, truncate

SIDEBAR_W = 320

# Sidebar states and their colours.
WAITING = "waiting"
WORKING = "working"
DONE = "done"
NEEDS_INPUT = "needs_input"
FAILED = "failed"
UNREVIEWED = "unreviewed"
STATE_COLORS = {
    WAITING: WHITE,
    WORKING: YELLOW,
    DONE: GREEN,
    NEEDS_INPUT: ORANGE,
    FAILED: RED,
    UNREVIEWED: GREY,
}
DEFAULT_LEGEND = [("waiting", WHITE), ("working", YELLOW), ("done", GREEN),
                  ("needs input", ORANGE), ("failed", RED)]

# Review outcomes (Sidebar.mark_reviewed): approved but not yet saved (grey
# check), approved and saved to disk (green check), skipped (red cross).
APPROVED_MARK = "approved"
SAVED_MARK = "saved"
REJECTED_MARK = "rejected"

_BG = 25
_X0 = 15                        # left / right text margin
_MARK_W = 26                    # width kept free at the right of a reviewed row for its mark
_REVIEWED_FADE = 0.55           # how much reviewed rows fade toward the background
_TOP = 62                       # first row
_ROW_H = 42
_LEGEND_H = 70
_SCROLL_ROWS_PER_NOTCH = 1.0    # rows moved by one scroll-wheel notch


def _faded(color):
    """A colour faded toward the background, for reviewed rows."""
    return tuple(int(c + (_BG - c) * _REVIEWED_FADE) for c in color)


def _draw_mark(img, outcome, center):
    """A check (grey: approved, green: saved) or a red cross (rejected) centred at `center`."""
    x, y = center
    if outcome in (APPROVED_MARK, SAVED_MARK):
        color = GREEN if outcome == SAVED_MARK else GREY
        cv2.polylines(img, [np.array([(x - 7, y), (x - 2, y + 5), (x + 8, y - 6)], np.int32)],
                      False, color, 2, cv2.LINE_AA)
    else:
        cv2.line(img, (x - 6, y - 6), (x + 6, y + 6), RED, 2, cv2.LINE_AA)
        cv2.line(img, (x - 6, y + 6), (x + 6, y - 6), RED, 2, cv2.LINE_AA)


class Sidebar(Panel):
    """The list of videos (see module docs). Rows and clicks are in sidebar pixels."""

    def __init__(self, x, y, h, names, title="Videos", click_hint="click to review", legend=None):
        """
        Args:
            x, y, h: where the sidebar sits on the canvas, and its height.
            names: file names listed.
            title: heading (shown with the file count).
            click_hint: shown beside the heading while files can be clicked.
            legend: [(label, colour)] under the list; defaults to DEFAULT_LEGEND.
        """
        super().__init__(x, y, SIDEBAR_W, h)
        self.names = list(names)
        self.title = title
        self.click_hint = click_hint
        self.legend = legend if legend is not None else DEFAULT_LEGEND
        self.states = [WAITING] * len(self.names)
        self.notes = [""] * len(self.names)
        self.reviewed = [None] * len(self.names)   # a *_MARK once reviewed
        self.review_targets = set()                # files that can be clicked
        self._active = None           # index of the highlighted file, or None
        self._scroll_first = None     # first row shown; None = follow the active file
        self._wheel_accum = 0.0       # scrolling not yet applied (fractional rows)

    # --- State -----------------------------------------------------------------

    @property
    def active(self):
        """Index of the highlighted file, or None."""
        return self._active

    @active.setter
    def active(self, i):
        # A new active file brings the list back to following it.
        if i != self._active:
            self._scroll_first = None
        self._active = i

    def set_state(self, i, state, note=""):
        """Set file i's state (WAITING, WORKING, ...) and its note line."""
        self.states[i] = state
        self.notes[i] = note

    def mark_reviewed(self, i, outcome):
        """Record file i's review outcome (APPROVED_MARK / SAVED_MARK / REJECTED_MARK); it moves to "Reviewed"."""
        self.reviewed[i] = outcome

    # --- Layout ----------------------------------------------------------------

    def _rows(self):
        """
        The rows, top to bottom: ("file", index) or ("header", text). Once any
        file has been reviewed, the list splits into an "Unreviewed" section and
        a "Reviewed" section below it; until then it's one list.
        """
        files = range(len(self.names))
        if not any(self.reviewed):
            return [("file", i) for i in files]
        todo = [i for i in files if not self.reviewed[i]]
        done = [i for i in files if self.reviewed[i]]
        return ([("header", f"Unreviewed ({len(todo)})")] + [("file", i) for i in todo]
                + [("header", f"Reviewed ({len(done)})")] + [("file", i) for i in done])

    def _visible_rows(self):
        """How many rows fit."""
        return max(1, (self.h - _TOP - _LEGEND_H) // _ROW_H)

    def _first_row(self):
        """Index of the first row shown: the user's scroll position, or centred on the active file."""
        rows = self._rows()
        visible = self._visible_rows()
        last_start = max(0, len(rows) - visible)
        if self._scroll_first is not None:
            return min(max(0, self._scroll_first), last_start)
        if self._active is None:
            return 0
        pos = rows.index(("file", self._active))
        return min(max(0, pos - visible // 2), last_start)

    def row_at(self, pt):
        """Index of the file whose row is at `pt` (sidebar pixels), or None (headers too)."""
        if pt is None or not 0 <= pt[0] < self.w:
            return None
        row = (pt[1] - (_TOP - 4)) // _ROW_H
        if pt[1] < _TOP - 4 or row >= self._visible_rows():
            return None
        rows = self._rows()
        n = self._first_row() + row
        return rows[n][1] if n < len(rows) and rows[n][0] == "file" else None

    def scroll(self, dy):
        """Scroll by a wheel / trackpad movement (dy > 0 = up)."""
        self._wheel_accum -= dy * _SCROLL_ROWS_PER_NOTCH
        rows = int(self._wheel_accum)
        if rows:
            self._wheel_accum -= rows
            self._scroll_first = self._first_row() + rows

    # --- Drawing ---------------------------------------------------------------

    def render(self, mouse, armed):
        """`armed` is the file row the mouse is held down on, or None."""
        panel = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)
        cv2.putText(panel, f"{self.title} ({len(self.names)})", (_X0, 35), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        if self.review_targets:
            (tw, _), _ = cv2.getTextSize(self.click_hint, FONT, 0.4, 1)
            cv2.putText(panel, self.click_hint, (self.w - _X0 - tw, 35), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        cv2.line(panel, (_X0, 48), (self.w - _X0, 48), DIM, 1)
        hovered = self.row_at(mouse)

        rows = self._rows()
        visible = self._visible_rows()
        first = self._first_row()
        for n, (kind, value) in enumerate(rows[first:first + visible]):
            y = _TOP + n * _ROW_H
            if kind == "header":
                cv2.putText(panel, value.upper(), (_X0, y + 22), FONT, 0.42, GREY, 1, cv2.LINE_AA)
                cv2.line(panel, (_X0, y + 30), (self.w - _X0, y + 30), (60, 60, 60), 1)
            else:
                self._draw_file_row(panel, value, y, hovered, armed)

        if len(rows) > visible:
            # Scrollbar: the thumb's size and position show which part of the list is in view.
            track_top, track_h = _TOP - 4, visible * _ROW_H
            thumb_h = max(20, track_h * visible // len(rows))
            thumb_y = track_top + (track_h - thumb_h) * first // (len(rows) - visible)
            cv2.rectangle(panel, (self.w - 6, track_top), (self.w - 3, track_top + track_h), (45, 45, 45), -1)
            cv2.rectangle(panel, (self.w - 6, thumb_y), (self.w - 3, thumb_y + thumb_h), DIM, -1)

        self._draw_legend(panel)
        return panel

    def _draw_file_row(self, panel, i, y, hovered, armed):
        """One file's row: highlight, name (in its state colour) and note; reviewed rows faded with a mark."""
        box = ((5, y - 4), (self.w - 5, y + _ROW_H - 8))
        if i == self.active:
            cv2.rectangle(panel, *box, (55, 55, 55), -1)
        elif i == hovered and i in self.review_targets:
            cv2.rectangle(panel, *box, (40, 40, 40) if armed == i else (45, 45, 45), -1)
            cv2.rectangle(panel, *box, DIM, 1)

        outcome = self.reviewed[i]
        name_color, note_color = STATE_COLORS[self.states[i]], GREY
        if outcome and i != self.active:
            name_color, note_color = _faded(name_color), _faded(note_color)
        text_w = self.w - 2 * _X0 - (_MARK_W if outcome else 0)
        cv2.putText(panel, truncate(f"{i + 1}. {self.names[i]}", text_w, 0.5),
                    (_X0, y + 14), FONT, 0.5, name_color, 1, cv2.LINE_AA)
        if self.notes[i]:
            cv2.putText(panel, truncate(self.notes[i], text_w - 12, 0.4),
                        (_X0 + 12, y + 31), FONT, 0.4, note_color, 1, cv2.LINE_AA)
        if outcome:
            _draw_mark(panel, outcome, (self.w - _X0 - 10, y + 12))

    def _draw_legend(self, panel):
        y = self.h - _LEGEND_H + 20
        cv2.line(panel, (_X0, y - 15), (self.w - _X0, y - 15), DIM, 1)
        x = _X0
        for label, color in self.legend:
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            if x + tw + 16 > self.w - _X0:
                x = _X0
                y += 20
            cv2.circle(panel, (x + 4, y - 4), 4, color, -1)
            cv2.putText(panel, label, (x + 12, y), FONT, 0.38, GREY, 1, cv2.LINE_AA)
            x += tw + 26
