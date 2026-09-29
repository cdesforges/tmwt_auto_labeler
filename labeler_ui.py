"""
The labeler's single window. Nothing else in the labeler opens a window, and
every choice the user makes is a clickable button (most also have an optional
keyboard shortcut).

Screens are drawn with OpenCV into a fixed-size canvas (MAIN_W + SIDEBAR_W by
MAIN_H) and shown in a resizable window (window.py), which scales the canvas
to fit and reports mouse positions in canvas pixels. So all layout and
hit-testing here is in canvas pixels, whatever size the window is.

Two regions:
  - Main area (left): analysis progress, real-time playback, endpoint picking,
    the review prompt and messages.
  - Sidebar (right): every video in the batch, colour-coded by state:
      white  = waiting
      yellow = being analysed / reviewed / saved
      green  = done (automatic timing found, or approved at review)
      orange = needs your input at review (e.g. endpoints must be clicked)
      red    = failed or rejected
      grey   = saved without review

Drawing building blocks (colours, buttons, icons, the seek bar) are in
widgets.py; playback controls are in player.py.

During review, files in `review_targets` can be clicked in the sidebar. The
click raises JumpTo from whatever screen is showing, so the caller (label.py)
can switch to reviewing that file.
"""

import time

import cv2
import numpy as np

import pose_common
from widgets import (BAR_H, BTN_H, DIM, FONT, GREEN, GREY, HEADER_H, KEY_BACKSPACE,
                     KEY_ENTER, KEY_ESC, MAIN_H, MAIN_W, ORANGE, RED, WHITE, YELLOW, BLUE,
                     SEEK_H, Button, bar_buttons, button_row, dimmed, draw_seek_bar,
                     frame_screen, on_seek_bar, put_centered, seek_fraction, truncate)
from window import Window

WINDOW = "TMWT Labeler"
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
_LEGEND = [("waiting", WHITE), ("working", YELLOW), ("done", GREEN),
           ("needs input", ORANGE), ("failed", RED)]

# Review prompt options: (shortcut label, text, value, keys).
REVIEW_OPTIONS = [
    ("1", "Looks good", "approve", (ord("1"),) + KEY_ENTER),
    ("2", "Rope endpoints inaccurate (re-click them)", "endpoints", (ord("2"),)),
    ("3", "Walk start/stop inaccurate (time it manually)", "timing", (ord("3"),)),
    ("4", "Skip this file", "skip", (ord("4"),)),
    ("R", "Replay", "replay", (ord("r"), ord("R"))),
    ("Esc", "Quit review (save the rest unreviewed)", "quit", (KEY_ESC,)),
]
# Offered on the review prompt only when more than one person was tracked.
WRONG_PERSON_OPTION = ("5", "Wrong person tracked (pick the walker)", "person", (ord("5"),))

# Colours for telling people apart on the "pick the walker" screen.
PERSON_COLORS = [(0, 255, 0), (255, 160, 0), (255, 0, 255), (0, 200, 255), (60, 60, 255)]

# Review outcomes shown in the sidebar (LabelerUI.mark_reviewed): approved but
# not yet saved (grey check), approved and saved to disk (green check), skipped
# (red cross).
APPROVED_MARK = "approved"
SAVED_MARK = "saved"
REJECTED_MARK = "rejected"
# Width kept free at the right of a reviewed row for its mark.
_MARK_W = 26
_SIDEBAR_BG = 25
# How much reviewed rows are faded toward the background (0 = not at all).
_REVIEWED_FADE = 0.55

# Sidebar layout, and how many rows one scroll-wheel notch moves.
_SIDEBAR_TOP = 62
_SIDEBAR_ROW_H = 42
_SIDEBAR_LEGEND_H = 70
_SCROLL_ROWS_PER_NOTCH = 1.0

# Minimum interval between progress redraws, so drawing never slows analysis.
_PROGRESS_REDRAW_S = 0.07
# Brightness of a background frame behind text.
_DIM_PROGRESS = 0.3
_DIM_MESSAGE = 0.25
_DIM_PROMPT = 0.12


# --- Sidebar drawing helpers ---------------------------------------------------

def _dim(color):
    """A colour faded toward the sidebar background, for reviewed rows."""
    return tuple(int(c + (_SIDEBAR_BG - c) * _REVIEWED_FADE) for c in color)


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



class WindowClosed(Exception):
    """The user closed the window. Raised from any screen so the program can stop cleanly."""


class JumpTo(Exception):
    """A file in the sidebar was clicked; `index` is which one."""

    def __init__(self, index):
        super().__init__(index)
        self.index = index


class LabelerUI:
    """The batch window. All drawing, key polling and clicks go through here."""

    def __init__(self, names, title="Videos", click_hint="click to review", legend=None):
        """
        Args:
            names: file names listed in the sidebar.
            title: sidebar heading (shown with the file count).
            click_hint: shown beside the heading while files can be clicked.
            legend: [(label, colour)] under the list; defaults to the review states.
        """
        self.names = list(names)
        self.title = title
        self.click_hint = click_hint
        self.legend = legend if legend is not None else _LEGEND
        self.states = [WAITING] * len(self.names)
        self.notes = [""] * len(self.names)
        self.reviewed = [None] * len(self.names)   # APPROVED_MARK / REJECTED_MARK once reviewed
        self._active = None           # index of the highlighted file, or None
        self._scroll_first = None     # first sidebar row shown; None = follow the active file
        self._last_progress_draw = 0.0
        self._armed = None            # value of the button the mouse is pressed on
        self._armed_at = None         # when that press happened (perf_counter)
        self._wheel_accum = 0.0       # sidebar scroll not yet applied (fractional rows)
        self.review_targets = set()   # files that can be clicked in the sidebar (see JumpTo)
        self._armed_row = None        # sidebar row the mouse was pressed on
        self._seeking = False         # the seek bar is being dragged
        self._window = Window(WINDOW, MAIN_W + SIDEBAR_W, MAIN_H)

    # --- Sidebar ---------------------------------------------------------------

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
        """Set file i's sidebar state (WAITING, WORKING, ...) and its note line."""
        self.states[i] = state
        self.notes[i] = note

    def mark_reviewed(self, i, outcome):
        """Record file i's review outcome (APPROVED_MARK / SAVED_MARK / REJECTED_MARK); it moves to "Reviewed"."""
        self.reviewed[i] = outcome

    def _display_rows(self):
        """
        The sidebar's rows, top to bottom: ("file", index) or ("header", text).
        Once any file has been reviewed, the list splits into an "Unreviewed"
        section and a "Reviewed" section below it; until then it's one list.
        """
        files = range(len(self.names))
        if not any(self.reviewed):
            return [("file", i) for i in files]
        todo = [i for i in files if not self.reviewed[i]]
        done = [i for i in files if self.reviewed[i]]
        return ([("header", f"Unreviewed ({len(todo)})")] + [("file", i) for i in todo]
                + [("header", f"Reviewed ({len(done)})")] + [("file", i) for i in done])

    def _sidebar(self):
        panel = np.full((MAIN_H, SIDEBAR_W, 3), _SIDEBAR_BG, dtype=np.uint8)
        x0 = 15
        cv2.putText(panel, f"{self.title} ({len(self.names)})", (x0, 35), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        if self.review_targets:
            hint = self.click_hint
            (tw, _), _ = cv2.getTextSize(hint, FONT, 0.4, 1)
            cv2.putText(panel, hint, (SIDEBAR_W - x0 - tw, 35), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        cv2.line(panel, (x0, 48), (SIDEBAR_W - x0, 48), DIM, 1)
        hovered = self._sidebar_row_at(self._window.mouse_pos)

        rows = self._display_rows()
        top, row_h = _SIDEBAR_TOP, _SIDEBAR_ROW_H
        visible = self._sidebar_rows()
        first = self._sidebar_first()
        for n, (kind, value) in enumerate(rows[first:first + visible]):
            y = top + n * row_h
            if kind == "header":
                cv2.putText(panel, value.upper(), (x0, y + 22), FONT, 0.42, GREY, 1, cv2.LINE_AA)
                cv2.line(panel, (x0, y + 30), (SIDEBAR_W - x0, y + 30), (60, 60, 60), 1)
                continue
            self._draw_file_row(panel, value, y, hovered)

        if len(rows) > visible:
            # Scrollbar: the thumb's size and position show which part of the list is in view.
            track_top, track_h = top - 4, visible * row_h
            thumb_h = max(20, track_h * visible // len(rows))
            thumb_y = track_top + (track_h - thumb_h) * first // (len(rows) - visible)
            cv2.rectangle(panel, (SIDEBAR_W - 6, track_top), (SIDEBAR_W - 3, track_top + track_h), (45, 45, 45), -1)
            cv2.rectangle(panel, (SIDEBAR_W - 6, thumb_y), (SIDEBAR_W - 3, thumb_y + thumb_h), DIM, -1)

        y = MAIN_H - _SIDEBAR_LEGEND_H + 20
        cv2.line(panel, (x0, y - 15), (SIDEBAR_W - x0, y - 15), DIM, 1)
        x = x0
        for label, color in self.legend:
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            if x + tw + 16 > SIDEBAR_W - x0:
                x = x0
                y += 20
            cv2.circle(panel, (x + 4, y - 4), 4, color, -1)
            cv2.putText(panel, label, (x + 12, y), FONT, 0.38, GREY, 1, cv2.LINE_AA)
            x += tw + 26
        return panel

    def _draw_file_row(self, panel, i, y, hovered):
        """One file's row: highlight, name (in its state colour) and note; reviewed rows dimmed with a mark."""
        x0, row_h = 15, _SIDEBAR_ROW_H
        box = ((5, y - 4), (SIDEBAR_W - 5, y + row_h - 8))
        if i == self.active:
            cv2.rectangle(panel, *box, (55, 55, 55), -1)
        elif i == hovered and i in self.review_targets:
            cv2.rectangle(panel, *box, (40, 40, 40) if self._armed_row == i else (45, 45, 45), -1)
            cv2.rectangle(panel, *box, DIM, 1)

        outcome = self.reviewed[i]
        name_color, note_color = STATE_COLORS[self.states[i]], GREY
        if outcome and i != self.active:
            name_color, note_color = _dim(name_color), _dim(note_color)
        text_w = SIDEBAR_W - 2 * x0 - (_MARK_W if outcome else 0)
        cv2.putText(panel, truncate(f"{i + 1}. {self.names[i]}", text_w, 0.5),
                    (x0, y + 14), FONT, 0.5, name_color, 1, cv2.LINE_AA)
        if self.notes[i]:
            cv2.putText(panel, truncate(self.notes[i], text_w - 12, 0.4),
                        (x0 + 12, y + 31), FONT, 0.4, note_color, 1, cv2.LINE_AA)
        if outcome:
            _draw_mark(panel, outcome, (SIDEBAR_W - x0 - 10, y + 12))

    def _sidebar_rows(self):
        """How many rows fit in the sidebar."""
        return max(1, (MAIN_H - _SIDEBAR_TOP - _SIDEBAR_LEGEND_H) // _SIDEBAR_ROW_H)

    def _sidebar_first(self):
        """Index of the first row shown: the user's scroll position, or centred on the active file."""
        rows = self._display_rows()
        visible = self._sidebar_rows()
        last_start = max(0, len(rows) - visible)
        if self._scroll_first is not None:
            return min(max(0, self._scroll_first), last_start)
        if self._active is None:
            return 0
        pos = rows.index(("file", self._active))
        return min(max(0, pos - visible // 2), last_start)

    def _sidebar_row_at(self, pt):
        """Index of the file whose sidebar row is at canvas point `pt`, or None (headers too)."""
        if pt is None or pt[0] < MAIN_W:
            return None
        row = (pt[1] - (_SIDEBAR_TOP - 4)) // _SIDEBAR_ROW_H
        if not 0 <= row < self._sidebar_rows():
            return None
        rows = self._display_rows()
        n = self._sidebar_first() + row
        return rows[n][1] if n < len(rows) and rows[n][0] == "file" else None

    def _apply_scrolling(self):
        """Scroll the sidebar by any wheel / trackpad movement made over it."""
        while self._window.wheel_events:
            dy, pos = self._window.wheel_events.popleft()
            if pos is not None and pos[0] >= MAIN_W:
                self._wheel_accum -= dy * _SCROLL_ROWS_PER_NOTCH
        rows = int(self._wheel_accum)
        if rows:
            self._wheel_accum -= rows
            self._scroll_first = self._sidebar_first() + rows

    # --- Showing a screen and handling input -----------------------------------

    def _show(self, main, wait_ms):
        """
        Display `main` + sidebar and wait up to wait_ms for a key (window.KEY_NONE if none).

        Raises:
            WindowClosed: the user closed the window.
        """
        self._window.show(np.hstack([main, self._sidebar()]))
        key = self._window.poll(wait_ms)
        if self._window.closed:
            raise WindowClosed()
        self._apply_scrolling()
        return key

    def _new_screen(self):
        """Forget clicks and presses left over from the previous screen."""
        self._window.mouse_events.clear()
        self._armed = self._armed_at = self._armed_row = None
        self._seeking = False

    def _draw_buttons(self, img, buttons):
        for b in buttons:
            over = b.contains(self._window.mouse_pos)
            if self._armed == b.value:
                state = "pressed" if over else "normal"
            else:
                state = "hover" if over and self._armed is None else "normal"
            b.draw(img, state)

    def _handle_input(self, buttons, key, hotkeys=None, seek_bar=False):
        """
        Apply a key and the queued mouse events to `buttons`. `hotkeys` maps
        extra keys (with no button) to values. With `seek_bar`, a press on the
        seek bar starts a drag (self._seeking) and the release ends it.

        Returns:
            (value, pressed_at, other_clicks): the chosen button's value (or
            None), when it was pressed (perf_counter), and mouse presses that
            didn't land on a button or a clickable sidebar row (canvas pixels).

        Raises:
            JumpTo: a file in review_targets was clicked in the sidebar.
        """
        now = time.perf_counter()
        for b in buttons:
            if key in b.keys:
                return b.value, now, []
        if hotkeys and key in hotkeys:
            return hotkeys[key], now, []
        chosen = pressed_at = None
        other_clicks = []
        events = self._window.mouse_events
        while events:
            kind, pt = events.popleft()
            if seek_bar and (self._seeking or (kind == "down" and on_seek_bar(pt))):
                self._seeking = kind == "down"
                if kind == "up":
                    chosen = ("seek_end", seek_fraction(pt[0]))
                continue
            row = self._sidebar_row_at(pt)
            if row is not None and row in self.review_targets:
                # A sidebar file: clicked when released on the row it was pressed on.
                if kind == "down":
                    self._armed_row, self._armed = row, None
                elif self._armed_row == row:
                    self._armed_row = None
                    raise JumpTo(row)
                continue
            self._armed_row = None if kind == "up" else self._armed_row
            hit = next((b for b in buttons if b.contains(pt)), None)
            if kind == "down":
                if hit is None:
                    other_clicks.append(pt)
                self._armed = hit.value if hit else None
                self._armed_at = now
            elif kind == "up":
                if (chosen is None and hit is not None and self._armed == hit.value):
                    chosen, pressed_at = hit.value, self._armed_at
                self._armed = None
        return chosen, pressed_at, other_clicks

    def _interact(self, main, buttons, wait_ms, hotkeys=None, seek_bar=False):
        """Draw `buttons` over `main`, show it for up to wait_ms, and handle input."""
        img = main.copy()
        self._draw_buttons(img, buttons)
        return self._handle_input(buttons, self._show(img, wait_ms), hotkeys, seek_bar)

    def _wait_for_choice(self, main, buttons):
        """Show `main` with `buttons` until one is chosen; return its value."""
        self._new_screen()
        while True:
            value, _, _ = self._interact(main, buttons, 20)
            if value is not None:
                return value

    # --- Screens ---------------------------------------------------------------

    def show_progress(self, title, subtitle, fraction, preview=None, force=False,
                      cancellable=False):
        """
        Progress screen: a title, subtitle and progress bar over an optional
        dimmed preview frame, plus a Cancel button if `cancellable`. Redraws are
        throttled unless `force`; input is handled on every call.

        Args:
            preview: an image, or a zero-argument callable returning one (so the
                caller only builds it when a redraw actually happens).

        Returns:
            True if the user asked to cancel.
        """
        buttons = button_row([("Cancel", "cancel", (KEY_ESC,))], MAIN_H - 90) if cancellable else []
        now = time.perf_counter()
        if not force and now - self._last_progress_draw < _PROGRESS_REDRAW_S:
            key = self._window.poll(0)
            if self._window.closed:
                raise WindowClosed()
            value, _, _ = self._handle_input(buttons, key)
            self._apply_scrolling()
            return value == "cancel"
        self._last_progress_draw = now

        main = dimmed(preview() if callable(preview) else preview, _DIM_PROGRESS)
        cy = MAIN_H // 2
        put_centered(main, title, cy - 40, 0.9, WHITE, 2)
        put_centered(main, subtitle, cy - 5, 0.6, GREY, 1)
        bar_w, bar_h = 560, 22
        bx, by = (MAIN_W - bar_w) // 2, cy + 20
        fraction = min(max(fraction, 0.0), 1.0)
        cv2.rectangle(main, (bx, by), (bx + bar_w, by + bar_h), (70, 70, 70), -1)
        cv2.rectangle(main, (bx, by), (bx + int(bar_w * fraction), by + bar_h), WHITE, -1)
        cv2.rectangle(main, (bx, by), (bx + bar_w, by + bar_h), GREY, 1)
        put_centered(main, f"{fraction * 100:.0f}%", by + bar_h + 28, 0.55, WHITE, 1)
        value, _, _ = self._interact(main, buttons, 1)
        return value == "cancel"

    def show_status(self, title, lines=()):
        """A centred status message with no buttons (e.g. while a model loads); returns at once."""
        main = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
        y = MAIN_H // 2 - 15 * len(lines) - 20
        put_centered(main, title, y, 0.9, WHITE, 2)
        for text in lines:
            y += 34
            put_centered(main, text, y, 0.55, GREY, 1)
        self._show(main, 1)

    def show_frame(self, img, wait_ms, specs, label=None, hotkeys=None, seek=None):
        """
        One playback frame above a bar of buttons, shown for up to wait_ms.

        Args:
            specs: button specs for the bar, (text, value, keys).
            label: optional text at the left of the bar (e.g. the mode).
            hotkeys: optional {key: value} for keys with no button.
            seek: optional (fraction, markers, text) to show a seek bar above
                the buttons: the position (0-1), [(fraction, colour)] marks,
                and text shown at its right (e.g. the time).

        Returns:
            (value, pressed_at): the chosen button's value or None, and when it
            was pressed (perf_counter) — use that, not the release, for timing.
            While the seek bar is dragged, value is ("seek", fraction); when
            it's released, ("seek_end", fraction).
        """
        main, _, _, _ = frame_screen(img, bottom=BAR_H + (SEEK_H if seek else 0))
        if label:
            cv2.putText(main, label, (16, MAIN_H - BAR_H // 2 + 6), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
        if seek:
            fraction = seek_fraction(self._window.mouse_pos[0]) if self._seeking else seek[0]
            active = self._seeking or on_seek_bar(self._window.mouse_pos)
            draw_seek_bar(main, fraction, seek[1], seek[2], active)
        value, pressed_at, _ = self._interact(main, bar_buttons(specs), wait_ms, hotkeys,
                                              seek_bar=bool(seek))
        if value is None and self._seeking:
            value = ("seek", seek_fraction(self._window.mouse_pos[0]))
        return value, pressed_at

    def start_playback(self):
        """Call before a playback loop so clicks from the previous screen are ignored."""
        self._new_screen()

    def show_message(self, lines, specs, background=None):
        """
        A centered message over an optional dimmed background, with a row of
        buttons below it. `lines` is a list of (text, colour); the first is the
        title. Returns the chosen button's value.
        """
        main = dimmed(background, _DIM_MESSAGE)
        y = MAIN_H // 2 - 18 * len(lines) - 30
        for k, (text, color) in enumerate(lines):
            put_centered(main, text, y, 0.8 if k == 0 else 0.55, color, 2 if k == 0 else 1)
            y += 45 if k == 0 else 30
        return self._wait_for_choice(main, button_row(specs, y + 20))

    def ask_review(self, background, summary_lines, note=None, wrong_person=False):
        """
        The review prompt over the last frame: the detection summary and one
        button per option in REVIEW_OPTIONS, plus WRONG_PERSON_OPTION if
        `wrong_person`. Returns the chosen option's value.
        """
        options = list(REVIEW_OPTIONS)
        if wrong_person:
            options.insert(-2, WRONG_PERSON_OPTION)   # before Replay and Quit
        main = dimmed(background, _DIM_PROMPT)
        y = 100 if wrong_person else 110
        put_centered(main, "Was the detection successful?", y, 0.9, WHITE, 2)
        y += 40
        for text in summary_lines:
            put_centered(main, text, y, 0.55, GREY, 1)
            y += 26
        y += 16

        btn_w, gap = 560, 10
        buttons = []
        for key_label, text, value, keys in options:
            buttons.append(Button(((MAIN_W - btn_w) // 2, y, btn_w, BTN_H), text, value,
                                  keys, key_label=f"[{key_label}]"))
            y += BTN_H + gap
        if note:
            put_centered(main, note, y + 18, 0.55, ORANGE, 1)
        return self._wait_for_choice(main, buttons)

    def pick_endpoints(self, frame, reason=None, start=None, finish=None):
        """
        Let the user set the rope endpoints on `frame`: the START point (far
        endpoint, where the walk begins) and the FINISH point (near endpoint).

        A given `start` (e.g. where the subject was detected standing) is
        pre-placed, so usually only the finish is clicked; "Move start point"
        lets the user click a new one. A given `finish` is pre-placed too.
        Once both are placed: Confirm, Redo finish point, Move start point, or
        Cancel.

        Args:
            frame: the video's first frame (the endpoints are in its pixels).
            reason: optional line explaining why clicks are needed.

        Returns:
            (start, finish, start_moved) in frame pixels — start_moved is True
            if the user placed the start point themselves — or None if cancelled.
        """
        base, scale, ox, oy = frame_screen(frame, top=HEADER_H)
        fh, fw = frame.shape[:2]

        def to_screen(p):
            return (int(p[0] * scale + ox), int(p[1] * scale + oy))

        prompts = {
            "start": "Click the START point (where the walk begins)",
            "finish": "Click the FINISH point (end of the course)",
            None: "Check the line, then Confirm",
        }
        cancel = ("Cancel", "cancel", (KEY_ESC,))
        move = ("Move start point", "move", ())
        placing = "start" if start is None else ("finish" if finish is None else None)
        moved = False
        self._new_screen()
        while True:
            main = base.copy()
            if start is not None and finish is not None:
                cv2.line(main, to_screen(start), to_screen(finish), YELLOW, 2)
            for p, color, label in ((start, BLUE, "START"), (finish, RED, "FINISH")):
                if p is not None:
                    sp = to_screen(p)
                    cv2.circle(main, sp, 7, color, -1)
                    cv2.putText(main, label, (sp[0] + 10, sp[1] - 8), FONT, 0.5, color, 2, cv2.LINE_AA)
            y = 32
            if reason:
                cv2.putText(main, reason, (16, y), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
                y += 30
            cv2.putText(main, prompts[placing], (16, y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

            if placing == "start":
                specs = [("Keep start point", "keep", ()), cancel] if start is not None else [cancel]
            elif placing == "finish":
                specs = [move, cancel]
            else:
                specs = [move, ("Redo finish point", "redo", KEY_BACKSPACE),
                         ("Confirm", "confirm", KEY_ENTER), cancel]
            value, _, clicks = self._interact(main, bar_buttons(specs), 30)

            if value == "cancel":
                return None
            if value == "confirm":
                return start, finish, moved
            if value == "move":
                placing = "start"
            elif value == "keep":
                placing = "finish" if finish is None else None
            elif value == "redo":
                finish, placing = None, "finish"
            for cx, cy in clicks:
                fx, fy = (cx - ox) / scale, (cy - oy) / scale
                if placing is None or cx >= MAIN_W or not (0 <= fx < fw and 0 <= fy < fh):
                    continue
                point = (int(round(fx)), int(round(fy)))
                if placing == "start":
                    start, moved = point, True
                    placing = "finish" if finish is None else None
                else:
                    finish, placing = point, None

    def pick_person(self, frame, poses, reason):
        """
        Let the user click the walking subject among `poses` (people in `frame`),
        each drawn in its own colour with a number. Returns the index of the
        chosen pose, or None if cancelled.
        """
        img = frame.copy()
        fh, fw = frame.shape[:2]
        boxes = []
        for k, pose in enumerate(poses):
            color = PERSON_COLORS[k % len(PERSON_COLORS)]
            pose_common.draw_pose(img, pose, color=color, point_radius=5, line_thickness=3)
            xs = [lm.x * fw for lm in pose if lm is not None]
            ys = [lm.y * fh for lm in pose if lm is not None]
            pad = 0.15 * (max(ys) - min(ys)) + 10
            boxes.append((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
            cv2.putText(img, str(k + 1), (int(min(xs)), int(min(ys) - pad)), FONT, 1.0, color, 3, cv2.LINE_AA)

        base, scale, ox, oy = frame_screen(img, top=HEADER_H)
        cv2.putText(base, reason, (16, 32), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
        cv2.putText(base, "Click the person doing the walk test", (16, 62), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        buttons = bar_buttons([("Cancel", "cancel", (KEY_ESC,))])
        self._new_screen()
        while True:
            value, _, clicks = self._interact(base, buttons, 30)
            if value == "cancel":
                return None
            for cx, cy in clicks:
                fx, fy = (cx - ox) / scale, (cy - oy) / scale
                hits = [k for k, (x0, y0, x1, y1) in enumerate(boxes) if x0 <= fx <= x1 and y0 <= fy <= y1]
                if hits:
                    # Overlapping boxes: take the person whose centre is closest.
                    return min(hits, key=lambda k: abs((boxes[k][0] + boxes[k][2]) / 2 - fx))

    def close(self):
        self._window.close()
