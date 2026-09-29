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

Buttons are described by specs, (text, value, keys): the label, the value
returned when it is chosen, and the key codes that also choose it.
"""

import time

import cv2
import numpy as np

import pose_common
from window import Window

WINDOW = "TMWT Labeler"
MAIN_W, MAIN_H = 960, 720
SIDEBAR_W = 320
FONT = cv2.FONT_HERSHEY_SIMPLEX

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DIM = (90, 90, 90)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 150, 255)
RED = (60, 60, 255)
BLUE = (255, 0, 0)

# Key codes, as returned by Window.poll (cv2.waitKey style). Closing the window
# arrives as KEY_ESC.
KEY_ESC = 27
KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_SPACE = ord(" ")

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
    ("4", "Body not detected (skip this file)", "body", (ord("4"),)),
    ("R", "Replay", "replay", (ord("r"), ord("R"))),
    ("Esc", "Quit review (save the rest unreviewed)", "quit", (KEY_ESC,)),
]
# Offered on the review prompt only when more than one person was tracked.
WRONG_PERSON_OPTION = ("5", "Wrong person tracked (pick the walker)", "person", (ord("5"),))

# Colours for telling people apart on the "pick the walker" screen.
PERSON_COLORS = [(0, 255, 0), (255, 160, 0), (255, 0, 255), (0, 200, 255), (60, 60, 255)]

# Layout.
BTN_H = 44
BAR_H = 64            # bottom button bar on image screens
HEADER_H = 84         # instruction strip above the frame when picking endpoints
_BTN_GAP = 16
_BTN_MIN_W = 150

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


# --- Drawing helpers -----------------------------------------------------------

def _put_centered(img, text, y, scale, color, thickness=1):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = max(10, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _truncate(text, max_w, scale, thickness=1):
    """Shorten `text` with a trailing '...' until it fits in max_w pixels."""
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_w:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, thickness)[0][0] > max_w:
        text = text[:-1]
    return text + "..."


def _fit(img, w, h):
    """
    Scale `img` to fit inside w x h (keeping aspect) and centre it on black.

    Returns:
        (canvas, scale, x0, y0): image pixel (x, y) lands at canvas pixel
        (x * scale + x0, y * scale + y0).
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None:
        return canvas, 1.0, 0, 0
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas, s, x0, y0


def _dimmed(img, brightness):
    """`img` fitted to the main area and darkened, as a background for text."""
    return (_fit(img, MAIN_W, MAIN_H)[0] * brightness).astype(np.uint8)


def _frame_screen(img, top=0):
    """
    Main-area canvas with `img` fitted between a `top` strip and the bottom
    button bar. Returns (canvas, scale, x0, y0) like _fit, in canvas pixels.
    """
    main = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
    fitted, s, x0, y0 = _fit(img, MAIN_W, MAIN_H - top - BAR_H)
    main[top:MAIN_H - BAR_H] = fitted
    cv2.rectangle(main, (0, MAIN_H - BAR_H), (MAIN_W, MAIN_H), (20, 20, 20), -1)
    return main, s, x0, y0 + top


class Button:
    """
    A clickable button drawn with OpenCV, behaving like a standard UI button:
    it highlights on hover, looks pushed in (inset shadow, label shifted) while
    held, and only counts as clicked if the mouse is released over it. Dragging
    off before releasing cancels the click; dragging back on re-arms it.
    """

    # Fill / border colours per visual state.
    _STYLES = {
        "normal": ((48, 48, 48), (95, 95, 95)),
        "hover": ((66, 66, 66), (170, 170, 170)),
        "pressed": ((36, 36, 36), YELLOW),
    }
    # Inset shadow lines along the top and left edges when pressed, outermost first.
    _SHADOW = ((0, 0, 0), (6, 6, 6), (12, 12, 12), (18, 18, 18), (24, 24, 24), (30, 30, 30))

    def __init__(self, rect, text, value, keys=(), key_label=""):
        self.x, self.y, self.w, self.h = rect
        self.text = text
        self.value = value          # returned when chosen
        self.keys = keys            # key codes that also choose it
        self.key_label = key_label  # shortcut shown at the left; text is centred if empty

    def contains(self, pt):
        return (pt is not None and self.x <= pt[0] < self.x + self.w
                and self.y <= pt[1] < self.y + self.h)

    def draw(self, img, state):
        """Draw in `state`: "normal", "hover" or "pressed"."""
        fill, border = self._STYLES[state]
        x0, y0, x1, y1 = self.x, self.y, self.x + self.w - 1, self.y + self.h - 1
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        shift = 0
        if state == "pressed":
            for k, shade in enumerate(self._SHADOW):
                cv2.line(img, (x0 + k, y0 + k), (x1, y0 + k), shade, 1)
                cv2.line(img, (x0 + k, y0 + k), (x0 + k, y1), shade, 1)
            shift = 2
        cv2.rectangle(img, (x0, y0), (x1, y1), border, 1)

        base_y = y0 + self.h // 2 + 7 + shift
        if self.key_label:
            cv2.putText(img, self.key_label, (x0 + 16 + shift, base_y), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
            text_x = x0 + 80
        else:
            (tw, _), _ = cv2.getTextSize(self.text, FONT, 0.6, 1)
            text_x = x0 + (self.w - tw) // 2
        cv2.putText(img, self.text, (text_x + shift, base_y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


def button_row(specs, y):
    """Buttons for `specs` (text, value, keys), side by side and centred at height y."""
    widths = [max(_BTN_MIN_W, cv2.getTextSize(text, FONT, 0.6, 1)[0][0] + 48)
              for text, _, _ in specs]
    x = (MAIN_W - sum(widths) - _BTN_GAP * (len(specs) - 1)) // 2
    buttons = []
    for (text, value, keys), w in zip(specs, widths):
        buttons.append(Button((x, y, w, BTN_H), text, value, keys))
        x += w + _BTN_GAP
    return buttons


def _bar_buttons(specs):
    """Button row in the bottom bar of an image screen."""
    return button_row(specs, MAIN_H - BAR_H + (BAR_H - BTN_H) // 2)


class LabelerUI:
    """The batch window. All drawing, key polling and clicks go through here."""

    def __init__(self, names):
        self.names = list(names)
        self.states = [WAITING] * len(self.names)
        self.notes = [""] * len(self.names)
        self._active = None           # index of the highlighted file, or None
        self._scroll_first = None     # first sidebar row shown; None = follow the active file
        self._last_progress_draw = 0.0
        self._armed = None            # value of the button the mouse is pressed on
        self._armed_at = None         # when that press happened (perf_counter)
        self._wheel_accum = 0.0       # sidebar scroll not yet applied (fractional rows)
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

    def _sidebar(self):
        panel = np.full((MAIN_H, SIDEBAR_W, 3), 25, dtype=np.uint8)
        x0 = 15
        cv2.putText(panel, f"Videos ({len(self.names)})", (x0, 35), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        cv2.line(panel, (x0, 48), (SIDEBAR_W - x0, 48), DIM, 1)

        top, legend_h = _SIDEBAR_TOP, _SIDEBAR_LEGEND_H
        visible = self._sidebar_rows()
        first = self._sidebar_first()
        row_h = _SIDEBAR_ROW_H

        for row, i in enumerate(range(first, min(len(self.names), first + visible))):
            y = top + row * row_h
            if i == self.active:
                cv2.rectangle(panel, (5, y - 4), (SIDEBAR_W - 5, y + row_h - 8), (55, 55, 55), -1)
            name = _truncate(f"{i + 1}. {self.names[i]}", SIDEBAR_W - 2 * x0, 0.5)
            cv2.putText(panel, name, (x0, y + 14), FONT, 0.5, STATE_COLORS[self.states[i]], 1, cv2.LINE_AA)
            if self.notes[i]:
                note = _truncate(self.notes[i], SIDEBAR_W - 2 * x0 - 12, 0.4)
                cv2.putText(panel, note, (x0 + 12, y + 31), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        if len(self.names) > visible:
            # Scrollbar: the thumb's size and position show which part of the list is in view.
            track_top, track_h = top - 4, visible * row_h
            thumb_h = max(20, track_h * visible // len(self.names))
            thumb_y = track_top + (track_h - thumb_h) * first // (len(self.names) - visible)
            cv2.rectangle(panel, (SIDEBAR_W - 6, track_top), (SIDEBAR_W - 3, track_top + track_h), (45, 45, 45), -1)
            cv2.rectangle(panel, (SIDEBAR_W - 6, thumb_y), (SIDEBAR_W - 3, thumb_y + thumb_h), DIM, -1)

        y = MAIN_H - legend_h + 20
        cv2.line(panel, (x0, y - 15), (SIDEBAR_W - x0, y - 15), DIM, 1)
        x = x0
        for label, color in _LEGEND:
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            if x + tw + 16 > SIDEBAR_W - x0:
                x = x0
                y += 20
            cv2.circle(panel, (x + 4, y - 4), 4, color, -1)
            cv2.putText(panel, label, (x + 12, y), FONT, 0.38, GREY, 1, cv2.LINE_AA)
            x += tw + 26
        return panel

    def _sidebar_rows(self):
        """How many file rows fit in the sidebar."""
        return max(1, (MAIN_H - _SIDEBAR_TOP - _SIDEBAR_LEGEND_H) // _SIDEBAR_ROW_H)

    def _sidebar_first(self):
        """Index of the first file row shown: the user's scroll position, or centred on the active file."""
        visible = self._sidebar_rows()
        last_start = max(0, len(self.names) - visible)
        if self._scroll_first is not None:
            return min(max(0, self._scroll_first), last_start)
        if self._active is None:
            return 0
        return min(max(0, self._active - visible // 2), last_start)

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
        """Display `main` + sidebar and wait up to wait_ms for a key (window.KEY_NONE if none)."""
        self._window.show(np.hstack([main, self._sidebar()]))
        key = self._window.poll(wait_ms)
        self._apply_scrolling()
        return key

    def _new_screen(self):
        """Forget clicks and presses left over from the previous screen."""
        self._window.mouse_events.clear()
        self._armed = self._armed_at = None

    def _draw_buttons(self, img, buttons):
        for b in buttons:
            over = b.contains(self._window.mouse_pos)
            if self._armed == b.value:
                state = "pressed" if over else "normal"
            else:
                state = "hover" if over and self._armed is None else "normal"
            b.draw(img, state)

    def _handle_input(self, buttons, key):
        """
        Apply a key and the queued mouse events to `buttons`.

        Returns:
            (value, pressed_at, other_clicks): the chosen button's value (or
            None), when it was pressed (perf_counter), and mouse presses that
            didn't land on a button (canvas pixels).
        """
        now = time.perf_counter()
        for b in buttons:
            if key in b.keys:
                return b.value, now, []
        chosen = pressed_at = None
        other_clicks = []
        events = self._window.mouse_events
        while events:
            kind, pt = events.popleft()
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

    def _interact(self, main, buttons, wait_ms):
        """Draw `buttons` over `main`, show it for up to wait_ms, and handle input."""
        img = main.copy()
        self._draw_buttons(img, buttons)
        return self._handle_input(buttons, self._show(img, wait_ms))

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
            value, _, _ = self._handle_input(buttons, self._window.poll(0))
            self._apply_scrolling()
            return value == "cancel"
        self._last_progress_draw = now

        main = _dimmed(preview() if callable(preview) else preview, _DIM_PROGRESS)
        cy = MAIN_H // 2
        _put_centered(main, title, cy - 40, 0.9, WHITE, 2)
        _put_centered(main, subtitle, cy - 5, 0.6, GREY, 1)
        bar_w, bar_h = 560, 22
        bx, by = (MAIN_W - bar_w) // 2, cy + 20
        fraction = min(max(fraction, 0.0), 1.0)
        cv2.rectangle(main, (bx, by), (bx + bar_w, by + bar_h), (70, 70, 70), -1)
        cv2.rectangle(main, (bx, by), (bx + int(bar_w * fraction), by + bar_h), YELLOW, -1)
        cv2.rectangle(main, (bx, by), (bx + bar_w, by + bar_h), GREY, 1)
        _put_centered(main, f"{fraction * 100:.0f}%", by + bar_h + 28, 0.55, WHITE, 1)
        value, _, _ = self._interact(main, buttons, 1)
        return value == "cancel"

    def show_frame(self, img, wait_ms, specs, label=None):
        """
        One playback frame above a bar of buttons, shown for up to wait_ms.

        Args:
            specs: button specs for the bar, (text, value, keys).
            label: optional text at the left of the bar (e.g. the mode).

        Returns:
            (value, pressed_at): the chosen button's value or None, and when it
            was pressed (perf_counter) — use that, not the release, for timing.
        """
        main, _, _, _ = _frame_screen(img)
        if label:
            cv2.putText(main, label, (16, MAIN_H - BAR_H // 2 + 6), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
        value, pressed_at, _ = self._interact(main, _bar_buttons(specs), wait_ms)
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
        main = _dimmed(background, _DIM_MESSAGE)
        y = MAIN_H // 2 - 18 * len(lines) - 30
        for k, (text, color) in enumerate(lines):
            _put_centered(main, text, y, 0.8 if k == 0 else 0.55, color, 2 if k == 0 else 1)
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
        main = _dimmed(background, _DIM_PROMPT)
        y = 100 if wrong_person else 110
        _put_centered(main, "Was the detection successful?", y, 0.9, WHITE, 2)
        y += 40
        for text in summary_lines:
            _put_centered(main, text, y, 0.55, GREY, 1)
            y += 26
        y += 16

        btn_w, gap = 560, 10
        buttons = []
        for key_label, text, value, keys in options:
            buttons.append(Button(((MAIN_W - btn_w) // 2, y, btn_w, BTN_H), text, value,
                                  keys, key_label=f"[{key_label}]"))
            y += BTN_H + gap
        if note:
            _put_centered(main, note, y + 18, 0.55, ORANGE, 1)
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
        base, scale, ox, oy = _frame_screen(frame, top=HEADER_H)
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
            value, _, clicks = self._interact(main, _bar_buttons(specs), 30)

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

        base, scale, ox, oy = _frame_screen(img, top=HEADER_H)
        cv2.putText(base, reason, (16, 32), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
        cv2.putText(base, "Click the person doing the walk test", (16, 62), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        buttons = _bar_buttons([("Cancel", "cancel", (KEY_ESC,))])
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
