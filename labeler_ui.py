"""
The labeler's single window. Nothing else in the labeler opens a window.

Two regions:
  - Main area (left): analysis progress, real-time playback, endpoint picking
    and the review prompt (clickable buttons, or their keyboard shortcuts).
  - Sidebar (right): every video in the batch, colour-coded by state:
      white  = waiting
      yellow = being analysed / reviewed / saved
      green  = done (automatic timing found, or approved at review)
      orange = needs your input at review (e.g. endpoints must be clicked)
      red    = failed or rejected
      grey   = saved without review
"""

import time
from collections import deque

import cv2
import numpy as np

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

# Key codes returned by cv2.waitKey (masked to 8 bits).
KEY_NONE = 255
KEY_ESC = 27
KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)

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

# Review prompt: (key label, description, choice id), and the keys for each choice.
REVIEW_OPTIONS = [
    ("1", "Looks good", "approve"),
    ("2", "Rope endpoints inaccurate (re-click them)", "endpoints"),
    ("3", "Walk start/stop inaccurate (time it manually)", "timing"),
    ("4", "Body not detected (skip this file)", "body"),
    ("R", "Replay", "replay"),
    ("Esc", "Quit review (save the rest unreviewed)", "quit"),
]
_REVIEW_KEYS = {
    ord("1"): "approve", **{k: "approve" for k in KEY_ENTER},
    ord("2"): "endpoints",
    ord("3"): "timing",
    ord("4"): "body",
    ord("r"): "replay", ord("R"): "replay",
    KEY_ESC: "quit",
}

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

    def __init__(self, rect, key_label, text, value):
        self.x, self.y, self.w, self.h = rect
        self.key_label = key_label  # keyboard shortcut shown on the button
        self.text = text
        self.value = value          # returned when clicked

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
        cv2.putText(img, self.key_label, (x0 + 16 + shift, base_y), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(img, self.text, (x0 + 80 + shift, base_y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


class LabelerUI:
    """The batch window. All drawing, key polling and clicks go through here."""

    def __init__(self, names):
        self.names = list(names)
        self.states = [WAITING] * len(self.names)
        self.notes = [""] * len(self.names)
        self.active = None            # index of the highlighted file, or None
        self._last_progress_draw = 0.0
        self._mouse_pos = None        # latest pointer position, window pixels
        self._mouse_events = deque()  # ("down" | "up", (x, y)) since last consumed
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, MAIN_W + SIDEBAR_W, MAIN_H)
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        # OpenCV calls this from inside cv2.waitKey, so events are queued and
        # handled by whichever screen is showing once waitKey returns.
        self._mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_events.append(("down", (x, y)))
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_events.append(("up", (x, y)))

    # --- Sidebar ---------------------------------------------------------------

    def set_state(self, i, state, note=""):
        """Set file i's sidebar state (WAITING, WORKING, ...) and its note line."""
        self.states[i] = state
        self.notes[i] = note

    def _sidebar(self):
        panel = np.full((MAIN_H, SIDEBAR_W, 3), 25, dtype=np.uint8)
        x0 = 15
        cv2.putText(panel, f"Videos ({len(self.names)})", (x0, 35), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        cv2.line(panel, (x0, 48), (SIDEBAR_W - x0, 48), DIM, 1)

        row_h, top, legend_h = 42, 62, 70
        visible = max(1, (MAIN_H - top - legend_h) // row_h)
        # Keep the active file in view when the batch is longer than the list.
        first = 0
        if self.active is not None and len(self.names) > visible:
            first = min(max(0, self.active - visible // 2), len(self.names) - visible)

        for row, i in enumerate(range(first, min(len(self.names), first + visible))):
            y = top + row * row_h
            if i == self.active:
                cv2.rectangle(panel, (5, y - 4), (SIDEBAR_W - 5, y + row_h - 8), (55, 55, 55), -1)
            name = _truncate(f"{i + 1}. {self.names[i]}", SIDEBAR_W - 2 * x0, 0.5)
            cv2.putText(panel, name, (x0, y + 14), FONT, 0.5, STATE_COLORS[self.states[i]], 1, cv2.LINE_AA)
            if self.notes[i]:
                note = _truncate(self.notes[i], SIDEBAR_W - 2 * x0 - 12, 0.4)
                cv2.putText(panel, note, (x0 + 12, y + 31), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        if first > 0:
            cv2.putText(panel, "...", (SIDEBAR_W - 40, top - 2), FONT, 0.5, GREY, 1)
        if first + visible < len(self.names):
            cv2.putText(panel, "...", (SIDEBAR_W - 40, MAIN_H - legend_h), FONT, 0.5, GREY, 1)

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

    def _show(self, main, wait_ms):
        """Display `main` + sidebar and wait up to wait_ms for a key (KEY_NONE if none)."""
        cv2.imshow(WINDOW, np.hstack([main, self._sidebar()]))
        return cv2.waitKey(max(1, int(wait_ms))) & 0xFF

    def _wait_for_key(self, main):
        """Display `main` until any key is pressed; return it."""
        while True:
            key = self._show(main, 50)
            if key != KEY_NONE:
                return key

    # --- Screens ---------------------------------------------------------------

    def show_progress(self, title, subtitle, fraction, preview=None, force=False):
        """
        Progress screen: a title, subtitle and progress bar over an optional
        dimmed preview frame. Redraws are throttled unless `force`.

        Args:
            preview: an image, or a zero-argument callable returning one (so the
                caller only builds it when a redraw actually happens).

        Returns:
            The key pressed, or KEY_NONE.
        """
        now = time.perf_counter()
        if not force and now - self._last_progress_draw < _PROGRESS_REDRAW_S:
            return cv2.waitKey(1) & 0xFF
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
        _put_centered(main, "Esc = quit", MAIN_H - 20, 0.45, DIM, 1)
        return self._show(main, 1)

    def show_frame(self, img, wait_ms, header=None):
        """Show one playback frame, waiting up to wait_ms; returns the key pressed."""
        main = _fit(img, MAIN_W, MAIN_H)[0]
        if header:
            cv2.putText(main, header, (12, 24), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
        return self._show(main, wait_ms)

    def show_message(self, lines, background=None):
        """
        Centered message over an optional dimmed background, shown until a key
        is pressed. `lines` is a list of (text, colour); the first is the title.
        Returns the key.
        """
        main = _dimmed(background, _DIM_MESSAGE)
        y = MAIN_H // 2 - 18 * len(lines)
        for k, (text, color) in enumerate(lines):
            _put_centered(main, text, y, 0.8 if k == 0 else 0.55, color, 2 if k == 0 else 1)
            y += 45 if k == 0 else 30
        return self._wait_for_key(main)

    def ask_review(self, background, summary_lines, note=None):
        """
        The review prompt over the last frame: the detection summary and one
        button per option in REVIEW_OPTIONS. Options can be clicked or chosen
        with their keyboard shortcut. Returns the chosen option's id.
        """
        base = _dimmed(background, _DIM_PROMPT)
        y = 110
        _put_centered(base, "Was the detection successful?", y, 0.9, WHITE, 2)
        y += 40
        for text in summary_lines:
            _put_centered(base, text, y, 0.55, GREY, 1)
            y += 26
        y += 16

        btn_w, btn_h, gap = 560, 44, 10
        buttons = []
        for key_label, text, value in REVIEW_OPTIONS:
            buttons.append(Button(((MAIN_W - btn_w) // 2, y, btn_w, btn_h), f"[{key_label}]", text, value))
            y += btn_h + gap
        if note:
            _put_centered(base, note, y + 18, 0.55, ORANGE, 1)

        self._mouse_events.clear()   # ignore clicks made during playback
        armed = None                 # button the mouse was pressed on, if any
        while True:
            main = base.copy()
            for b in buttons:
                if b is armed and b.contains(self._mouse_pos):
                    state = "pressed"
                elif armed is None and b.contains(self._mouse_pos):
                    state = "hover"
                else:
                    state = "normal"
                b.draw(main, state)

            key = self._show(main, 20)
            if key in _REVIEW_KEYS:
                return _REVIEW_KEYS[key]
            while self._mouse_events:
                kind, pt = self._mouse_events.popleft()
                if kind == "down":
                    armed = next((b for b in buttons if b.contains(pt)), None)
                elif kind == "up":
                    if armed is not None and armed.contains(pt):
                        return armed.value
                    armed = None

    def pick_endpoints(self, frame, header, reason=None, previous=None):
        """
        Let the user click the rope endpoints on `frame`: first the far endpoint
        (start of the walk), then the near endpoint (finish line). Backspace or U
        undoes the last click, Enter confirms, Esc cancels.

        Args:
            frame: the video's first frame (the endpoints are in its pixels).
            header: line shown at the top, e.g. the file name.
            reason: optional line explaining why clicks are needed.
            previous: optional (far, near) currently in use, drawn faintly.

        Returns:
            ((far_x, far_y), (near_x, near_y)) in frame pixels, or None if cancelled.
        """
        base, scale, ox, oy = _fit(frame, MAIN_W, MAIN_H)
        fh, fw = frame.shape[:2]

        def to_screen(p):
            return (int(p[0] * scale + ox), int(p[1] * scale + oy))

        prompts = ["Click the FAR endpoint (start of the walk)",
                   "Click the NEAR endpoint (finish line)",
                   "Enter = confirm"]
        points = []
        self._mouse_events.clear()
        while True:
            main = base.copy()
            if previous is not None:
                cv2.line(main, to_screen(previous[0]), to_screen(previous[1]), DIM, 1)
                cv2.putText(main, "previous", to_screen(previous[0]), FONT, 0.45, DIM, 1, cv2.LINE_AA)
            if len(points) == 2:
                cv2.line(main, to_screen(points[0]), to_screen(points[1]), YELLOW, 2)
            for p, color in zip(points, (BLUE, RED)):
                cv2.circle(main, to_screen(p), 7, color, -1)

            cv2.rectangle(main, (0, 0), (MAIN_W, 100 if reason else 76), (0, 0, 0), -1)
            cv2.putText(main, header, (12, 24), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
            y = 52
            if reason:
                cv2.putText(main, reason, (12, y), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
                y += 24
            cv2.putText(main, prompts[len(points)], (12, y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
            _put_centered(main, "Backspace / U = undo      Esc = cancel", MAIN_H - 16, 0.45, GREY, 1)

            key = self._show(main, 30)
            while self._mouse_events:
                kind, (cx, cy) = self._mouse_events.popleft()
                fx, fy = (cx - ox) / scale, (cy - oy) / scale
                if (kind == "down" and len(points) < 2 and cx < MAIN_W
                        and 0 <= fx < fw and 0 <= fy < fh):
                    points.append((int(round(fx)), int(round(fy))))
            if key == KEY_ESC:
                return None
            if (key in KEY_BACKSPACE or key in (ord("u"), ord("U"))) and points:
                points.pop()
            if key in KEY_ENTER and len(points) == 2:
                return points[0], points[1]

    def close(self):
        cv2.destroyWindow(WINDOW)
