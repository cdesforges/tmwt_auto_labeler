"""
Batch viewer window for label.py.

One OpenCV window with two regions:
  - Main area (left): analysis progress, real-time playback and the review prompt.
  - Sidebar (right): every video in the batch, colour-coded by status —
      white  = waiting
      yellow = being analysed / reviewed / saved
      green  = done (auto timing found, or approved at review)
      orange = needs your input at review (e.g. no ArUco marker to place endpoints)
      red    = failed or rejected
"""

import time

import cv2
import numpy as np

WINDOW = "TMWT Labeler"
MAIN_W, MAIN_H = 960, 720
SIDEBAR_W = 320

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DIM = (90, 90, 90)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 150, 255)
RED = (60, 60, 255)

STATUS_COLORS = {
    "pending": WHITE,
    "analysing": YELLOW,
    "reviewing": YELLOW,
    "saving": YELLOW,
    "ok": GREEN,
    "approved": GREEN,
    "needs_input": ORANGE,
    "failed": RED,
    "rejected": RED,
    "unreviewed": GREY,
}

# Review prompt options: (key label, description, choice id).
REVIEW_OPTIONS = [
    ("1", "Looks good", "approve"),
    ("2", "Rope endpoints inaccurate (re-click them)", "endpoints"),
    ("3", "Walk start/stop inaccurate (time it manually)", "timing"),
    ("4", "Body not detected (skip this file)", "body"),
    ("R", "Replay", "replay"),
    ("Esc", "Quit review (save the rest unreviewed)", "quit"),
]
_REVIEW_KEYS = {
    ord("1"): "approve", 13: "approve", 10: "approve",
    ord("2"): "endpoints",
    ord("3"): "timing",
    ord("4"): "body",
    ord("r"): "replay", ord("R"): "replay",
    27: "quit",
}

# Minimum interval between progress redraws, so drawing never slows analysis.
_PROGRESS_REDRAW_S = 0.07

FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def fit_to(img, w, h):
    """Scale `img` to fit inside w x h (keeping aspect) and centre it on black."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None:
        return canvas
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class LabelerUI:
    """The single batch window. All drawing and key polling goes through here."""

    def __init__(self, names):
        self.names = list(names)
        self.status = ["pending"] * len(self.names)
        self.notes = [""] * len(self.names)
        self.active = None
        self._last_progress_draw = 0.0
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, MAIN_W + SIDEBAR_W, MAIN_H)

    # --- Sidebar -------------------------------------------------------------

    def set_status(self, i, status, note=""):
        self.status[i] = status
        self.notes[i] = note

    def _sidebar(self):
        panel = np.full((MAIN_H, SIDEBAR_W, 3), 25, dtype=np.uint8)
        x0 = 15
        cv2.putText(panel, f"Videos ({len(self.names)})", (x0, 35), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        cv2.line(panel, (x0, 48), (SIDEBAR_W - x0, 48), DIM, 1)

        row_h = 42
        top = 62
        legend_h = 70
        visible = max(1, (MAIN_H - top - legend_h) // row_h)
        # Keep the active file in view when the batch is longer than the list.
        first = 0
        if self.active is not None and len(self.names) > visible:
            first = min(max(0, self.active - visible // 2), len(self.names) - visible)

        for row, i in enumerate(range(first, min(len(self.names), first + visible))):
            y = top + row * row_h
            color = STATUS_COLORS.get(self.status[i], WHITE)
            if i == self.active:
                cv2.rectangle(panel, (5, y - 4), (SIDEBAR_W - 5, y + row_h - 8), (55, 55, 55), -1)
            name = _truncate(f"{i + 1}. {self.names[i]}", SIDEBAR_W - 2 * x0, 0.5)
            cv2.putText(panel, name, (x0, y + 14), FONT, 0.5, color, 1, cv2.LINE_AA)
            if self.notes[i]:
                note = _truncate(self.notes[i], SIDEBAR_W - 2 * x0 - 12, 0.4)
                cv2.putText(panel, note, (x0 + 12, y + 31), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        if first > 0:
            cv2.putText(panel, "...", (SIDEBAR_W - 40, top - 2), FONT, 0.5, GREY, 1)
        if first + visible < len(self.names):
            cv2.putText(panel, "...", (SIDEBAR_W - 40, MAIN_H - legend_h), FONT, 0.5, GREY, 1)

        # Legend
        y = MAIN_H - legend_h + 20
        cv2.line(panel, (x0, y - 15), (SIDEBAR_W - x0, y - 15), DIM, 1)
        entries = [("waiting", WHITE), ("working", YELLOW), ("done", GREEN),
                   ("needs input", ORANGE), ("failed", RED)]
        x = x0
        for label, color in entries:
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.38, 1)
            if x + tw + 16 > SIDEBAR_W - x0:
                x = x0
                y += 20
            cv2.circle(panel, (x + 4, y - 4), 4, color, -1)
            cv2.putText(panel, label, (x + 12, y), FONT, 0.38, GREY, 1, cv2.LINE_AA)
            x += tw + 26
        return panel

    def _show(self, main, wait_ms):
        cv2.imshow(WINDOW, np.hstack([main, self._sidebar()]))
        return cv2.waitKey(max(1, int(wait_ms))) & 0xFF

    # --- Screens -------------------------------------------------------------

    def show_progress(self, title, subtitle, fraction, preview=None, force=False):
        """
        Progress screen: optional dimmed preview frame behind a title, a subtitle
        and a progress bar. Redraws are throttled; returns the key pressed (255 = none).

        `preview` may be an image or a zero-argument callable returning one, so
        callers can skip building it on frames that aren't redrawn.
        """
        now = time.perf_counter()
        if not force and now - self._last_progress_draw < _PROGRESS_REDRAW_S:
            return cv2.waitKey(1) & 0xFF
        self._last_progress_draw = now

        if callable(preview):
            preview = preview()
        main = fit_to(preview, MAIN_W, MAIN_H)
        main = (main * 0.3).astype(np.uint8)
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
        """Show one playback frame in the main area; returns the key pressed."""
        main = fit_to(img, MAIN_W, MAIN_H)
        if header:
            cv2.putText(main, header, (12, 24), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
        return self._show(main, wait_ms)

    def show_message(self, lines, background=None, wait=True):
        """
        Centered message over an optional dimmed background. If `wait`, blocks
        until a key is pressed and returns it.
        """
        main = (fit_to(background, MAIN_W, MAIN_H) * 0.25).astype(np.uint8)
        y = MAIN_H // 2 - 18 * len(lines)
        for k, (text, color) in enumerate(lines):
            _put_centered(main, text, y, 0.8 if k == 0 else 0.55, color, 2 if k == 0 else 1)
            y += 45 if k == 0 else 30
        if not wait:
            return self._show(main, 1)
        while True:
            key = self._show(main, 50)
            if key != 255:
                return key

    def ask_review(self, background, summary_lines, note=None):
        """
        Pause on the review prompt over the last frame. Returns one of the
        choice ids in REVIEW_OPTIONS.
        """
        main = (fit_to(background, MAIN_W, MAIN_H) * 0.12).astype(np.uint8)
        y = 120
        _put_centered(main, "Was the detection successful?", y, 0.9, WHITE, 2)
        y += 40
        for text in summary_lines:
            _put_centered(main, text, y, 0.55, GREY, 1)
            y += 26
        y += 25
        x = MAIN_W // 2 - 250
        for key_label, text, _ in REVIEW_OPTIONS:
            cv2.putText(main, f"[{key_label}]", (x, y), FONT, 0.65, YELLOW, 2, cv2.LINE_AA)
            cv2.putText(main, text, (x + 80, y), FONT, 0.65, WHITE, 1, cv2.LINE_AA)
            y += 42
        if note:
            _put_centered(main, note, y + 20, 0.55, ORANGE, 1)
        while True:
            choice = _REVIEW_KEYS.get(self._show(main, 50))
            if choice:
                return choice

    def close(self):
        cv2.destroyWindow(WINDOW)
