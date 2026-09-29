"""
LabelerUI: the labeler's window with all its screens. BaseWindow
(base_window.py) provides the window, panels and input handling, and
PickerScreens (pickers.py) the endpoint and person pickers; this adds the
progress and status screens, the playback frame (with the seek bar), and menus.
Nothing else in the labeler opens a window.

Menu options (ask_menu) are (shortcut label, text, value, keys); the review's
are built in session/review.py.
"""

import time

import cv2
import numpy as np

from tmwt.ui.base_window import BaseWindow
from tmwt.ui.pickers import PickerScreens
from tmwt.ui.seek_bar import SEEK_H
from tmwt.ui.widgets import (BAR_H, BTN_H, GREY, KEY_ESC, MAIN_H, MAIN_W, ORANGE, RED, WHITE, YELLOW,
                             KeyedButton, bar_buttons, button_row, dimmed, draw_badge, frame_screen,
                             put_centered)

# Minimum interval between progress redraws, so drawing never slows analysis.
_PROGRESS_REDRAW_S = 0.07
# Brightness of a background frame behind text.
_DIM_PROGRESS = 0.3
_DIM_PROMPT = 0.12


class LabelerUI(PickerScreens, BaseWindow):
    """The batch window with every screen the labeler uses (see module docs)."""

    def __init__(self, names, title="Videos", click_hint="click to review", legend=None,
                 heading=""):
        """Same arguments as BaseWindow."""
        super().__init__(names, title, click_hint, legend, heading)
        self._last_progress_draw = 0.0

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
            value, _, _ = self._handle_input(buttons, self._poll(0))
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

    def show_frame(self, img, wait_ms, specs, label=None, hotkeys=None, seek=None, alert=None,
                   notice=None):
        """
        One playback frame above a bar of buttons, shown for up to wait_ms.

        Args:
            specs: button specs for the bar, (text, value, keys).
            label: optional text in a badge at the top left (e.g. the mode).
            hotkeys: optional {key: value} for keys with no button.
            seek: optional seek_bar.SeekState, to show a seek bar above the
                buttons.
            alert: optional error text, in red, centred above the seek bar.
            notice: optional information in orange, in the same place (an
                alert takes its place).

        Returns:
            (value, pressed_at): the chosen button's value or None, and when it
            was pressed (perf_counter) — use that, not the release, for timing.
            While the seek bar is dragged, value is ("seek", fraction), and
            ("seek_end", fraction) when it's released; while a mark is dragged
            by its tab, ("mark_drag", id, fraction), then ("mark_drop", id,
            fraction).
        """
        bottom = BAR_H + (SEEK_H if seek else 0)
        main, _, _, _ = frame_screen(img, bottom=bottom)
        if label:
            draw_badge(main, label, (12, 12), YELLOW)
        if alert or notice:
            draw_badge(main, alert or notice, (MAIN_W // 2, MAIN_H - bottom - 46),
                       RED if alert else ORANGE, scale=0.6 if alert else 0.5,
                       thickness=2 if alert else 1, center=True)
        if seek:
            mouse = self._mouse(self.main)
            drag_fraction = self.seek_bar.fraction_at(mouse[0]) if self._drag and mouse else None
            self.seek_bar.draw(main, seek, hover=self.seek_bar.grab(mouse, seek.markers),
                               drag=self._drag, drag_fraction=drag_fraction)
        value, pressed_at, _ = self._interact(main, bar_buttons(specs), wait_ms, hotkeys,
                                              seek_markers=seek.markers if seek else None)
        if value is None:
            value = self._live_drag_value()
        return value, pressed_at

    def start_playback(self):
        """Call before a playback loop so clicks from the previous screen are ignored."""
        self._new_screen()

    def ask_menu(self, background, title, lines, options, note=None):
        """
        A menu over a dimmed frame: `title`, some grey `lines`, and one
        full-width button per option, (shortcut label, text, value, keys), with
        its shortcut at the left; `note` in orange below. Returns the chosen
        option's value.
        """
        main = dimmed(background, _DIM_PROMPT)
        y = 120 - 12 * max(0, len(options) - 5)
        put_centered(main, title, y, 0.9, WHITE, 2)
        y += 40
        for text in lines:
            put_centered(main, text, y, 0.55, GREY, 1)
            y += 26
        y += 16

        btn_w, gap = 560, 10
        buttons = []
        for key_label, text, value, keys in options:
            buttons.append(KeyedButton(((MAIN_W - btn_w) // 2, y, btn_w, BTN_H), text, value,
                                       keys, key_label=f"[{key_label}]"))
            y += BTN_H + gap
        if note:
            put_centered(main, note, y + 18, 0.55, ORANGE, 1)
        return self._wait_for_choice(main, buttons)
