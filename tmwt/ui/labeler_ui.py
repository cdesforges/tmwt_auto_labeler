"""
The labeler's single window. Nothing else in the labeler opens a window, and
every choice the user makes is a clickable button (most also have an optional
keyboard shortcut).

Screens are drawn with OpenCV into a fixed-size canvas and shown in a
resizable window (window.py), which scales the canvas to fit and reports mouse
positions in canvas pixels. The canvas is made of panels (panel.py):

    +--------------------------------------------------+
    |  top bar (top_bar.py): quit, save, title, logo    |
    +-------------------------------+------------------+
    |  main area (MAIN_W x MAIN_H): |  sidebar         |
    |  progress, playback, endpoint |  (sidebar.py):   |
    |  picking, the review prompt,  |  every video,    |
    |  messages                     |  colour-coded    |
    +-------------------------------+------------------+

Screens draw the main area and lay out its buttons in main-area pixels; this
class adds the top bar and sidebar, and converts mouse positions to the right
panel's pixels. Drawing building blocks (colours, buttons, icons, the seek bar)
are in widgets.py; playback controls are in player.py.

Leaving a screen other than through its own buttons raises an exception from
whatever screen is showing:
  - JumpTo: a file in `review_targets` was clicked in the sidebar.
  - UserQuit and its subclasses: the user asked to stop —
      WindowClosed      the window was closed (or the top bar's X clicked
                        outside a review);
      SaveAndQuit       "save progress and quit", from the top bar (save icon,
                        or X then "Save progress & quit") during a review;
      QuitWithoutSaving the top bar's X, then "Quit without saving".
The caller (review_session.py) decides what each means for the review.
"""

import time

import cv2
import numpy as np

from tmwt.pose import pose_common
from tmwt.ui import top_bar
from tmwt.ui.panel import Panel
from tmwt.ui.sidebar import DEFAULT_LEGEND, Sidebar, SIDEBAR_W
from tmwt.ui.top_bar import TOPBAR_H, TopBar
from tmwt.ui.seek_bar import MARK, SEEK_H, SeekBar
from tmwt.ui.widgets import (BAR_H, BTN_H, FONT, GREY, HEADER_H, KEY_BACKSPACE, KEY_ENTER,
                             KEY_ESC, MAIN_H, MAIN_W, ORANGE, RED, WHITE, YELLOW, BLUE,
                             IconButton, KeyedButton, bar_buttons, button_row, dimmed, draw_badge,
                             draw_buttons, draw_tooltip, frame_screen, put_centered)
from tmwt.ui.window import Window

WINDOW = "TMWT Labeler"
CANVAS_W, CANVAS_H = MAIN_W + SIDEBAR_W, TOPBAR_H + MAIN_H

# Review prompt options: (shortcut label, text, value, keys).
REVIEW_OPTIONS = [
    ("1", "Looks good", "approve", (ord("1"),) + KEY_ENTER),
    ("2", "Rope endpoints inaccurate (re-click them)", "endpoints", (ord("2"),)),
    ("3", "Walk start/stop inaccurate (time it manually)", "timing", (ord("3"),)),
    ("4", "Skip this file", "skip", (ord("4"),)),
    ("R", "Replay", "replay", (ord("r"), ord("R"))),
    ("F", "Finish review (save results)", "quit", (ord("f"), ord("F"))),
]
# Offered on the review prompt only when more than one person was tracked.
WRONG_PERSON_OPTION = ("5", "Wrong person tracked (pick the walker)", "person", (ord("5"),))

# Colours for telling people apart on the "pick the walker" screen.
PERSON_COLORS = [(0, 255, 0), (255, 160, 0), (255, 0, 255), (0, 200, 255), (60, 60, 255)]

# Minimum interval between progress redraws, so drawing never slows analysis.
_PROGRESS_REDRAW_S = 0.07
# Brightness of a background frame behind text.
_DIM_PROGRESS = 0.3
_DIM_MESSAGE = 0.25
_DIM_PROMPT = 0.12


class JumpTo(Exception):
    """A file in the sidebar was clicked; `index` is which one."""

    def __init__(self, index):
        super().__init__(index)
        self.index = index


class UserQuit(Exception):
    """Base class: the user asked to stop, from any screen."""


class WindowClosed(UserQuit):
    """The window was closed (or the top bar's X clicked outside a review)."""


class SaveAndQuit(UserQuit):
    """During a review: save the review progress and stop, to continue later."""


class QuitWithoutSaving(UserQuit):
    """During a review: stop and discard the review progress."""


class LabelerUI:
    """The batch window. All drawing, key polling and clicks go through here."""

    def __init__(self, names, title="Videos", click_hint="click to review", legend=None,
                 heading=""):
        """
        Args:
            names: file names listed in the sidebar.
            title: sidebar heading (shown with the file count).
            click_hint: shown beside the sidebar heading while files can be clicked.
            legend: [(label, colour)] under the list; defaults to the review states.
            heading: the top bar's title (e.g. the folder's name).
        """
        self.top_bar = TopBar(CANVAS_W, heading)
        self.main = Panel(0, TOPBAR_H, MAIN_W, MAIN_H)
        self.sidebar = Sidebar(MAIN_W, TOPBAR_H, MAIN_H, names, title, click_hint,
                               legend if legend is not None else DEFAULT_LEGEND)
        self.dialogs_shown = 0        # counts pop-up dialogs, so playback can re-sync after one
        self._last_progress_draw = 0.0
        self._armed = None            # value of the button the mouse is pressed on
        self._armed_at = None         # when that press happened (perf_counter)
        self._armed_row = None        # sidebar row the mouse was pressed on
        self.seek_bar = SeekBar()
        self._drag = None             # what's dragged on the seek bar (see SeekBar.grab), or None
        self._in_dialog = False
        self._last_main = None        # the main area last shown (a dialog's background)
        self._window = Window(WINDOW, CANVAS_W, CANVAS_H)

    # --- Sidebar and top bar state ---------------------------------------------

    @property
    def active(self):
        """Index of the highlighted file, or None."""
        return self.sidebar.active

    @active.setter
    def active(self, i):
        self.sidebar.active = i

    @property
    def review_targets(self):
        """Files that can be clicked in the sidebar (see JumpTo)."""
        return self.sidebar.review_targets

    @review_targets.setter
    def review_targets(self, targets):
        self.sidebar.review_targets = set(targets)

    @property
    def states(self):
        return self.sidebar.states

    @property
    def notes(self):
        return self.sidebar.notes

    def set_state(self, i, state, note=""):
        """Set file i's sidebar state (WAITING, WORKING, ...) and its note line."""
        self.sidebar.set_state(i, state, note)

    def mark_reviewed(self, i, outcome):
        """Record file i's review outcome (see sidebar.py); it moves to "Reviewed"."""
        self.sidebar.mark_reviewed(i, outcome)

    @property
    def in_review(self):
        """
        True while a review is open: the top bar shows the save button, and its
        X asks whether to save the review progress before quitting.
        """
        return self.top_bar.show_save

    @in_review.setter
    def in_review(self, value):
        self.top_bar.show_save = bool(value)

    # --- Showing a screen and handling input -----------------------------------

    def _mouse(self, panel):
        """The pointer in `panel`'s pixels (it may be outside the panel), or None."""
        return panel.to_local(self._window.mouse_pos)

    def _compose(self, main):
        """The whole canvas: the top bar over the main area `main` and the sidebar."""
        pos = self._window.mouse_pos
        bar = self.top_bar.render(self.top_bar.local_if_inside(pos), self._armed)
        side = self.sidebar.render(self.sidebar.local_if_inside(pos), self._armed_row)
        canvas = np.vstack([bar, np.hstack([main, side])])
        hovered = self.top_bar.hovered_button(self.top_bar.local_if_inside(pos))
        if hovered is not None and self._armed is None:
            draw_tooltip(canvas, hovered.text, (hovered.x, hovered.y + hovered.h))
        return canvas

    def _show(self, main, wait_ms):
        """
        Display `main` with the top bar and sidebar, and wait up to wait_ms for
        a key (window.KEY_NONE if none).

        Raises:
            WindowClosed: the user closed the window.
        """
        self._last_main = main
        self._window.show(self._compose(main))
        return self._poll(wait_ms)

    def _poll(self, wait_ms):
        """Wait up to wait_ms for a key, then apply sidebar scrolling."""
        key = self._window.poll(wait_ms)
        if self._window.closed:
            raise WindowClosed()
        while self._window.wheel_events:
            dy, pos = self._window.wheel_events.popleft()
            if self.sidebar.contains(pos):
                self.sidebar.scroll(dy)
        return key

    def _new_screen(self):
        """Forget clicks and presses left over from the previous screen."""
        self._window.mouse_events.clear()
        self._armed = self._armed_at = self._armed_row = None
        self._drag = None

    def _button_at(self, buttons, pt):
        """The button at canvas point `pt`: a top bar button, or one of the main area's `buttons`."""
        bar_pt = self.top_bar.local_if_inside(pt)
        if bar_pt is not None:
            return next((b for b in self.top_bar.buttons if b.contains(bar_pt)), None)
        main_pt = self.main.local_if_inside(pt)
        return next((b for b in buttons if b.contains(main_pt)), None)

    def _handle_input(self, buttons, key, hotkeys=None, seek_markers=None):
        """
        Apply a key and the queued mouse events to `buttons` (main-area
        pixels), the top bar and the sidebar. `hotkeys` maps extra keys (with no
        button) to values. With `seek_markers` (the seek bar's [Marker]; None if
        there's no seek bar), a press on the seek bar starts a drag (self._drag)
        — scrubbing, or moving a mark by its tab — and the release ends it with
        a value from _drag_value.

        Returns:
            (value, pressed_at, other_clicks): the chosen button's value (or
            None), when it was pressed (perf_counter), and presses in the main
            area that didn't land on a button (main-area pixels).

        Raises:
            JumpTo: a file in review_targets was clicked in the sidebar.
            UserQuit: the user quit from the top bar (see the module docs).
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
            main_pt = self.main.to_local(pt)
            if seek_markers is not None:
                if self._drag is None and kind == "down":
                    self._drag = self.seek_bar.grab(main_pt, seek_markers)
                    if self._drag is not None:
                        continue
                elif self._drag is not None:
                    if kind == "up":
                        chosen = self._drag_value(main_pt[0], done=True)
                        self._drag = None
                    continue
            row = self.sidebar.row_at(self.sidebar.local_if_inside(pt))
            if row is not None and row in self.review_targets:
                # A sidebar file: clicked when released on the row it was pressed on.
                if kind == "down":
                    self._armed_row, self._armed = row, None
                elif self._armed_row == row:
                    self._armed_row = None
                    raise JumpTo(row)
                continue
            self._armed_row = None if kind == "up" else self._armed_row
            hit = self._button_at(buttons, pt)
            if kind == "down":
                if hit is None and self.main.contains(pt):
                    other_clicks.append(main_pt)
                self._armed = hit.value if hit else None
                self._armed_at = now
            elif kind == "up":
                if chosen is None and hit is not None and self._armed == hit.value:
                    self._armed = None
                    if hit.value in (top_bar.CLOSE, top_bar.SAVE):
                        self._top_bar_clicked(hit.value)
                        continue
                    chosen, pressed_at = hit.value, self._armed_at
                self._armed = None
        return chosen, pressed_at, other_clicks

    def _top_bar_clicked(self, value):
        """
        The top bar's X or save button was clicked.

        Raises:
            UserQuit: unless the user cancelled the quit dialog.
        """
        if value == top_bar.SAVE:
            raise SaveAndQuit()
        if not self.in_review:
            raise WindowClosed()
        if self._in_dialog:
            return   # X on the quit dialog itself: nothing more to ask
        self._ask_quit()

    def _ask_quit(self):
        """
        "Quit the review?" dialog over the current screen. Returns if the user
        cancels (the screen underneath carries on).

        Raises:
            SaveAndQuit, QuitWithoutSaving: the user's choice.
        """
        self._in_dialog = True
        background = self._last_main
        try:
            while True:
                choice = self.show_message([
                    ("Quit the review?", WHITE),
                    ("Save your progress to continue this review later,", GREY),
                    ("or quit without saving to discard its decisions.", GREY),
                ], [("Save progress & quit", "save", KEY_ENTER),
                    ("Quit without saving", "discard", ()),
                    ("Cancel", "cancel", (KEY_ESC,))], background=background, dim=0.35)
                if choice != "discard" or self.confirm(
                        "Quit without saving?",
                        ["Every decision in this review will be discarded.",
                         "The videos stay analysed, but you'll review them from the start."],
                        "Discard and quit", background=background):
                    break
        finally:
            self._in_dialog = False
            self.dialogs_shown += 1
            self._new_screen()
        if choice == "save":
            raise SaveAndQuit()
        if choice == "discard":
            raise QuitWithoutSaving()

    def _drag_value(self, x, done):
        """
        The value for the seek bar drag in progress, with the pointer at main-area
        x: ("seek", fraction) while scrubbing, ("seek_end", fraction) on release;
        ("mark_drag", id, fraction) / ("mark_drop", id, fraction) for a mark.
        """
        fraction = self.seek_bar.fraction_at(x)
        if self._drag[0] == MARK:
            return ("mark_drop" if done else "mark_drag", self._drag[1], fraction)
        return ("seek_end" if done else "seek", fraction)

    def _interact(self, main, buttons, wait_ms, hotkeys=None, seek_markers=None):
        """Draw `buttons` over `main`, show it for up to wait_ms, and handle input."""
        img = main.copy()
        mouse = self._mouse(self.main)
        draw_buttons(img, buttons, mouse, self._armed)
        hovered = next((b for b in buttons if isinstance(b, IconButton) and b.contains(mouse)), None)
        if hovered is not None and self._armed is None and self._drag is None:
            draw_tooltip(img, hovered.text, (hovered.x, hovered.y), above=True)
        return self._handle_input(buttons, self._show(img, wait_ms), hotkeys, seek_markers)

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

    def show_frame(self, img, wait_ms, specs, label=None, hotkeys=None, seek=None, alert=None):
        """
        One playback frame above a bar of buttons, shown for up to wait_ms.

        Args:
            specs: button specs for the bar, (text, value, keys).
            label: optional text in a badge at the top left (e.g. the mode).
            hotkeys: optional {key: value} for keys with no button.
            seek: optional seek_bar.SeekState, to show a seek bar above the
                buttons.
            alert: optional error text, in red, centred above the seek bar.

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
        if alert:
            draw_badge(main, alert, (MAIN_W // 2, MAIN_H - bottom - 46), RED, scale=0.6,
                       thickness=2, center=True)
        if seek:
            mouse = self._mouse(self.main)
            drag_fraction = self.seek_bar.fraction_at(mouse[0]) if self._drag and mouse else None
            self.seek_bar.draw(main, seek, hover=self.seek_bar.grab(mouse, seek.markers),
                               drag=self._drag, drag_fraction=drag_fraction)
        value, pressed_at, _ = self._interact(main, bar_buttons(specs), wait_ms, hotkeys,
                                              seek_markers=seek.markers if seek else None)
        if value is None and self._drag is not None and self._window.mouse_pos is not None:
            value = self._drag_value(self._mouse(self.main)[0], done=False)
        return value, pressed_at

    def start_playback(self):
        """Call before a playback loop so clicks from the previous screen are ignored."""
        self._new_screen()

    def show_message(self, lines, specs, background=None, dim=_DIM_MESSAGE):
        """
        A centered message over an optional background dimmed to `dim`, with a
        row of buttons below it. `lines` is a list of (text, colour); the first
        is the title. Returns the chosen button's value.
        """
        main = dimmed(background, dim)
        y = MAIN_H // 2 - 18 * len(lines) - 30
        for k, (text, color) in enumerate(lines):
            put_centered(main, text, y, 0.8 if k == 0 else 0.55, color, 2 if k == 0 else 1)
            y += 45 if k == 0 else 30
        return self._wait_for_choice(main, button_row(specs, y + 20))

    def confirm(self, title, lines, yes, no="Go back", background=None):
        """
        "Are you sure?" screen before something that can't be undone: `title`
        (in orange), explanation `lines`, and the buttons `no` (Enter / Esc,
        the safe choice) and `yes`. Returns True if the user chose `yes`.
        """
        choice = self.show_message([(title, ORANGE)] + [(text, GREY) for text in lines],
                                   [(no, "no", KEY_ENTER + (KEY_ESC,)), (yes, "yes", ())],
                                   background=background, dim=0.2)
        return choice == "yes"

    def ask_review(self, background, summary_lines, note=None, wrong_person=False):
        """
        The review prompt over the last frame: the detection summary and one
        button per option in REVIEW_OPTIONS, plus WRONG_PERSON_OPTION if
        `wrong_person`. Returns the chosen option's value.
        """
        options = list(REVIEW_OPTIONS)
        if wrong_person:
            options.insert(-2, WRONG_PERSON_OPTION)   # before Replay and Finish
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
            buttons.append(KeyedButton(((MAIN_W - btn_w) // 2, y, btn_w, BTN_H), text, value,
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
                if placing is None or not (0 <= fx < fw and 0 <= fy < fh):
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
