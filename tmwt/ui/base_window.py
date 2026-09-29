"""
BaseWindow: the labeler's single window, with its panels and input handling.
Screens (labeler_ui.py, pickers.py) are built on it.

Screens are drawn with OpenCV into a fixed-size canvas and shown in a
resizable window (window.py), which scales the canvas to fit and reports mouse
positions in canvas pixels. The canvas is made of panels (panel.py):

    +--------------------------------------------------+
    |  top bar (top_bar.py): quit, save, title, logo    |
    +-------------------------------+------------------+
    |  main area (MAIN_W x MAIN_H): |  sidebar         |
    |  progress, playback, endpoint |  (sidebar.py):   |
    |  picking, menus, messages     |  every video,    |
    |                               |  colour-coded    |
    +-------------------------------+------------------+

A screen draws the main area and lays out its buttons in main-area pixels,
then calls _interact (draw, show, handle input) or _wait_for_choice; this class
adds the top bar and sidebar, converts mouse positions to the right panel's
pixels, handles the seek bar's drags, and raises events.py's exceptions when
the user leaves a screen through the sidebar or the top bar. Drawing building
blocks (colours, buttons, icons) are in widgets.py.

Every choice the user makes is a clickable button (most also have an optional
keyboard shortcut); a click counts when the mouse is released over the button
it was pressed on.
"""

import time

import numpy as np

from tmwt.ui import top_bar
from tmwt.ui.events import JumpTo, QuitWithoutSaving, SaveAndQuit, WindowClosed
from tmwt.ui.panel import Panel
from tmwt.ui.seek_bar import MARK, SeekBar
from tmwt.ui.sidebar import DEFAULT_LEGEND, SIDEBAR_W, Sidebar
from tmwt.ui.top_bar import TOPBAR_H, TopBar
from tmwt.ui.widgets import (GREY, KEY_ENTER, KEY_ESC, MAIN_H, MAIN_W, ORANGE, WHITE, IconButton,
                             button_row, dimmed, draw_buttons, draw_tooltip, put_centered)
from tmwt.ui.window import Window

WINDOW = "TMWT Labeler"
CANVAS_W, CANVAS_H = MAIN_W + SIDEBAR_W, TOPBAR_H + MAIN_H

# Brightness of a background frame behind a message.
DIM_MESSAGE = 0.25
# Drag kind for a screen's draggable points (see _handle_input's `handles`),
# and how close (main-area pixels) a press must be to grab one.
POINT = "point"
HANDLE_RADIUS = 14


class BaseWindow:
    """The window: panels, drawing, key polling and clicks (see module docs)."""

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

    def _handle_input(self, buttons, key, hotkeys=None, seek_markers=None, handles=None):
        """
        Apply a key and the queued mouse events to `buttons` (main-area
        pixels), the top bar and the sidebar. `hotkeys` maps extra keys (with no
        button) to values. A press can start a drag (self._drag), which the
        release ends with a value from _drag_value:
          - with `seek_markers` (the seek bar's [Marker]; None if there's no
            seek bar): scrubbing, or moving a mark by its tab;
          - with `handles` ({id: (x, y)} in main-area pixels): moving the
            point within HANDLE_RADIUS of the press (the nearest).

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
            if self._drag is not None:
                if kind == "up":        # the release ends the drag, wherever it is
                    chosen = self._drag_value(main_pt, done=True)
                    self._drag = None
                continue
            if kind == "down":
                self._drag = self._grab(main_pt, seek_markers, handles)
                if self._drag is not None:
                    self._armed = self._armed_row = None
                    continue
            row = self.sidebar.row_at(self.sidebar.local_if_inside(pt))
            if row is not None and row in self.review_targets:
                # A sidebar file: clicked when released on the row it was pressed on.
                if kind == "down":
                    self._armed_row, self._armed = row, None
                else:
                    pressed, self._armed_row, self._armed = self._armed_row, None, None
                    if pressed == row:
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

    def _grab(self, pt, seek_markers, handles):
        """What a press at main-area `pt` starts dragging (see _handle_input), or None."""
        if seek_markers is not None:
            grab = self.seek_bar.grab(pt, seek_markers)
            if grab is not None:
                return grab
        if handles and pt is not None:
            near = [(float(np.hypot(pt[0] - x, pt[1] - y)), hid) for hid, (x, y) in handles.items()]
            dist, hid = min(near)
            if dist <= HANDLE_RADIUS:
                return (POINT, hid)
        return None

    def _drag_value(self, pt, done):
        """
        The value for the drag in progress, with the pointer at main-area `pt`:
        ("seek", fraction) while scrubbing and ("seek_end", fraction) on
        release; ("mark_drag" / "mark_drop", id, fraction) for a seek-bar mark;
        ("point_drag" / "point_drop", id, (x, y)) for a handle.
        """
        kind, drag_id = self._drag
        if kind == POINT:
            return ("point_drop" if done else "point_drag", drag_id, tuple(pt))
        fraction = self.seek_bar.fraction_at(pt[0])
        if kind == MARK:
            return ("mark_drop" if done else "mark_drag", drag_id, fraction)
        return ("seek_end" if done else "seek", fraction)

    def _live_drag_value(self):
        """The value for the drag in progress at the pointer now (None if not dragging)."""
        if self._drag is None or self._window.mouse_pos is None:
            return None
        return self._drag_value(self._mouse(self.main), done=False)

    def _interact(self, main, buttons, wait_ms, hotkeys=None, seek_markers=None, handles=None):
        """Draw `buttons` over `main`, show it for up to wait_ms, and handle input."""
        img = main.copy()
        mouse = self._mouse(self.main)
        draw_buttons(img, buttons, mouse, self._armed)
        hovered = next((b for b in buttons if isinstance(b, IconButton) and b.contains(mouse)), None)
        if hovered is not None and self._armed is None and self._drag is None:
            draw_tooltip(img, hovered.text, (hovered.x, hovered.y), above=True)
        return self._handle_input(buttons, self._show(img, wait_ms), hotkeys, seek_markers, handles)

    def _wait_for_choice(self, main, buttons):
        """Show `main` with `buttons` until one is chosen; return its value."""
        self._new_screen()
        while True:
            value, _, _ = self._interact(main, buttons, 20)
            if value is not None:
                return value

    # --- Messages (used by the quit dialog and every screen family) ------------

    def show_message(self, lines, specs, background=None, dim=DIM_MESSAGE):
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

    def close(self):
        self._window.close()
