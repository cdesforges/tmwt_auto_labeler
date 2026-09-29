"""
Tests for the window's input handling (tmwt/ui/base_window.py) and LabelerUI's
dialogs (tmwt/ui/labeler_ui.py), with a scripted stand-in for the real window:
no window or window process is ever started.
"""

import unittest
from collections import deque
from unittest import mock

import numpy as np

from tmwt.ui import base_window, sidebar as sb, top_bar, widgets
from tmwt.ui.base_window import CANVAS_H, CANVAS_W, BaseWindow
from tmwt.ui.events import JumpTo, QuitWithoutSaving, SaveAndQuit, WindowClosed
from tmwt.ui.labeler_ui import LabelerUI
from tmwt.ui.pickers import PickerScreens
from tmwt.ui.seek_bar import MARK, SCRUB, Marker, SeekBar, SeekState
from tmwt.ui.top_bar import TOPBAR_H
from tmwt.ui.widgets import GREEN, KEY_ESC, MAIN_H, MAIN_W, RED, TextButton, button_row
from tmwt.ui.window import KEY_NONE


class ScriptExhausted(Exception):
    """The test didn't script enough input (stops a screen that would wait forever)."""


class FakeWindow:
    """
    Same interface as window.Window. Each poll() takes the next step of
    `script`: a dict with any of key, events [("down"|"up", (x, y))], pos,
    wheel [(dy, (x, y))], close; or a callable(window) returning one (so a
    step can depend on what's been drawn).
    """

    def __init__(self, title, canvas_w, canvas_h):
        self.size = (canvas_w, canvas_h)
        self.shown = []
        self.script = deque()
        self.polls = []
        self.mouse_pos = None
        self.mouse_events = deque()
        self.wheel_events = deque()
        self._closed = False
        self.close_calls = 0

    def show(self, canvas):
        assert canvas.shape == (self.size[1], self.size[0], 3), canvas.shape
        self.shown.append(canvas)

    def poll(self, wait_ms):
        self.polls.append(wait_ms)
        if not self.script:
            raise ScriptExhausted()
        step = self.script.popleft()
        if callable(step):
            step = step(self)
        for kind, pt in step.get("events", ()):
            self.mouse_events.append((kind, pt))
            self.mouse_pos = pt
        if "pos" in step:
            self.mouse_pos = step["pos"]
        self.wheel_events.extend(step.get("wheel", ()))
        if step.get("close"):
            self._closed = True
        return step.get("key", KEY_NONE)

    @property
    def closed(self):
        return self._closed

    def close(self):
        self.close_calls += 1
        self._closed = True


def canvas_pt(main_pt):
    """A main-area point in canvas pixels."""
    return (main_pt[0], main_pt[1] + TOPBAR_H)


def centre(button, panel_y=TOPBAR_H):
    """The centre of a main-area button (or with panel_y=0, a top bar button) in canvas pixels."""
    return (button.x + button.w // 2, button.y + button.h // 2 + panel_y)


def click(pt):
    return [("down", pt), ("up", pt)]


def sidebar_row(n):
    """Canvas point inside the sidebar's visible row n."""
    return (MAIN_W + 60, TOPBAR_H + sb._TOP + n * sb._ROW_H + 10)


class WindowTestCase(unittest.TestCase):
    cls = BaseWindow

    def setUp(self):
        patcher = mock.patch.object(base_window, "Window", FakeWindow)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ui = self.cls([f"walk_{i}.mp4" for i in range(6)], heading="control_vids")
        self.win = self.ui._window
        self.buttons = button_row([("OK", "ok", (13,)), ("Cancel", "cancel", (KEY_ESC,))], 300)
        self.ok, self.cancel = self.buttons

    def handle(self, events, key=KEY_NONE, hotkeys=None, seek_markers=None, buttons=None):
        self.win.mouse_events.extend(events)
        if events:
            self.win.mouse_pos = events[-1][1]
        return self.ui._handle_input(self.buttons if buttons is None else buttons, key, hotkeys,
                                     seek_markers)


class ClickTest(WindowTestCase):
    def test_uses_the_fake_window(self):
        self.assertIsInstance(self.win, FakeWindow)
        self.assertEqual(self.win.size, (CANVAS_W, CANVAS_H))

    def test_press_and_release_on_a_button(self):
        with mock.patch.object(base_window.time, "perf_counter", return_value=42.0):
            value, pressed_at, others = self.handle(click(centre(self.ok)))
        self.assertEqual((value, pressed_at, others), ("ok", 42.0, []))

    def test_pressed_at_is_the_press_not_the_release(self):
        with mock.patch.object(base_window.time, "perf_counter", return_value=1.0):
            self.handle([("down", centre(self.ok))])
        with mock.patch.object(base_window.time, "perf_counter", return_value=5.0):
            value, pressed_at, _ = self.handle([("up", centre(self.ok))])
        self.assertEqual((value, pressed_at), ("ok", 1.0))

    def test_drag_off_cancels(self):
        value, _, _ = self.handle([("down", centre(self.ok)), ("up", canvas_pt((5, 5)))])
        self.assertIsNone(value)
        self.assertIsNone(self.ui._armed)

    def test_release_on_another_button_cancels(self):
        value, _, _ = self.handle([("down", centre(self.ok)), ("up", centre(self.cancel))])
        self.assertIsNone(value)

    def test_drag_back_re_arms(self):
        self.assertEqual(self.handle([("down", centre(self.ok))])[0], None)
        self.assertEqual(self.ui._armed, "ok")
        # while held: pressed over the button, normal when dragged off
        off = (5, 5)
        on = (self.ok.x + 3, self.ok.y + 3)
        self.assertEqual(widgets.button_state(self.ok, off, self.ui._armed), "normal")
        self.assertEqual(widgets.button_state(self.ok, on, self.ui._armed), "pressed")
        self.assertEqual(self.handle([("up", centre(self.ok))])[0], "ok")

    def test_release_without_a_press_does_nothing(self):
        self.assertIsNone(self.handle([("up", centre(self.ok))])[0])

    def test_edges_of_a_button(self):
        corner = canvas_pt((self.ok.x, self.ok.y))
        self.assertEqual(self.handle(click(corner))[0], "ok")
        past = canvas_pt((self.ok.x + self.ok.w, self.ok.y))    # right edge is exclusive
        self.assertNotEqual(self.handle(click(past))[0], "ok")

    def test_only_the_first_click_counts(self):
        value, _, _ = self.handle(click(centre(self.ok)) + click(centre(self.cancel)))
        self.assertEqual(value, "ok")

    def test_key_of_a_button(self):
        self.assertEqual(self.handle([], key=KEY_ESC)[0], "cancel")
        self.assertEqual(self.handle([], key=13)[0], "ok")
        self.assertIsNone(self.handle([], key=ord("q"))[0])
        self.assertIsNone(self.handle([], key=KEY_NONE)[0])

    def test_hotkeys(self):
        self.assertEqual(self.handle([], key=ord("m"), hotkeys={ord("m"): "menu"})[0], "menu")
        # a button's key wins over a hotkey
        self.assertEqual(self.handle([], key=KEY_ESC, hotkeys={KEY_ESC: "x"})[0], "cancel")

    def test_other_clicks_in_main_area_pixels(self):
        value, _, others = self.handle([("down", canvas_pt((300, 200))), ("up", canvas_pt((300, 200))),
                                        ("down", canvas_pt((0, 0)))])
        self.assertIsNone(value)
        self.assertEqual(others, [(300, 200), (0, 0)])

    def test_presses_outside_the_main_area_are_not_other_clicks(self):
        _, _, others = self.handle([("down", (600, 10)),                 # top bar, no button
                                    ("down", (MAIN_W + 50, TOPBAR_H + 5)),    # sidebar heading
                                    ("down", canvas_pt((5, MAIN_H)))])   # below the canvas
        self.assertEqual(others, [])

    def test_press_on_a_button_is_not_an_other_click(self):
        self.assertEqual(self.handle([("down", centre(self.ok))])[2], [])


class SidebarClickTest(WindowTestCase):
    def setUp(self):
        super().setUp()
        self.ui.review_targets = [2, 3]

    def test_click_on_a_review_target_jumps(self):
        with self.assertRaises(JumpTo) as caught:
            self.handle(click(sidebar_row(2)))
        self.assertEqual(caught.exception.index, 2)
        self.assertIsNone(self.ui._armed_row)

    def test_not_a_target(self):
        self.assertEqual(self.handle(click(sidebar_row(1))), (None, None, []))

    def test_released_elsewhere(self):
        self.handle([("down", sidebar_row(2)), ("up", canvas_pt((100, 100)))])
        self.assertIsNone(self.ui._armed_row)

    def test_released_on_another_target(self):
        self.handle([("down", sidebar_row(2)), ("up", sidebar_row(3))])     # no JumpTo

    def test_press_and_release_across_calls(self):
        self.handle([("down", sidebar_row(3))])
        self.assertEqual(self.ui._armed_row, 3)
        with self.assertRaises(JumpTo):
            self.handle([("up", sidebar_row(3))])

    def test_release_on_a_row_pressed_elsewhere(self):
        self.handle([("down", canvas_pt((100, 100))), ("up", sidebar_row(2))])   # no JumpTo

    # Pressing one sidebar row and releasing on another does nothing, and
    # leaves nothing armed for a later release.
    def test_release_on_another_target_disarms(self):
        self.handle([("down", sidebar_row(2)), ("up", sidebar_row(3))])
        self.handle([("down", canvas_pt((100, 100))), ("up", sidebar_row(2))])   # must not jump

    # A button press released over a sidebar row disarms the button.
    def test_button_press_released_on_a_row_disarms(self):
        self.handle([("down", centre(self.ok)), ("up", sidebar_row(2))])
        self.assertIsNone(self.ui._armed)

    def test_wheel_over_the_sidebar_scrolls_it(self):
        ui = self.cls([f"walk_{i}.mp4" for i in range(40)])
        ui._window.script.extend([{"wheel": [(-3, (MAIN_W + 50, 300))]},
                                  {"wheel": [(-3, (100, 300))]}])     # over the main area: ignored
        ui._poll(0)
        self.assertEqual(ui.sidebar._first_row(), 3)
        ui._poll(0)
        self.assertEqual(ui.sidebar._first_row(), 3)
        self.assertFalse(ui._window.wheel_events)


class TopBarClickTest(WindowTestCase):
    def test_save_button_saves_and_quits(self):
        self.ui.in_review = True
        with self.assertRaises(SaveAndQuit):
            self.handle(click(centre(self.ui.top_bar.save_button, 0)))

    def test_save_button_hidden_outside_a_review(self):
        self.assertEqual(self.handle(click(centre(self.ui.top_bar.save_button, 0)))[0], None)

    def test_x_outside_a_review_closes(self):
        with self.assertRaises(WindowClosed):
            self.handle(click(centre(self.ui.top_bar.close_button, 0)))

    def test_x_needs_a_release_on_it(self):
        self.handle([("down", centre(self.ui.top_bar.close_button, 0)), ("up", (600, 10))])

    def test_top_bar_value_not_returned(self):
        # The X's value never reaches the screen, even with its button hit.
        self.ui.in_review = True
        self.ui.dialogs_shown = 0
        self.win.script.append(lambda w: {"key": KEY_ESC})       # the quit dialog: Cancel
        value, _, _ = self.handle(click(centre(self.ui.top_bar.close_button, 0)))
        self.assertIsNone(value)
        self.assertEqual(self.ui.dialogs_shown, 1)

    def test_x_in_a_review_then_save(self):
        self.ui.in_review = True
        self.win.script.append({"key": 13})                     # Enter: "Save progress & quit"
        with self.assertRaises(SaveAndQuit):
            self.handle(click(centre(self.ui.top_bar.close_button, 0)))

    def test_x_in_a_review_then_discard(self):
        self.ui.in_review = True
        buttons = capture_buttons(self)
        self.win.script.extend([click_on(buttons, "discard"), click_on(buttons, "yes")])
        with self.assertRaises(QuitWithoutSaving):
            self.handle(click(centre(self.ui.top_bar.close_button, 0)))

    def test_x_in_a_review_discard_then_go_back(self):
        self.ui.in_review = True
        buttons = capture_buttons(self)
        self.win.script.extend([click_on(buttons, "discard"), {"key": KEY_ESC},    # "Go back"
                                {"key": KEY_ESC}])                                # then Cancel
        self.handle(click(centre(self.ui.top_bar.close_button, 0)))
        self.assertFalse(self.ui._in_dialog)


class ShowTest(WindowTestCase):
    def test_compose_size(self):
        main = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        self.assertEqual(self.ui._compose(main).shape, (CANVAS_H, CANVAS_W, 3))
        self.win.mouse_pos = centre(self.ui.top_bar.close_button, 0)    # with a tooltip
        self.assertEqual(self.ui._compose(main).shape, (CANVAS_H, CANVAS_W, 3))

    def test_main_area_placement(self):
        main = np.full((MAIN_H, MAIN_W, 3), 77, np.uint8)
        canvas = self.ui._compose(main)
        self.assertTrue((canvas[TOPBAR_H:, :MAIN_W] == 77).all())
        self.assertFalse((canvas[:TOPBAR_H, :MAIN_W] == 77).all())

    def test_show_returns_the_key(self):
        self.win.script.append({"key": ord("a")})
        self.assertEqual(self.ui._show(np.zeros((MAIN_H, MAIN_W, 3), np.uint8), 5), ord("a"))
        self.assertEqual(self.win.polls, [5])
        self.assertEqual(len(self.win.shown), 1)

    def test_closed_window_raises(self):
        self.win.script.append({"close": True})
        with self.assertRaises(WindowClosed):
            self.ui._show(np.zeros((MAIN_H, MAIN_W, 3), np.uint8), 1)

    def test_new_screen_forgets_old_input(self):
        self.handle([("down", centre(self.ok))])
        self.win.mouse_events.append(("up", centre(self.ok)))
        self.ui._new_screen()
        self.assertFalse(self.win.mouse_events)
        self.assertIsNone(self.ui._armed)

    def test_close(self):
        self.ui.close()
        self.assertEqual(self.win.close_calls, 1)


def capture_buttons(test):
    """Record every Button drawn (newest last) while the test runs."""
    drawn = []
    original = widgets.Button.draw

    def draw(button, img, state):
        drawn.append(button)
        return original(button, img, state)

    patcher = mock.patch.object(widgets.Button, "draw", draw)
    patcher.start()
    test.addCleanup(patcher.stop)
    return drawn


def click_on(drawn, value):
    """A script step clicking the most recently drawn main-area button with `value`."""
    def step(window):
        button = next(b for b in reversed(drawn) if b.value == value)
        return {"events": click(centre(button))}
    return step


class SeekDragTest(WindowTestCase):
    cls = LabelerUI

    def setUp(self):
        super().setUp()
        self.bar = SeekBar()
        self.times_state = SeekState(0.5, [Marker(0.25, GREEN, "start"), Marker(0.75, RED, None)], "")
        self.frame = np.full((240, 320, 3), 60, np.uint8)
        self.ui.start_playback()

    def show(self, step):
        self.win.script.append(step)
        return self.ui.show_frame(self.frame, 1, [("Confirm", "confirm", ())], seek=self.times_state)

    def track(self, fraction):
        return canvas_pt((self.bar.x_at(fraction), self.bar.Y))

    def tab(self, fraction):
        return canvas_pt((self.bar.x_at(fraction), self.bar.Y + (self.bar.TAB_TOP + self.bar.TAB_BOTTOM) // 2))

    def test_scrub(self):
        value, _ = self.show({"events": [("down", self.track(0.5))], "pos": self.track(0.6)})
        self.assertEqual(value[0], "seek")
        self.assertAlmostEqual(value[1], 0.6, places=2)
        self.assertEqual(self.ui._drag, (SCRUB, None))
        value, _ = self.show({"pos": self.track(0.9)})
        self.assertAlmostEqual(value[1], 0.9, places=2)
        value, _ = self.show({"events": [("up", self.track(0.7))]})
        self.assertEqual(value[0], "seek_end")
        self.assertAlmostEqual(value[1], 0.7, places=2)
        self.assertIsNone(self.ui._drag)

    def test_click_on_the_track(self):
        value, _ = self.show({"events": click(self.track(0.3))})
        self.assertEqual(value[0], "seek_end")
        self.assertAlmostEqual(value[1], 0.3, places=2)

    def test_drag_past_the_ends_is_clamped(self):
        self.show({"events": [("down", self.track(0.5))], "pos": canvas_pt((-50, 300))})
        value, _ = self.show({"events": [("up", canvas_pt((MAIN_W + 200, 300)))]})
        self.assertEqual(value, ("seek_end", 1.0))

    def test_drag_a_mark_by_its_tab(self):
        value, _ = self.show({"events": [("down", self.tab(0.25))], "pos": self.tab(0.4)})
        self.assertEqual(value[:2], ("mark_drag", "start"))
        self.assertAlmostEqual(value[2], 0.4, places=2)
        self.assertEqual(self.ui._drag, (MARK, "start"))
        value, _ = self.show({"events": [("up", self.tab(0.45))]})
        self.assertEqual(value[:2], ("mark_drop", "start"))
        self.assertAlmostEqual(value[2], 0.45, places=2)

    def test_fixed_mark_tab_does_not_drag(self):
        # Just below the track: a press there scrubs instead.
        value, _ = self.show({"events": click(self.tab(0.75))})
        self.assertEqual(value[0], "seek_end")

    def test_buttons_still_work(self):
        confirm = widgets.bar_buttons([("Confirm", "confirm", ())])[0]
        value, _ = self.show({"events": click(centre(confirm))})
        self.assertEqual(value, "confirm")

    def test_release_over_a_button_ends_the_drag_not_a_click(self):
        confirm = widgets.bar_buttons([("Confirm", "confirm", ())])[0]
        self.show({"events": [("down", self.track(0.5))]})
        value, _ = self.show({"events": [("up", centre(confirm))]})
        self.assertEqual(value[0], "seek_end")

    def test_no_seek_bar_no_drag(self):
        self.win.script.append({"events": click(self.track(0.5))})
        value, _ = self.ui.show_frame(self.frame, 1, [])
        self.assertIsNone(value)
        self.assertIsNone(self.ui._drag)


class DialogTest(WindowTestCase):
    cls = LabelerUI

    def setUp(self):
        super().setUp()
        self.drawn = capture_buttons(self)

    def test_mro(self):
        self.assertEqual(LabelerUI.__mro__[:3], (LabelerUI, PickerScreens, BaseWindow))

    def test_show_message(self):
        self.win.script.append(click_on(self.drawn, "b"))
        choice = self.ui.show_message([("Title", widgets.WHITE), ("line", widgets.GREY)],
                                      [("A", "a", ()), ("B", "b", ()), ("C", "c", ())])
        self.assertEqual(choice, "b")

    def test_show_message_waits_for_a_choice(self):
        self.win.script.extend([{}, {"events": [("down", canvas_pt((5, 5)))]},
                                {"events": [("up", canvas_pt((5, 5)))]}, click_on(self.drawn, "a")])
        self.assertEqual(self.ui.show_message([("T", widgets.WHITE)], [("A", "a", ())]), "a")
        self.assertEqual(len(self.win.polls), 4)

    def test_show_message_ignores_a_click_from_before(self):
        # A release left over from the previous screen doesn't choose anything.
        self.win.mouse_events.append(("up", (0, 0)))
        self.win.script.append({"key": ord("y")})
        self.assertEqual(self.ui.show_message([("T", widgets.WHITE)], [("Yes", "y", (ord("y"),))]), "y")

    def test_confirm(self):
        self.win.script.append(click_on(self.drawn, "yes"))
        self.assertTrue(self.ui.confirm("Delete?", ["It can't be undone."], "Delete"))
        self.win.script.append(click_on(self.drawn, "no"))
        self.assertFalse(self.ui.confirm("Delete?", [], "Delete"))

    def test_confirm_keys_choose_the_safe_option(self):
        for key in (13, 10, KEY_ESC):
            self.win.script.append({"key": key})
            self.assertFalse(self.ui.confirm("Delete?", [], "Delete"))

    def test_ask_menu(self):
        options = [("1", "Approve", "approve", (ord("1"),)),
                   ("2", "Change endpoints", "endpoints", (ord("2"),)),
                   ("Esc", "Skip", "skip", (KEY_ESC,))]
        self.win.script.append(click_on(self.drawn, "endpoints"))
        self.assertEqual(self.ui.ask_menu(None, "Review", ["a line"], options, note="note"), "endpoints")
        self.win.script.append({"key": ord("1")})
        self.assertEqual(self.ui.ask_menu(np.zeros((100, 100, 3), np.uint8), "Review", [], options),
                         "approve")
        menu = [b for b in self.drawn if b.value in ("approve", "endpoints", "skip")]
        self.assertTrue(menu and all(isinstance(b, widgets.KeyedButton) for b in menu))

    def test_many_menu_options_fit(self):
        options = [(str(i), f"Option {i}", i, (ord(str(i)),)) for i in range(1, 10)]
        self.win.script.append(click_on(self.drawn, 9))
        self.assertEqual(self.ui.ask_menu(None, "Menu", ["a", "b"], options), 9)
        last = next(b for b in reversed(self.drawn) if b.value == 9)
        self.assertLessEqual(last.y + last.h, MAIN_H)

    def test_closing_the_window_in_a_dialog(self):
        self.win.script.append({"close": True})
        with self.assertRaises(WindowClosed):
            self.ui.show_message([("T", widgets.WHITE)], [("A", "a", ())])

    def test_jump_from_a_dialog(self):
        self.ui.review_targets = {4}
        self.win.script.append({"events": click(sidebar_row(4))})
        with self.assertRaises(JumpTo):
            self.ui.show_message([("T", widgets.WHITE)], [("A", "a", ())])


if __name__ == "__main__":
    unittest.main()


class ButtonTooltipTest(WindowTestCase):
    """Hovering a button that has a tooltip draws it above the button."""

    def drawn_near(self, button):
        """Whether something light was drawn just above `button` (where the tooltip goes)."""
        canvas = self.win.shown[-1]
        y0 = button.y + TOPBAR_H - 30
        region = canvas[y0:button.y + TOPBAR_H - 2, button.x:button.x + 120]
        return (region > 200).any()

    def show(self, buttons, mouse):
        self.win.mouse_pos = mouse
        self.win.script.append({})
        main = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        self.ui._interact(main, buttons, 1)

    def test_confirm_shows_its_shortcut_on_hover(self):
        (confirm,) = button_row([widgets.CONFIRM], 300)
        self.show([confirm], None)
        self.assertFalse(self.drawn_near(confirm))
        self.show([confirm], centre(confirm))
        self.assertTrue(self.drawn_near(confirm))

    def test_button_without_a_tooltip_shows_none(self):
        (ok,) = button_row([("OK", "ok", ())], 300)
        self.show([ok], centre(ok))
        self.assertFalse(self.drawn_near(ok))
