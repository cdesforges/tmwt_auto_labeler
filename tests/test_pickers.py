"""
Tests for the picking screens (tmwt/ui/pickers.py): the EndpointPicking state
rules, and pick_endpoints / pick_person driven through a scripted stand-in for
the window (see test_base_window.FakeWindow).
"""

import unittest
from unittest import mock

import numpy as np

from tests.test_base_window import FakeWindow, ScriptExhausted, click
from tmwt.pose import pose_common as pc
from tmwt.ui import base_window, pickers
from tmwt.ui.labeler_ui import LabelerUI
from tmwt.ui.pickers import EndpointPicking
from tmwt.ui.top_bar import TOPBAR_H
from tmwt.ui.widgets import BAR_H, HEADER_H, KEY_ENTER, KEY_ESC, MAIN_H, MAIN_W, bar_buttons

# A frame exactly the size of the picture area, so frame pixels map 1:1
# (offset by the header strip and the top bar) to canvas pixels.
FRAME = np.zeros((MAIN_H - HEADER_H - BAR_H, MAIN_W, 3), np.uint8)


def canvas(frame_pt):
    return (frame_pt[0], frame_pt[1] + HEADER_H + TOPBAR_H)


def values(state):
    return [s[1] for s in state.specs()]


class EndpointPickingTest(unittest.TestCase):
    def test_nothing_placed_asks_for_the_start_then_the_finish(self):
        st = EndpointPicking()
        self.assertEqual(st.placing, "start")
        self.assertEqual(values(st), ["cancel"])
        st.place((10, 20))
        self.assertEqual((st.start, st.placing), ((10, 20), "finish"))
        st.place((30, 400))
        self.assertEqual((st.finish, st.placing), ((30, 400), None))
        self.assertEqual(st.result(), ((10, 20), (30, 400), True))

    def test_pre_placed_start_asks_for_the_finish(self):
        st = EndpointPicking(start=(5, 5), auto_start=(5, 5))
        self.assertEqual(st.placing, "finish")
        self.assertEqual(values(st), ["cancel"])                   # no Auto: it's already automatic
        self.assertFalse(st.start_moved)

    def test_both_placed_offers_confirm_with_enter(self):
        st = EndpointPicking((5, 5), (9, 9), auto_start=(5, 5))
        self.assertIsNone(st.placing)
        confirm = [s for s in st.specs() if s[1] == "confirm"][0]
        self.assertEqual(confirm.text, "Confirm")                  # the shortcut is shown on hover
        self.assertEqual(confirm.keys, KEY_ENTER)
        (button,) = bar_buttons([confirm])
        self.assertEqual(button.tooltip, "Confirm (Enter)")
        self.assertEqual(values(st), ["confirm", "cancel"])        # no move buttons: points are dragged

    def test_no_confirm_until_both_are_placed(self):
        self.assertNotIn("confirm", values(EndpointPicking()))
        self.assertNotIn("confirm", values(EndpointPicking(start=(5, 5))))

    def test_clicks_do_nothing_once_both_are_placed(self):
        st = EndpointPicking((5, 5), (9, 9))
        st.place((100, 100))
        self.assertEqual((st.start, st.finish), ((5, 5), (9, 9)))

    def test_dragging_moves_a_placed_point(self):
        st = EndpointPicking((5, 5), (9, 9))
        st.drag("finish", (50, 60), done=True)
        self.assertEqual((st.finish, st.placing), ((50, 60), None))

    def test_auto_start_point(self):
        st = EndpointPicking((5, 5), (9, 9), auto_start=(5, 5))
        st.drag("start", (40, 40), done=True)
        self.assertTrue(st.start_moved)
        self.assertIn("auto_start", values(st))
        st.press("auto_start")
        self.assertEqual((st.start, st.start_moved), ((5, 5), False))
        self.assertNotIn("auto_start", values(st))

    def test_auto_start_when_placing_the_start(self):
        st = EndpointPicking(None, None, auto_start=(5, 5))
        st.press("auto_start")
        self.assertEqual((st.start, st.placing), ((5, 5), "finish"))

    def test_auto_start_without_a_detected_spot_is_not_offered(self):
        st = EndpointPicking((5, 5), (9, 9))
        self.assertNotIn("auto_start", values(st))
        st.press("auto_start")                                    # ignored
        self.assertEqual(st.start, (5, 5))

    def test_dragging_a_point_back_onto_the_auto_spot_counts_as_automatic(self):
        st = EndpointPicking((5, 5), (9, 9), auto_start=(5, 5))
        st.drag("start", (6, 6), done=False)
        st.drag("start", (5, 5), done=True)
        self.assertFalse(st.start_moved)

    def test_live_drag_updates_the_point_before_the_drop(self):
        st = EndpointPicking((5, 5), (9, 9))
        st.drag("finish", (70, 80), done=False)
        self.assertEqual(st.finish, (70, 80))

    def test_dragging_the_other_point_keeps_placing(self):
        st = EndpointPicking((5, 5), None)
        st.drag("start", (8, 8), done=True)
        self.assertEqual((st.start, st.placing), ((8, 8), "finish"))

    def test_prompts(self):
        self.assertIn("START", EndpointPicking().prompt())
        self.assertIn("FINISH", EndpointPicking((1, 1)).prompt())
        self.assertIn("Drag", EndpointPicking((1, 1), (2, 2)).prompt())

    def test_widest_button_row_fits(self):
        st = EndpointPicking((5, 5), (9, 9), auto_start=(1, 1))
        buttons = bar_buttons(st.specs())
        self.assertGreaterEqual(buttons[0].x, 0)
        self.assertLessEqual(buttons[-1].x + buttons[-1].w, MAIN_W)


class PickEndpointsScreenTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(base_window, "Window", FakeWindow)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ui = LabelerUI(["walk.mp4"])
        self.win = self.ui._window

    def run_script(self, *steps, **kwargs):
        self.win.script.extend(steps)
        return self.ui.pick_endpoints(FRAME, **kwargs)

    def test_drag_the_start_point(self):
        start, finish = (100, 100), (300, 400)
        result = self.run_script(
            {},                                                        # first draw
            {"events": [("down", canvas(start))]},                    # grab the start point
            {"pos": canvas((150, 120))},                               # dragging (live)
            {"events": [("up", canvas((200, 150)))]},                  # drop it
            {"key": KEY_ENTER[0]},                                     # Confirm (Enter)
            start=start, finish=finish, auto_start=start)
        self.assertEqual(result, ((200, 150), finish, True))

    def test_press_near_a_point_grabs_it_not_a_click(self):
        start, finish = (100, 100), (300, 400)
        result = self.run_script(
            {}, {"events": click(canvas((105, 104)))},                 # within the grab radius
            {"key": KEY_ENTER[0]},
            start=start, finish=finish)
        self.assertEqual(result[:2], ((105, 104), finish))

    def test_click_places_the_finish(self):
        result = self.run_script(
            {}, {"events": click(canvas((320, 410)))}, {"key": KEY_ENTER[0]},
            start=(100, 100), auto_start=(100, 100))
        self.assertEqual(result, ((100, 100), (320, 410), False))

    def test_cancel(self):
        self.assertIsNone(self.run_script({}, {"key": KEY_ESC}, start=(1, 1), finish=(2, 2)))

    def test_auto_start_point_button(self):
        def press_auto(win):
            button = next(b for b in bar_buttons(state_specs) if b.value == "auto_start")
            pt = (button.x + button.w // 2, button.y + button.h // 2 + TOPBAR_H)
            return {"events": click(pt)}
        state_specs = EndpointPicking((50, 50), (300, 400), auto_start=(100, 100)).specs()
        result = self.run_script({}, press_auto, {"key": KEY_ENTER[0]},
                                 start=(50, 50), finish=(300, 400), auto_start=(100, 100))
        self.assertEqual(result, ((100, 100), (300, 400), False))

    def test_drag_is_kept_inside_the_frame(self):
        result = self.run_script(
            {}, {"events": [("down", canvas((100, 100)))]},
            {"events": [("up", (5, TOPBAR_H + 3))]},                   # dropped above the picture
            {"key": KEY_ENTER[0]},
            start=(100, 100), finish=(300, 400))
        self.assertEqual(result[0], (5, 0))

    def test_waits_for_input(self):
        with self.assertRaises(ScriptExhausted):
            self.run_script({}, {}, start=(1, 1), finish=(2, 2))


class PickPersonScreenTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(base_window, "Window", FakeWindow)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ui = LabelerUI(["walk.mp4"])
        self.win = self.ui._window

    def person(self, x):
        """A pose standing at frame x (normalized landmarks), about 200 px tall."""
        h, w = FRAME.shape[:2]
        pose = [None] * pc.NUM_LANDMARKS
        for idx, dy in ((0, -200), (23, -100), (24, -100), (27, 0), (28, 0)):
            pose[idx] = pc.Landmark(x / w, (400 + dy) / h)
        return pose

    def hip_colour(self, x):
        """Colour drawn at the hip of the person standing at frame x, in the last canvas shown."""
        cx, cy = canvas((x, 300))
        return tuple(int(v) for v in self.win.shown[-1][cy, cx])

    def test_click_selects_then_confirm_returns_them(self):
        poses = [self.person(200), self.person(600)]
        self.win.script.extend([{}, {"events": click(canvas((600, 300)))}, {"key": KEY_ENTER[0]}])
        self.assertEqual(self.ui.pick_person(FRAME, poses, "two people"), 1)

    def test_a_click_alone_does_not_finish(self):
        poses = [self.person(200), self.person(600)]
        self.win.script.extend([{}, {"events": click(canvas((600, 300)))}])
        with self.assertRaises(ScriptExhausted):              # still waiting for Confirm
            self.ui.pick_person(FRAME, poses, "two people")

    def test_no_confirm_before_anyone_is_chosen(self):
        poses = [self.person(200)]
        self.win.script.extend([{}, {"key": KEY_ENTER[0]}, {"key": KEY_ESC}])
        self.assertIsNone(self.ui.pick_person(FRAME, poses, "one person"))

    def test_clicking_someone_else_changes_the_choice(self):
        poses = [self.person(200), self.person(600)]
        self.win.script.extend([{}, {"events": click(canvas((600, 300)))},
                                {"events": click(canvas((200, 300)))}, {"key": KEY_ENTER[0]}])
        self.assertEqual(self.ui.pick_person(FRAME, poses, "two people"), 0)

    def test_current_subject_starts_chosen(self):
        poses = [self.person(200), self.person(600)]
        self.win.script.extend([{}, {"key": KEY_ENTER[0]}])
        self.assertEqual(self.ui.pick_person(FRAME, poses, "two people", selected=1), 1)

    def test_click_on_nobody_keeps_the_choice(self):
        poses = [self.person(200)]
        self.win.script.extend([{}, {"events": click(canvas((800, 50)))}, {"key": KEY_ENTER[0]}])
        self.assertEqual(self.ui.pick_person(FRAME, poses, "one person", selected=0), 0)

    def test_colours_idle_hover_chosen(self):
        poses = [self.person(200), self.person(600)]
        self.win.script.extend([
            {"pos": canvas((850, 50))},                          # pointer on nobody
            {"pos": canvas((600, 300))},                         # over person 2
            {"events": click(canvas((600, 300)))},               # choose person 2
            {"pos": canvas((850, 50))},                          # pointer away again
            {"key": KEY_ESC}])
        seen = []
        orig_show = self.win.show

        def show(canvas_img):
            orig_show(canvas_img)
            seen.append((self.hip_colour(200), self.hip_colour(600)))
        self.win.show = show
        self.ui.pick_person(FRAME, poses, "two people")
        self.assertEqual(seen[1], (pickers.PERSON_IDLE, pickers.PERSON_IDLE))
        self.assertEqual(seen[2], (pickers.PERSON_IDLE, pickers.PERSON_HOVER))
        self.assertEqual(seen[-1], (pickers.PERSON_IDLE, pickers.PERSON_CHOSEN))

    def test_no_red_for_people(self):
        # Red is kept for errors.
        for colour in (pickers.PERSON_IDLE, pickers.PERSON_HOVER):
            b, g, r = colour
            self.assertGreater(b, r)


class PersonHitTest(unittest.TestCase):
    def test_boxes_are_padded_bounds(self):
        pose = [None] * pc.NUM_LANDMARKS
        pose[0], pose[27] = pc.Landmark(0.5, 0.25), pc.Landmark(0.5, 0.75)
        (box,) = pickers.person_boxes([pose], (400, 200))       # 200 px tall -> pad 0.15 * 200 + 10 = 40 px
        self.assertEqual(box, (60.0, 60.0, 140.0, 340.0))

    def test_person_at(self):
        boxes = [(0, 0, 100, 100), (200, 0, 300, 100)]
        self.assertEqual(pickers.person_at(boxes, (50, 50)), 0)
        self.assertEqual(pickers.person_at(boxes, (250, 50)), 1)
        self.assertIsNone(pickers.person_at(boxes, (150, 50)))
        self.assertIsNone(pickers.person_at([], (0, 0)))

    def test_overlapping_boxes_pick_the_nearest_centre(self):
        boxes = [(0, 0, 100, 100), (60, 0, 200, 100)]           # centres 50 and 130
        self.assertEqual(pickers.person_at(boxes, (70, 50)), 0)
        self.assertEqual(pickers.person_at(boxes, (95, 50)), 1)


if __name__ == "__main__":
    unittest.main()
