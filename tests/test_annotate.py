"""Tests for frame annotation: the info panel and annotated frames (tmwt/ui/annotate.py)."""

import unittest

import numpy as np

from tmwt.core.job import FrameResult
from tmwt.pose import pose_common as pc
from tmwt.ui import annotate
from tmwt.ui.annotate import PANEL_W, draw_info_panel, render_frame

H, W = 480, 640


def has_color(img, color):
    return bool((img == np.array(color, np.uint8)).all(axis=-1).any())


def panel(time_s, walk_start=1.0, walk_end=3.0, **kwargs):
    return draw_info_panel(H, time_s, 10, 0.5, walk_start, walk_end, **kwargs)


class InfoPanelStatusTest(unittest.TestCase):
    def test_size(self):
        self.assertEqual(panel(0.0).shape, (H, PANEL_W, 3))

    def test_the_three_states_differ(self):
        waiting, walking, finished = panel(0.5), panel(2.0), panel(3.5)
        self.assertFalse(np.array_equal(waiting, walking))
        self.assertFalse(np.array_equal(walking, finished))
        self.assertFalse(np.array_equal(waiting, finished))

    def test_status_colours(self):
        waiting, walking, finished = panel(0.5), panel(2.0), panel(3.5)
        self.assertFalse(has_color(waiting, annotate._YELLOW))
        self.assertFalse(has_color(waiting, annotate._BRIGHT_GREEN))
        self.assertTrue(has_color(walking, annotate._YELLOW))
        self.assertFalse(has_color(walking, annotate._GREEN))
        self.assertTrue(has_color(finished, annotate._BRIGHT_GREEN))
        self.assertFalse(has_color(finished, annotate._YELLOW))

    def test_boundaries(self):
        self.assertTrue(np.array_equal(panel(1.0), panel(1.0, walk_end=None)))  # walking at walk_start
        self.assertTrue(has_color(panel(1.0), annotate._YELLOW))
        self.assertTrue(has_color(panel(3.0), annotate._BRIGHT_GREEN))            # finished at walk_end

    def test_missing_timing(self):
        # No start: waiting whatever the time; no end: walking forever.
        self.assertFalse(has_color(panel(99.0, walk_start=None), annotate._YELLOW))
        self.assertFalse(has_color(panel(99.0, walk_start=None, walk_end=None), annotate._YELLOW))
        self.assertTrue(has_color(panel(99.0, walk_end=None), annotate._YELLOW))

    def test_elapsed_time_changes(self):
        self.assertFalse(np.array_equal(panel(1.5), panel(2.5)))

    def test_walk_time_is_fixed_once_finished(self):
        a, b = panel(3.5), panel(9.0)
        # Only the frame / time lines differ, not the result block.
        self.assertTrue(np.array_equal(a[:230], b[:230]))

    def test_waiting_lines(self):
        a = panel(0.0)
        b = panel(0.0, waiting_lines=("Click the endpoints", "to start"))
        self.assertFalse(np.array_equal(a, b))

    def test_zero_duration_walk(self):
        self.assertEqual(panel(2.0, walk_start=2.0, walk_end=2.0).shape, (H, PANEL_W, 3))


class InfoPanelOptionalLinesTest(unittest.TestCase):
    def test_model_strength_only_when_given(self):
        base = panel(0.0)
        self.assertTrue(np.array_equal(base, panel(0.0, model_strength=None)))
        self.assertTrue(np.array_equal(base, panel(0.0, model_strength="")))
        self.assertFalse(np.array_equal(base, panel(0.0, model_strength="rtmlib balanced")))

    def test_t_along_only_when_given(self):
        a = draw_info_panel(H, 0.0, 10, None, None, None)
        b = draw_info_panel(H, 0.0, 10, 0.25, None, None)
        self.assertFalse(np.array_equal(a, b))

    def test_flagged_lines_in_orange(self):
        self.assertFalse(has_color(panel(0.0), annotate._ORANGE))
        self.assertTrue(has_color(panel(0.0, flagged=["left_ankle"]), annotate._ORANGE))

    def test_smoothed_lines_in_yellow(self):
        self.assertFalse(has_color(panel(0.0), annotate._SMOOTHED))
        self.assertTrue(has_color(panel(0.0, smoothed=["right_heel"]), annotate._SMOOTHED))

    def test_more_names_more_lines(self):
        one = panel(0.0, flagged=["left_ankle"])
        two = panel(0.0, flagged=["left_ankle", "left_heel"])
        rows = lambda img: np.where((img == annotate._ORANGE).all(axis=-1).any(axis=1))[0].max()
        self.assertGreater(rows(two), rows(one))

    def test_subtitle_and_controls(self):
        base = panel(0.0)
        self.assertFalse(np.array_equal(base, panel(0.0, subtitle="walk_01.mp4")))
        self.assertFalse(np.array_equal(base, panel(0.0, controls="Space: play / pause")))

    def test_short_panel_does_not_crash(self):
        img = draw_info_panel(60, 2.0, 1, 0.1, 1.0, 3.0, flagged=["a"] * 20, smoothed=["b"] * 20)
        self.assertEqual(img.shape, (60, PANEL_W, 3))


def pose_at(points):
    pose = [None] * pc.NUM_LANDMARKS
    for i, (x, y) in points.items():
        pose[i] = pc.Landmark(x, y)
    return pose


def result(pose=None, flags=None, smoothed=None, **kwargs):
    return FrameResult(frame_idx=5, time_s=0.2, H=None, pose=pose,
                       pose_flags=flags or {}, pose_smoothed=smoothed or set(), **kwargs)


def frame():
    return np.full((H, W, 3), 40, np.uint8)


class RenderFrameTest(unittest.TestCase):
    def test_shape_without_skeleton(self):
        annotated, skeleton = render_frame(frame(), result(), None, None)
        self.assertEqual(annotated.shape, (H, W + PANEL_W, 3))
        self.assertIsNone(skeleton)

    def test_shape_with_skeleton(self):
        r = result(pose_at({27: (0.25, 0.5)}), far_ep=(10, 10), near_ep=(600, 400))
        annotated, skeleton = render_frame(frame(), r, 0.1, 1.0, with_skeleton=True)
        self.assertEqual(annotated.shape, (H, W + PANEL_W, 3))
        self.assertEqual(skeleton.shape, annotated.shape)
        self.assertTrue(np.array_equal(skeleton[:, W:], annotated[:, W:]))   # same panel
        # the skeleton version has the drawing on black, not the frame
        self.assertEqual(tuple(skeleton[5, 300]), (0, 0, 0))
        self.assertEqual(tuple(annotated[5, 300]), (40, 40, 40))
        self.assertEqual(tuple(skeleton[240, 160]), tuple(annotated[240, 160]))  # same point drawn

    def test_draws_on_the_frame_in_place(self):
        img = frame()
        render_frame(img, result(pose_at({27: (0.25, 0.5)})), None, None)
        self.assertFalse((img == 40).all())

    def test_flagged_orange_smoothed_yellow(self):
        left, right = (0.25, 0.5), (0.75, 0.5)
        r = result(pose_at({27: left, 28: right}), flags={27: "spike", 28: "spike"}, smoothed={28})
        annotated, skeleton = render_frame(frame(), r, None, None, with_skeleton=True)
        for img in (annotated, skeleton):
            self.assertEqual(tuple(img[int(0.5 * H), int(0.25 * W)]), pc.FLAGGED_COLOR)
            self.assertEqual(tuple(img[int(0.5 * H), int(0.75 * W)]), pc.SMOOTHED_COLOR)

    def test_panel_lists_flagged_and_smoothed_separately(self):
        r = result(pose_at({27: (0.25, 0.5)}), flags={27: "spike", 28: "spike"}, smoothed={28})
        annotated, _ = render_frame(frame(), r, None, None)
        expected = draw_info_panel(H, 0.2, 5, None, None, None, flagged=["left_ankle"],
                                   smoothed=["right_ankle"])
        self.assertTrue(np.array_equal(annotated[:, W:], expected))

    def test_unflagged_point_drawn_normally(self):
        r = result(pose_at({27: (0.25, 0.5)}))
        annotated, _ = render_frame(frame(), r, None, None)
        self.assertEqual(tuple(annotated[int(0.5 * H), int(0.25 * W)]), (0, 255, 0))

    def test_rope_and_body_point(self):
        r = result(pose_at({27: (0.1, 0.1)}), body_px=(320, 240), far_ep=(100, 400), near_ep=(500, 400))
        annotated, _ = render_frame(frame(), r, None, None)
        self.assertEqual(tuple(annotated[240, 320]), annotate.BODY_COLOR)
        # the endpoint dots (the rope is drawn across their centres)
        self.assertEqual(tuple(annotated[405, 100]), annotate.FAR_COLOR)
        self.assertEqual(tuple(annotated[405, 500]), annotate.NEAR_COLOR)
        self.assertEqual(tuple(annotated[400, 300]), annotate.ROPE_COLOR)

    def test_one_endpoint_draws_no_rope(self):
        annotated, _ = render_frame(frame(), result(far_ep=(100, 400)), None, None)
        self.assertTrue((annotated[:, :W] == 40).all())


if __name__ == "__main__":
    unittest.main()
