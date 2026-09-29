"""Tests for the seek bar's geometry and hit-testing (tmwt/ui/seek_bar.py)."""

import unittest

import numpy as np

from tmwt.ui.player import seek_state
from tmwt.ui.seek_bar import MARK, SCRUB, Marker, SeekBar, SeekState
from tmwt.ui.widgets import GREEN, MAIN_H, MAIN_W, RED


class SeekBarTest(unittest.TestCase):
    def setUp(self):
        self.bar = SeekBar()
        self.markers = [Marker(0.25, GREEN, "start"), Marker(0.75, RED, "stop")]

    def tab(self, fraction):
        return (self.bar.x_at(fraction), self.bar.Y + (self.bar.TAB_TOP + self.bar.TAB_BOTTOM) // 2)

    def test_fraction_round_trip_and_clamping(self):
        for f in (0.0, 0.1, 0.5, 1.0):
            self.assertAlmostEqual(self.bar.fraction_at(self.bar.x_at(f)), f, places=2)
        self.assertEqual(self.bar.fraction_at(-100), 0.0)
        self.assertEqual(self.bar.fraction_at(10_000), 1.0)

    def test_tab_grabs_its_mark(self):
        self.assertEqual(self.bar.grab(self.tab(0.25), self.markers), (MARK, "start"))
        self.assertEqual(self.bar.grab(self.tab(0.75), self.markers), (MARK, "stop"))

    def test_track_over_a_mark_scrubs(self):
        # Right after marking, the mark is under the playhead: a press on the
        # track itself must still scrub.
        pt = (self.bar.x_at(0.25), self.bar.Y)
        self.assertEqual(self.bar.grab(pt, self.markers), (SCRUB, None))

    def test_fixed_marks_are_not_draggable(self):
        fixed = [Marker(0.25, GREEN, None)]
        self.assertIsNone(self.bar.marker_at(self.tab(0.25), fixed))

    def test_overlapping_tabs_pick_the_nearest(self):
        close = [Marker(0.500, GREEN, "start"), Marker(0.505, RED, "stop")]
        x, y = self.tab(0.505)
        self.assertEqual(self.bar.marker_at((x + 1, y), close), "stop")

    def test_off_the_bar(self):
        self.assertIsNone(self.bar.grab((self.bar.x_at(0.5), self.bar.Y - 40), self.markers))
        self.assertIsNone(self.bar.grab(None, self.markers))

    def test_tabs_stay_inside_the_seek_strip(self):
        # The tab (even enlarged) must not reach the button bar below.
        from tmwt.ui.widgets import BAR_H
        self.assertLess(self.bar.Y + self.bar.TAB_BOTTOM + 2, MAIN_H - BAR_H)

    def test_playhead_hidden_while_dragging_a_mark(self):
        img = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        self.bar.draw(img, SeekState(0.5, self.markers, ""), drag=(MARK, "start"), drag_fraction=0.1)
        x, y = self.bar.x_at(0.5), self.bar.Y
        self.assertFalse((img[y - 8:y - 5, x] == 255).all(axis=-1).any())   # no white playhead dot
        img = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        self.bar.draw(img, SeekState(0.5, self.markers, ""))
        self.assertTrue((img[y - 6, x] == 255).all())

    def test_playhead_filled_on_a_mark_keeps_white_outline(self):
        img = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        self.bar.draw(img, SeekState(0.5, self.markers, "", GREEN))
        x, y = self.bar.x_at(0.5), self.bar.Y
        self.assertEqual(tuple(img[y, x]), GREEN)              # centre filled
        self.assertTrue((img[y - 6, x] == 255).all())           # rim still white


class SeekStateTest(unittest.TestCase):
    TIMES = [i / 25 for i in range(100)]   # 25 fps, 4 s

    def test_fill_when_a_mark_is_on_the_frame(self):
        marks = [(1.0, GREEN, "start"), (3.0, RED, "stop")]
        self.assertEqual(seek_state(self.TIMES, 25, marks).playhead_fill, GREEN)
        self.assertEqual(seek_state(self.TIMES, 75, marks).playhead_fill, RED)
        self.assertIsNone(seek_state(self.TIMES, 26, marks).playhead_fill)

    def test_mark_between_frames_uses_its_nearest_frame(self):
        marks = [(1.015, GREEN, "start")]   # between frames 25 (1.00) and 26 (1.04)
        self.assertEqual(seek_state(self.TIMES, 25, marks).playhead_fill, GREEN)
        self.assertIsNone(seek_state(self.TIMES, 26, marks).playhead_fill)

    def test_flagged_frames_become_stretches(self):
        state = seek_state(self.TIMES, 0, [], flagged=[10, 11, 12, 50])
        self.assertEqual(len(state.flagged), 2)                    # 10-12 merge, 50 alone
        a, b = state.flagged[0]
        self.assertAlmostEqual(a, self.TIMES[10] / self.TIMES[-1])
        self.assertAlmostEqual(b, self.TIMES[13] / self.TIMES[-1])

    def test_flagged_stretch_is_drawn_orange(self):
        from tmwt.ui.seek_bar import FLAGGED_COLOR
        bar = SeekBar()
        img = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        bar.draw(img, SeekState(0.0, [], "", None, [(0.5, 0.5)]))    # one frame, zero width
        self.assertEqual(tuple(img[bar.Y, bar.x_at(0.5)]), FLAGGED_COLOR)

    def test_unset_marks_are_ignored(self):
        state = seek_state(self.TIMES, 0, [(None, GREEN, "start")])
        self.assertEqual(state.markers, [])
        self.assertIsNone(state.playhead_fill)


if __name__ == "__main__":
    unittest.main()
