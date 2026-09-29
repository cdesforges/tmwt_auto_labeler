"""Tests for the seek bar's geometry and hit-testing (tmwt/ui/seek_bar.py)."""

import unittest

import numpy as np

from tmwt.ui.seek_bar import MARK, SCRUB, Marker, SeekBar
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
        self.bar.draw(img, 0.5, self.markers, "", drag=(MARK, "start"), drag_fraction=0.1)
        x, y = self.bar.x_at(0.5), self.bar.Y
        self.assertFalse((img[y - 8:y - 5, x] == 255).all(axis=-1).any())   # no white playhead dot
        img = np.zeros((MAIN_H, MAIN_W, 3), np.uint8)
        self.bar.draw(img, 0.5, self.markers, "")
        self.assertTrue((img[y - 6, x] == 255).all())


if __name__ == "__main__":
    unittest.main()
