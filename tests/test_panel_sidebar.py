"""Tests for panels (tmwt/ui/panel.py) and the file-list sidebar (tmwt/ui/sidebar.py)."""

import unittest

import numpy as np

from tmwt.ui import sidebar as sb
from tmwt.ui.panel import Panel
from tmwt.ui.sidebar import (APPROVED_MARK, DONE, FAILED, REJECTED_MARK, SAVED_MARK, SIDEBAR_W,
                             WAITING, WORKING, Sidebar)
from tmwt.ui.widgets import MAIN_H


class PanelTest(unittest.TestCase):
    def setUp(self):
        self.panel = Panel(100, 50, 200, 80)

    def test_contains_edges(self):
        self.assertTrue(self.panel.contains((100, 50)))
        self.assertTrue(self.panel.contains((299, 129)))
        self.assertFalse(self.panel.contains((300, 60)))
        self.assertFalse(self.panel.contains((150, 130)))
        self.assertFalse(self.panel.contains((99, 60)))
        self.assertFalse(self.panel.contains((150, 49)))
        self.assertFalse(self.panel.contains(None))

    def test_to_local(self):
        self.assertEqual(self.panel.to_local((130, 70)), (30, 20))
        self.assertEqual(self.panel.to_local((0, 0)), (-100, -50))   # outside is still converted
        self.assertIsNone(self.panel.to_local(None))

    def test_local_if_inside(self):
        self.assertEqual(self.panel.local_if_inside((130, 70)), (30, 20))
        self.assertIsNone(self.panel.local_if_inside((0, 0)))
        self.assertIsNone(self.panel.local_if_inside(None))

    def test_render_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            self.panel.render(None, None)


def names(n):
    return [f"walk_{i:02d}.mp4" for i in range(n)]


def row_y(n):
    """A y (sidebar pixels) inside visible row n."""
    return sb._TOP + n * sb._ROW_H + 10


class SidebarRowsTest(unittest.TestCase):
    def setUp(self):
        self.bar = Sidebar(960, 48, MAIN_H, names(5))

    def test_one_list_before_any_review(self):
        self.assertEqual(self.bar._rows(), [("file", i) for i in range(5)])

    def test_sections_once_reviewed(self):
        self.bar.mark_reviewed(1, APPROVED_MARK)
        self.bar.mark_reviewed(3, REJECTED_MARK)
        self.assertEqual(self.bar._rows(), [
            ("header", "Unreviewed (3)"), ("file", 0), ("file", 2), ("file", 4),
            ("header", "Reviewed (2)"), ("file", 1), ("file", 3)])

    def test_all_reviewed(self):
        for i in range(5):
            self.bar.mark_reviewed(i, SAVED_MARK)
        rows = self.bar._rows()
        self.assertEqual(rows[0], ("header", "Unreviewed (0)"))
        self.assertEqual(rows[1], ("header", "Reviewed (5)"))
        self.assertEqual(len(rows), 7)

    def test_set_state_and_mark_reviewed(self):
        self.assertEqual(self.bar.states, [WAITING] * 5)
        self.bar.set_state(2, FAILED, "no person found")
        self.assertEqual(self.bar.states[2], FAILED)
        self.assertEqual(self.bar.notes[2], "no person found")
        self.bar.set_state(2, WORKING)
        self.assertEqual(self.bar.notes[2], "")                     # the note is replaced
        self.bar.mark_reviewed(2, SAVED_MARK)
        self.assertEqual(self.bar.reviewed, [None, None, SAVED_MARK, None, None])

    def test_empty_list(self):
        bar = Sidebar(0, 0, MAIN_H, [])
        self.assertEqual(bar._rows(), [])
        self.assertIsNone(bar.row_at((50, row_y(0))))
        self.assertEqual(bar.render(None, None).shape, (MAIN_H, SIDEBAR_W, 3))


class SidebarRowAtTest(unittest.TestCase):
    def setUp(self):
        self.bar = Sidebar(960, 48, MAIN_H, names(5))

    def test_rows_map_to_files(self):
        for i in range(5):
            self.assertEqual(self.bar.row_at((50, row_y(i))), i)

    def test_row_edges(self):
        top = sb._TOP - 4
        self.assertEqual(self.bar.row_at((50, top)), 0)
        self.assertEqual(self.bar.row_at((50, top + sb._ROW_H - 1)), 0)
        self.assertEqual(self.bar.row_at((50, top + sb._ROW_H)), 1)
        self.assertIsNone(self.bar.row_at((50, top - 1)))

    def test_outside(self):
        self.assertIsNone(self.bar.row_at(None))
        self.assertIsNone(self.bar.row_at((-1, row_y(0))))
        self.assertIsNone(self.bar.row_at((SIDEBAR_W, row_y(0))))
        self.assertIsNone(self.bar.row_at((50, 10)))                # the heading
        self.assertIsNone(self.bar.row_at((50, row_y(7))))          # below the last file
        self.assertIsNone(self.bar.row_at((50, MAIN_H - 5)))        # the legend

    def test_headers_are_not_files(self):
        self.bar.mark_reviewed(0, APPROVED_MARK)
        # rows: header, 1, 2, 3, 4, header, 0
        self.assertIsNone(self.bar.row_at((50, row_y(0))))
        self.assertEqual(self.bar.row_at((50, row_y(1))), 1)
        self.assertIsNone(self.bar.row_at((50, row_y(5))))
        self.assertEqual(self.bar.row_at((50, row_y(6))), 0)

    def test_under_scrolling(self):
        bar = Sidebar(960, 48, MAIN_H, names(40))
        bar.scroll(-5)
        self.assertEqual(bar.row_at((50, row_y(0))), 5)
        self.assertEqual(bar.row_at((50, row_y(3))), 8)

    def test_rows_below_the_visible_ones(self):
        bar = Sidebar(960, 48, MAIN_H, names(40))
        visible = bar._visible_rows()
        self.assertEqual(bar.row_at((50, row_y(visible - 1))), visible - 1)
        self.assertIsNone(bar.row_at((50, row_y(visible))))


class SidebarScrollTest(unittest.TestCase):
    N = 40

    def setUp(self):
        self.bar = Sidebar(960, 48, MAIN_H, names(self.N))
        self.visible = self.bar._visible_rows()
        self.last_start = self.N - self.visible

    def test_fractional_steps_accumulate(self):
        self.bar.scroll(-0.4)
        self.bar.scroll(-0.4)
        self.assertEqual(self.bar._first_row(), 0)
        self.bar.scroll(-0.4)
        self.assertEqual(self.bar._first_row(), 1)
        # the leftover 0.2 row is kept
        self.bar.scroll(-0.8)
        self.assertEqual(self.bar._first_row(), 2)

    def test_opposite_directions_cancel(self):
        self.bar.scroll(-3)
        self.bar.scroll(0.6)
        self.bar.scroll(-0.6)
        self.assertEqual(self.bar._first_row(), 3)

    def test_clamped_at_the_top(self):
        self.bar.scroll(5)
        self.assertEqual(self.bar._first_row(), 0)
        self.bar.scroll(-1)                  # no leftover from the overshoot
        self.assertEqual(self.bar._first_row(), 1)

    def test_clamped_at_the_bottom(self):
        self.bar.scroll(-1000)
        self.assertEqual(self.bar._first_row(), self.last_start)
        self.bar.scroll(1)                   # one notch back up moves at once
        self.assertEqual(self.bar._first_row(), self.last_start - 1)

    def test_short_list_does_not_scroll(self):
        bar = Sidebar(960, 48, MAIN_H, names(3))
        bar.scroll(-10)
        self.assertEqual(bar._first_row(), 0)
        self.assertEqual(bar.row_at((50, row_y(0))), 0)

    def test_follows_the_active_file(self):
        self.bar.active = 30
        self.assertEqual(self.bar._first_row(), 30 - self.visible // 2)
        self.bar.active = 39
        self.assertEqual(self.bar._first_row(), self.last_start)
        self.bar.active = 1
        self.assertEqual(self.bar._first_row(), 0)

    def test_new_active_file_resets_scrolling(self):
        self.bar.active = 30
        self.bar.scroll(10)
        scrolled = self.bar._first_row()
        self.assertEqual(scrolled, 30 - self.visible // 2 - 10)
        self.bar.active = 30                 # same file: stays where the user scrolled
        self.assertEqual(self.bar._first_row(), scrolled)
        self.bar.active = 31
        self.assertEqual(self.bar._first_row(), 31 - self.visible // 2)

    def test_follows_the_active_file_into_the_reviewed_section(self):
        self.bar.mark_reviewed(35, SAVED_MARK)
        self.bar.active = 35
        pos = self.bar._rows().index(("file", 35))
        first = self.bar._first_row()
        self.assertTrue(first <= pos < first + self.visible)


class SidebarRenderTest(unittest.TestCase):
    def setUp(self):
        self.bar = Sidebar(960, 48, MAIN_H, names(5))
        for i in range(5):
            self.bar.set_state(i, DONE)

    def name_region(self, img, row):
        """Pixels of a row's name, left of the space kept for its mark."""
        y = sb._TOP + row * sb._ROW_H
        return img[y:y + 18, sb._X0:SIDEBAR_W - sb._X0 - sb._MARK_W]

    def test_render_size(self):
        self.assertEqual(self.bar.render(None, None).shape, (MAIN_H, SIDEBAR_W, 3))
        tall = Sidebar(0, 0, 300, names(40))
        self.assertEqual(tall.render((50, 100), None).shape, (300, SIDEBAR_W, 3))

    def test_reviewed_row_is_faded(self):
        self.bar.mark_reviewed(0, APPROVED_MARK)
        img = self.bar.render(None, None)
        # rows: header, 1, 2, 3, 4, header, 0
        fresh = self.name_region(img, 1)[..., 1].max()
        faded = self.name_region(img, 6)[..., 1].max()
        self.assertGreater(fresh, 180)                 # green name
        self.assertLess(faded, 120)                    # faded toward the background
        self.assertGreater(faded, sb._BG)

    def test_active_reviewed_row_is_not_faded(self):
        self.bar.mark_reviewed(0, APPROVED_MARK)
        self.bar.active = 0
        img = self.bar.render(None, None)
        self.assertGreater(self.name_region(img, 6)[..., 1].max(), 180)

    def test_marks_differ(self):
        images = []
        for outcome in (APPROVED_MARK, SAVED_MARK, REJECTED_MARK):
            bar = Sidebar(0, 0, MAIN_H, names(2))
            bar.mark_reviewed(0, outcome)
            images.append(bar.render(None, None).tobytes())
        self.assertEqual(len(set(images)), 3)

    def test_hover_highlights_only_review_targets(self):
        mouse = (50, row_y(2))
        plain = self.bar.render(None, None)
        self.assertTrue(np.array_equal(self.bar.render(mouse, None), plain))
        self.bar.review_targets = {2}
        hover = self.bar.render(mouse, None)
        pressed = self.bar.render(mouse, 2)
        self.assertFalse(np.array_equal(hover, self.bar.render(None, None)))
        self.assertFalse(np.array_equal(hover, pressed))

    def test_scrollbar_only_when_the_list_is_long(self):
        x = SIDEBAR_W - 5
        short = self.bar.render(None, None)
        self.assertTrue((short[sb._TOP:sb._TOP + 100, x] == sb._BG).all())
        long_ = Sidebar(0, 0, MAIN_H, names(40)).render(None, None)
        self.assertFalse((long_[sb._TOP:sb._TOP + 100, x] == sb._BG).all())

    def test_long_names_and_notes_render(self):
        bar = Sidebar(0, 0, MAIN_H, ["x" * 300 + ".mp4"])
        bar.set_state(0, FAILED, "n" * 300)
        bar.mark_reviewed(0, REJECTED_MARK)
        self.assertEqual(bar.render(None, None).shape, (MAIN_H, SIDEBAR_W, 3))


if __name__ == "__main__":
    unittest.main()
