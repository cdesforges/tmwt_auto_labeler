"""
Tests for the top bar's title fitting and layout, and the text helpers behind
it. Run from the repository root:

    python -m unittest discover tests
"""

import unittest

import cv2
import numpy as np

from tmwt.ui.top_bar import (TOPBAR_H, TopBar, _TITLE_SCALE, _TITLE_THICKNESS, fit_title,
                             folder_title)
from tmwt.ui.widgets import FONT, ascii_text, truncate_middle


def text_width(text, scale=_TITLE_SCALE, thickness=_TITLE_THICKNESS):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


class TruncateMiddleTest(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(truncate_middle("control_vids", 500, 0.65, 2), "control_vids")

    def test_exact_fit_is_unchanged(self):
        text = "control_vids"
        self.assertEqual(truncate_middle(text, text_width(text), _TITLE_SCALE, 2), text)

    def test_long_text_fits_and_keeps_both_ends(self):
        text = "control_vids_cluster_version_test2_" * 10 + "END"
        for max_w in (40, 100, 300, 876):
            with self.subTest(max_w=max_w):
                out = truncate_middle(text, max_w, _TITLE_SCALE, 2)
                self.assertLessEqual(text_width(out), max_w)
                self.assertIn("...", out)
                head, tail = out.split("...")
                self.assertTrue(text.startswith(head))
                self.assertTrue(text.endswith(tail))
                self.assertLessEqual(abs(len(head) - len(tail)), 1)   # balanced

    def test_uses_the_space_it_has(self):
        # One more character would no longer fit.
        text = "a_really_long_folder_name_for_testing_truncation" * 3
        max_w = 300
        out = truncate_middle(text, max_w, _TITLE_SCALE, 2)
        kept = len(out) - 3
        head, tail = (kept + 2) // 2, (kept + 1) // 2
        longer = text[:head] + "..." + text[len(text) - tail:]
        self.assertGreater(text_width(longer), max_w)

    def test_too_narrow_gives_empty(self):
        self.assertEqual(truncate_middle("control_vids", 5, _TITLE_SCALE, 2), "")
        self.assertEqual(truncate_middle("control_vids", 0, _TITLE_SCALE, 2), "")

    def test_empty_text(self):
        self.assertEqual(truncate_middle("", 100, _TITLE_SCALE, 2), "")


class AsciiTextTest(unittest.TestCase):
    def test_accents_are_dropped(self):
        self.assertEqual(ascii_text("vidéos_Zürich"), "videos_Zurich")

    def test_unprintable_becomes_question_mark(self):
        self.assertEqual(ascii_text("walk_歩行"), "walk_??")
        self.assertEqual(ascii_text("a\tb"), "a?b")

    def test_plain_ascii_is_unchanged(self):
        self.assertEqual(ascii_text("control_vids (2026)"), "control_vids (2026)")


class FitTitleTest(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(fit_title("  control \n vids  ", 500), "control vids")

    def test_long_unicode_title_fits(self):
        title = "Études_de_marche_" * 20
        out = fit_title(title, 400)
        self.assertLessEqual(text_width(out), 400)
        self.assertTrue(out.isascii())
        self.assertTrue(out.startswith("Etudes"))

    def test_negative_width(self):
        self.assertEqual(fit_title("control_vids", -10), "")


class FolderTitleTest(unittest.TestCase):
    def test_trailing_slash(self):
        self.assertEqual(folder_title("media/control_vids/"), "control_vids")

    def test_plain_name(self):
        self.assertEqual(folder_title("control_vids"), "control_vids")


class TopBarLayoutTest(unittest.TestCase):
    WIDTH = 1280

    def title_box(self, bar, title):
        """Columns where the rendered title differs from a bar with no title."""
        bar.title = title
        with_title = bar.render(None, None)
        bar.title = ""
        without = bar.render(None, None)
        cols = np.where((with_title != without).any(axis=(0, 2)))[0]
        return (cols.min(), cols.max()) if cols.size else None

    def assert_title_clear_of_controls(self, bar, title):
        box = self.title_box(bar, title)
        self.assertIsNotNone(box)
        left_edge = bar.save_button.x + bar.save_button.w
        self.assertGreaterEqual(box[0] - left_edge, 20)
        logo_left = bar.w - 8 - bar._logo[0].shape[1] if bar._logo else bar.w
        self.assertGreaterEqual(logo_left - box[1], 20)

    def test_render_size(self):
        img = TopBar(self.WIDTH, "control_vids").render(None, None)
        self.assertEqual(img.shape, (TOPBAR_H, self.WIDTH, 3))

    def test_long_title_stays_clear_of_buttons_and_logo(self):
        for show_save in (False, True):
            bar = TopBar(self.WIDTH)
            bar.show_save = show_save
            self.assert_title_clear_of_controls(bar, "control_vids_cluster_version_" * 20)

    def test_title_is_centred(self):
        bar = TopBar(self.WIDTH)
        lo, hi = self.title_box(bar, "control_vids")
        self.assertLessEqual(abs((lo + hi) / 2 - self.WIDTH / 2), 3)

    def test_narrow_window_does_not_crash(self):
        bar = TopBar(300, "control_vids_cluster_version_test2")
        self.assertEqual(bar.render(None, None).shape, (TOPBAR_H, 300, 3))

    def test_missing_logo(self):
        bar = TopBar(self.WIDTH, "control_vids", logo_path="does/not/exist.png")
        self.assertIsNone(bar._logo)
        self.assert_title_clear_of_controls(bar, "x" * 400)

    def test_save_button_only_during_review(self):
        bar = TopBar(self.WIDTH)
        self.assertEqual([b.icon for b in bar.buttons], ["close"])
        bar.show_save = True
        self.assertEqual([b.icon for b in bar.buttons], ["close", "save"])

    def test_hover_and_pressed_states_render(self):
        bar = TopBar(self.WIDTH, "control_vids")
        bar.show_save = True
        b = bar.save_button
        inside = (b.x + 5, b.y + 5)
        normal = bar.render(None, None)
        hover = bar.render(inside, None)
        pressed = bar.render(inside, b.value)
        self.assertFalse(np.array_equal(normal, hover))
        self.assertFalse(np.array_equal(hover, pressed))
        self.assertIs(bar.hovered_button(inside), b)


if __name__ == "__main__":
    unittest.main()
