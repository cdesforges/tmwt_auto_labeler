"""Tests for the drawing building blocks: text, layout, buttons, icons, badges (tmwt/ui/widgets.py)."""

import unittest

import cv2
import numpy as np

from tmwt.ui import widgets
from tmwt.ui.widgets import (BAR_H, BTN_H, FONT, ICONS, MAIN_H, MAIN_W, WHITE, Button, IconButton,
                             KeyedButton, TextButton, bar_buttons, button_row, button_state, dimmed,
                             draw_badge, draw_icon, draw_tooltip, fit, frame_screen, truncate)


def text_width(text, scale, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def drawn_box(img):
    """(x0, y0, x1, y1) of the non-black pixels of `img`, or None."""
    ys, xs = np.where(img.any(axis=-1))
    return (xs.min(), ys.min(), xs.max(), ys.max()) if xs.size else None


def blank(w=MAIN_W, h=MAIN_H):
    return np.zeros((h, w, 3), np.uint8)


class TruncateTest(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(truncate("walk_01.mp4", 500, 0.5), "walk_01.mp4")

    def test_long_text_fits_and_keeps_the_start(self):
        text = "a_very_long_video_file_name_from_the_clinic.mp4"
        for max_w in (60, 120, 200):
            with self.subTest(max_w=max_w):
                out = truncate(text, max_w, 0.5)
                self.assertTrue(out.endswith("..."))
                self.assertTrue(text.startswith(out[:-3]))
                self.assertLessEqual(text_width(out, 0.5), max_w)

    def test_keeps_as_much_as_fits(self):
        text = "a_very_long_video_file_name_from_the_clinic.mp4"
        out = truncate(text, 150, 0.5)
        one_more = text[:len(out) - 3 + 1] + "..."
        self.assertGreater(text_width(one_more, 0.5), 150)

    def test_thickness_is_taken_into_account(self):
        text = "a_very_long_video_file_name.mp4"
        thin, thick = truncate(text, 150, 0.6, 1), truncate(text, 150, 0.6, 3)
        self.assertLessEqual(text_width(thick, 0.6, 3), 150)
        self.assertLessEqual(len(thick), len(thin))

    def test_empty_text(self):
        self.assertEqual(truncate("", 100, 0.5), "")

    def test_too_narrow_still_fits(self):
        out = truncate("walk.mp4", 5, 0.5)
        self.assertLessEqual(text_width(out, 0.5), 5)


class ButtonRowTest(unittest.TestCase):
    SPECS = [("OK", "ok", (13,)), ("A much longer button label here", "long", ()),
             ("Play", "toggle", (32,), "play")]

    def test_types_values_and_keys(self):
        buttons = button_row(self.SPECS, 100)
        self.assertEqual([type(b) for b in buttons], [TextButton, TextButton, IconButton])
        self.assertEqual([b.value for b in buttons], ["ok", "long", "toggle"])
        self.assertEqual([b.keys for b in buttons], [(13,), (), (32,)])
        self.assertEqual(buttons[2].icon, "play")
        self.assertTrue(all(b.y == 100 and b.h == BTN_H for b in buttons))

    def test_centred_with_even_gaps(self):
        buttons = button_row(self.SPECS, 100)
        for a, b in zip(buttons, buttons[1:]):
            self.assertEqual(b.x - (a.x + a.w), widgets._BTN_GAP)
        left = buttons[0].x
        right = MAIN_W - (buttons[-1].x + buttons[-1].w)
        self.assertLessEqual(abs(left - right), 1)

    def test_min_width_and_text_width(self):
        short, long_, icon = button_row(self.SPECS, 0)
        self.assertEqual(short.w, widgets._BTN_MIN_W)
        self.assertEqual(long_.w, text_width(long_.text, 0.6) + 48)
        self.assertGreater(long_.w, widgets._BTN_MIN_W)

    def test_icon_buttons_have_a_fixed_width(self):
        buttons = button_row([("Some very long tooltip text for an icon", "a", (), "menu"),
                              ("x", "b", (), "close")], 0)
        self.assertEqual([b.w for b in buttons], [widgets._ICON_BTN_W] * 2)

    def test_single_button_is_centred(self):
        (b,) = button_row([("Cancel", "cancel", ())], 10)
        self.assertLessEqual(abs(b.x + b.w / 2 - MAIN_W / 2), 1)

    def test_no_specs(self):
        self.assertEqual(button_row([], 0), [])

    def test_bar_buttons_sit_in_the_bottom_bar(self):
        for b in bar_buttons(self.SPECS):
            self.assertGreaterEqual(b.y, MAIN_H - BAR_H)
            self.assertLessEqual(b.y + b.h, MAIN_H)
            # vertically centred in the bar
            self.assertEqual(b.y - (MAIN_H - BAR_H), MAIN_H - (b.y + b.h))


class ButtonTest(unittest.TestCase):
    def make_buttons(self):
        rect = (100, 50, 200, BTN_H)
        return [TextButton(rect, "Confirm", "confirm", (13,)),
                KeyedButton(rect, "Approve", "approve", (ord("1"),), key_label="[1]"),
                IconButton(rect, "Play", "toggle", (32,), icon="play")]

    def test_contains_edges(self):
        b = Button((10, 20, 30, 40), "t", "v")
        self.assertTrue(b.contains((10, 20)))            # top-left is inside
        self.assertTrue(b.contains((39, 59)))            # last pixel inside
        self.assertFalse(b.contains((40, 30)))           # right edge is exclusive
        self.assertFalse(b.contains((20, 60)))           # bottom edge is exclusive
        self.assertFalse(b.contains((9, 30)))
        self.assertFalse(b.contains((20, 19)))
        self.assertFalse(b.contains(None))

    def test_button_state(self):
        b = Button((0, 0, 100, 40), "t", "v")
        inside, outside = (50, 20), (500, 20)
        self.assertEqual(button_state(b, outside, None), "normal")
        self.assertEqual(button_state(b, None, None), "normal")
        self.assertEqual(button_state(b, inside, None), "hover")
        self.assertEqual(button_state(b, inside, "v"), "pressed")
        self.assertEqual(button_state(b, outside, "v"), "normal")     # dragged off
        self.assertEqual(button_state(b, inside, "other"), "normal")  # another button held

    def test_every_subclass_draws_in_every_state(self):
        for b in self.make_buttons():
            images = {}
            for state in ("normal", "hover", "pressed"):
                img = blank(400, 150)
                b.draw(img, state)
                self.assertIsNotNone(drawn_box(img))
                x0, y0, x1, y1 = drawn_box(img)
                self.assertGreaterEqual(x0, b.x)
                self.assertGreaterEqual(y0, b.y)
                self.assertLess(x1, b.x + b.w)
                self.assertLess(y1, b.y + b.h)
                images[state] = img
            with self.subTest(button=type(b).__name__):
                self.assertFalse(np.array_equal(images["normal"], images["hover"]))
                self.assertFalse(np.array_equal(images["hover"], images["pressed"]))
                self.assertFalse(np.array_equal(images["normal"], images["pressed"]))

    def test_pressed_shows_the_yellow_border(self):
        b = TextButton((10, 10, 150, BTN_H), "OK", "ok")
        img = blank(200, 80)
        b.draw(img, "pressed")
        self.assertEqual(tuple(img[10 + BTN_H - 1, 80]), widgets.YELLOW)

    def test_base_class_has_no_content(self):
        with self.assertRaises(NotImplementedError):
            Button((0, 0, 50, 20), "t", "v").draw(blank(100, 50), "normal")

    def test_keyed_button_shows_its_shortcut(self):
        rect = (0, 0, 300, BTN_H)
        with_key = blank(300, 60)
        KeyedButton(rect, "Approve", "a", key_label="[1]").draw(with_key, "normal")
        without = blank(300, 60)
        KeyedButton(rect, "Approve", "a", key_label="").draw(without, "normal")
        self.assertFalse(np.array_equal(with_key, without))

    def test_draw_buttons_uses_each_state(self):
        a = TextButton((0, 0, 100, BTN_H), "A", "a")
        b = TextButton((120, 0, 100, BTN_H), "B", "b")
        img = blank(300, 60)
        widgets.draw_buttons(img, [a, b], (50, 20), "a")
        pressed = blank(300, 60)
        a.draw(pressed, "pressed")
        b.draw(pressed, "normal")
        self.assertTrue(np.array_equal(img, pressed))


class DrawIconTest(unittest.TestCase):
    def test_every_icon_draws_around_its_centre(self):
        for name in ICONS:
            with self.subTest(icon=name):
                img = blank(200, 200)
                draw_icon(img, name, (100, 100), 11)
                box = drawn_box(img)
                self.assertIsNotNone(box)
                x0, y0, x1, y1 = box
                self.assertTrue(100 - 32 <= x0 <= x1 <= 100 + 32, box)
                self.assertTrue(100 - 16 <= y0 <= y1 <= 100 + 16, box)

    def test_icons_differ(self):
        drawn = []
        for name in ICONS:
            img = blank(100, 100)
            draw_icon(img, name, (50, 50))
            drawn.append(img.tobytes())
        self.assertEqual(len(set(drawn)), len(ICONS))

    def test_colour(self):
        img = blank(60, 60)
        draw_icon(img, "pause", (30, 30), 11, color=(0, 0, 255))
        self.assertEqual(tuple(img[30, 25]), (0, 0, 255))

    def test_mark_icons_use_the_mark_colours(self):
        for name, color in (("mark_start", widgets.GREEN), ("mark_stop", widgets.RED)):
            img = blank(60, 60)
            draw_icon(img, name, (30, 30))
            self.assertEqual(tuple(img[30, 30]), color)

    def test_unknown_icon(self):
        with self.assertRaises(ValueError):
            draw_icon(blank(50, 50), "rewind", (25, 25))


class DrawBadgeTest(unittest.TestCase):
    TEXT = "Walk start marked"

    def test_left_aligned_box(self):
        img = blank(600, 200)
        bottom = draw_badge(img, self.TEXT, (40, 30), WHITE)
        (tw, th), _ = cv2.getTextSize(self.TEXT, FONT, 0.55, 1)
        self.assertEqual(bottom, 30 + th + 16)
        self.assertEqual(drawn_box(img), (40, 30, 40 + tw + 16, bottom))

    def test_centred_box(self):
        img = blank(600, 200)
        draw_badge(img, self.TEXT, (300, 30), WHITE, center=True)
        x0, _, x1, _ = drawn_box(img)
        self.assertLessEqual(abs((x0 + x1) / 2 - 300), 1)

    def test_bigger_text_gives_a_bigger_box(self):
        small, big = blank(800, 200), blank(800, 200)
        b_small = draw_badge(small, self.TEXT, (10, 10), WHITE)
        b_big = draw_badge(big, self.TEXT, (10, 10), WHITE, scale=0.9, thickness=2)
        self.assertGreater(b_big, b_small)
        self.assertGreater(drawn_box(big)[2], drawn_box(small)[2])

    def test_border_in_the_colour(self):
        img = blank(600, 200)
        draw_badge(img, self.TEXT, (40, 30), widgets.ORANGE)
        self.assertEqual(tuple(img[30, 60]), widgets.ORANGE)
        self.assertEqual(tuple(img[35, 42]), (20, 20, 20))


class DrawTooltipTest(unittest.TestCase):
    def test_below_the_anchor(self):
        img = blank(400, 200)
        draw_tooltip(img, "Quit", (50, 80))
        x0, y0, _, _ = drawn_box(img)
        self.assertEqual((x0, y0), (50, 84))

    def test_above_the_anchor(self):
        img = blank(400, 200)
        draw_tooltip(img, "Quit", (50, 80), above=True)
        _, _, _, y1 = drawn_box(img)
        self.assertEqual(y1, 76)

    def test_clamped_inside_the_image(self):
        for anchor in ((395, 50), (-40, 50), (0, 50)):
            with self.subTest(anchor=anchor):
                img = blank(400, 200)
                draw_tooltip(img, "Save progress & quit", anchor)
                x0, _, x1, _ = drawn_box(img)
                self.assertGreaterEqual(x0, 2)
                self.assertLessEqual(x1, 400 - 2)


class FitTest(unittest.TestCase):
    def test_wide_image_is_letterboxed(self):
        img = np.full((50, 100, 3), 200, np.uint8)
        canvas, s, x0, y0 = fit(img, 960, 720)
        self.assertEqual(canvas.shape, (720, 960, 3))
        self.assertAlmostEqual(s, 9.6)
        self.assertEqual((x0, y0), (0, 120))
        self.assertEqual(drawn_box(canvas), (0, 120, 959, 599))

    def test_tall_image_is_pillarboxed(self):
        img = np.full((400, 100, 3), 200, np.uint8)
        canvas, s, x0, y0 = fit(img, 960, 720)
        self.assertAlmostEqual(s, 1.8)
        self.assertEqual((x0, y0), (390, 0))
        self.assertEqual(drawn_box(canvas), (390, 0, 569, 719))

    def test_offsets_map_image_pixels(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[60:70, 150:160] = 255
        canvas, s, x0, y0 = fit(img, 500, 500)
        cx, cy = int(155 * s + x0), int(65 * s + y0)
        self.assertTrue((canvas[cy, cx] == 255).all())

    def test_downscaling(self):
        img = np.full((1080, 1920, 3), 90, np.uint8)
        canvas, s, x0, y0 = fit(img, 960, 720)
        self.assertAlmostEqual(s, 0.5)
        self.assertEqual((x0, y0), (0, 90))

    def test_none_image(self):
        canvas, s, x0, y0 = fit(None, 30, 20)
        self.assertEqual(canvas.shape, (20, 30, 3))
        self.assertFalse(canvas.any())
        self.assertEqual((s, x0, y0), (1.0, 0, 0))

    def test_extreme_aspect_keeps_at_least_one_pixel(self):
        canvas, _, _, _ = fit(np.full((1, 5000, 3), 255, np.uint8), 100, 100)
        self.assertTrue(canvas.any())


class DimmedTest(unittest.TestCase):
    def test_darkens_and_fits_the_main_area(self):
        out = dimmed(np.full((MAIN_H, MAIN_W, 3), 200, np.uint8), 0.25)
        self.assertEqual(out.shape, (MAIN_H, MAIN_W, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out[100, 100, 0]), 50)

    def test_none_is_black(self):
        out = dimmed(None, 0.5)
        self.assertEqual(out.shape, (MAIN_H, MAIN_W, 3))
        self.assertFalse(out.any())


class FrameScreenTest(unittest.TestCase):
    def test_image_between_the_strips(self):
        img = np.full((480, 640, 3), 200, np.uint8)
        main, s, x0, y0 = frame_screen(img, top=84, bottom=98)
        self.assertEqual(main.shape, (MAIN_H, MAIN_W, 3))
        avail_h = MAIN_H - 84 - 98
        self.assertAlmostEqual(s, min(MAIN_W / 640, avail_h / 480))
        self.assertGreaterEqual(y0, 84)
        # the image's corners map inside the free strip
        self.assertTrue((main[y0, x0] == 200).all())
        self.assertTrue((main[int(y0 + 479 * s), int(x0 + 639 * s)] == 200).all())
        self.assertTrue((main[MAIN_H - 98:] == 20).all())      # the bottom bar
        self.assertFalse(main[:84].any())                       # the top strip

    def test_default_bottom_is_the_button_bar(self):
        main, _, _, y0 = frame_screen(np.full((100, 100, 3), 255, np.uint8))
        self.assertEqual(y0, 0)
        self.assertTrue((main[MAIN_H - BAR_H:] == 20).all())
        self.assertTrue((main[MAIN_H - BAR_H - 1, MAIN_W // 2] == 255).all())

    def test_none_image(self):
        main, s, x0, y0 = frame_screen(None, top=10)
        self.assertEqual(main.shape, (MAIN_H, MAIN_W, 3))
        self.assertEqual((s, x0, y0), (1.0, 0, 10))


if __name__ == "__main__":
    unittest.main()


class RowFittingTest(unittest.TestCase):
    """button_row tightens a row too wide for the main area."""

    def test_normal_row_keeps_its_gaps(self):
        a, b = button_row([("One", "a", ()), ("Two", "b", ())], 100)
        self.assertEqual(b.x - (a.x + a.w), 16)

    def test_wide_row_is_tightened_to_fit(self):
        specs = [(f"A fairly long button label {i}", str(i), ()) for i in range(4)]
        buttons = button_row(specs, 100)
        self.assertGreaterEqual(buttons[0].x, 0)
        self.assertLessEqual(buttons[-1].x + buttons[-1].w, MAIN_W)
        for b in buttons:                                      # labels still fit their buttons
            self.assertGreater(b.w, cv2.getTextSize(b.text, FONT, 0.6, 1)[0][0])

    def test_far_too_wide_row_shortens_labels(self):
        specs = [(f"An extremely long button label that cannot possibly fit {i}", str(i), ())
                 for i in range(5)]
        buttons = button_row(specs, 100)
        self.assertGreaterEqual(buttons[0].x, 0)
        self.assertLessEqual(buttons[-1].x + buttons[-1].w, MAIN_W)
        self.assertTrue(all(b.text.endswith("...") for b in buttons))
        self.assertEqual([b.value for b in buttons], ["0", "1", "2", "3", "4"])   # values untouched


class TooltipSpecTest(unittest.TestCase):
    """Tooltips: "description (shortcut)" by default; shown on hover (base_window draws them)."""

    def test_confirm_shows_its_shortcut(self):
        (b,) = button_row([widgets.CONFIRM], 100)
        self.assertEqual((b.text, b.tooltip), ("Confirm", "Confirm (Enter)"))

    def test_icon_button_with_a_shortcut(self):
        (b,) = button_row([("Back one frame", "back", (widgets.KEY_LEFT,), "prev_frame")], 100)
        self.assertEqual(b.tooltip, "Back one frame (Left arrow)")

    def test_without_a_shortcut(self):
        text, icon = button_row([("OK", "ok", ()), ("Play", "play", (), "play")], 100)
        self.assertIsNone(text.tooltip)                  # the label says it all
        self.assertEqual(icon.tooltip, "Play")           # an icon needs its description

    def test_explicit_tooltip_wins(self):
        (b,) = button_row([widgets.ButtonSpec("OK", "ok", (13,), None, "Custom")], 100)
        self.assertEqual(b.tooltip, "Custom")

    def test_key_names(self):
        self.assertEqual([widgets.key_name(k) for k in (13, 10, 27, 32, 8, 127, widgets.KEY_LEFT,
                                                        widgets.KEY_RIGHT, ord("m"), ord("["), ord("5"))],
                         ["Enter", "Enter", "Esc", "Space", "Backspace", "Backspace", "Left arrow",
                          "Right arrow", "M", "[", "5"])
        self.assertEqual(widgets.key_name(5000), "key 5000")

    def test_shortcut_text_merges_codes_for_one_key(self):
        self.assertEqual(widgets.shortcut_text((ord("m"), ord("M"))), "M")
        self.assertEqual(widgets.shortcut_text(widgets.KEY_ENTER + (widgets.KEY_ESC,)), "Enter / Esc")
        self.assertEqual(widgets.shortcut_text(()), "")

    def test_icon_button_with_its_own_tooltip(self):
        (b,) = button_row([widgets.ButtonSpec("Play", "play", (), "play", "Play (Space)")], 100)
        self.assertEqual(b.tooltip, "Play (Space)")
