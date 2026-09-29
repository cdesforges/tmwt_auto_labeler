"""Tests for letterbox / pillarbox detection and cropping (tmwt/core/matte.py)."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from tmwt.core import matte

W, H = 64, 48


def picture(h=H, w=W, seed=0):
    """Photographic-ish content: random 4x4 blocks, no uniform rows or columns."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (h // 4 + 1, w // 4 + 1, 3)).astype(np.uint8)
    return cv2.resize(small, (w // 4 * 4 + 4, h // 4 * 4 + 4), interpolation=cv2.INTER_NEAREST)[:h, :w]


def matted(top=0, bottom=0, left=0, right=0, color=(0, 0, 0), h=H, w=W):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    img[top:h - bottom, left:w - right] = picture(h, w)[top:h - bottom, left:w - right]
    return img


class DetectContentCropTest(unittest.TestCase):
    def test_no_matte(self):
        crop = matte.detect_content_crop(picture())
        self.assertEqual(crop, (0, 0, W, H))
        self.assertTrue(matte.is_full_frame(crop, (H, W, 3)))

    def test_letterbox(self):
        self.assertEqual(matte.detect_content_crop(matted(top=8, bottom=8)), (0, 8, W, 32))

    def test_pillarbox(self):
        self.assertEqual(matte.detect_content_crop(matted(left=16, right=16)), (16, 0, 32, H))

    def test_uneven_bars_on_all_sides(self):
        self.assertEqual(matte.detect_content_crop(matted(top=4, bottom=6, left=10, right=2)),
                         (10, 4, 52, 38))

    def test_grey_and_white_mattes(self):
        for color in ((128, 128, 128), (255, 255, 255)):
            self.assertEqual(matte.detect_content_crop(matted(top=6, bottom=6, color=color)),
                             (0, 6, W, 36), color)

    def test_odd_size_forced_even(self):
        x, y, w, h = matte.detect_content_crop(matted(top=5, bottom=6))
        self.assertEqual((x, y), (0, 5))
        self.assertEqual(h, 36)                                  # 37 rounded down to even
        self.assertEqual(w % 2, 0)

    def test_noise_in_bar_tolerated(self):
        img = matted(top=8, bottom=8)
        img[2, :3] = 255                                         # a few ringing pixels
        img[6, 10] = 200
        self.assertEqual(matte.detect_content_crop(img), (0, 8, W, 32))

    def test_colour_change_inside_bar(self):
        # A grey border strip inside a black bar: the scan crosses it.
        img = matted(top=10, bottom=0)
        img[6:10] = (90, 90, 90)
        self.assertEqual(matte.detect_content_crop(img)[1], 10)

    def test_crop_capped_per_side(self):
        # A bar covering most of the frame is only trimmed up to max_crop_frac.
        img = matted(top=30)
        x, y, w, h = matte.detect_content_crop(img)
        self.assertEqual(y, int(H * matte.DEFAULT_MAX_CROP_FRAC))

    def test_uniform_frame_falls_back_to_full(self):
        for img in (np.zeros((H, W, 3), np.uint8), np.full((H + 1, W + 1, 3), 77, np.uint8)):
            h, w = img.shape[:2]
            self.assertEqual(matte.detect_content_crop(img), (0, 0, w - w % 2, h - h % 2))

    def test_scan_edge_empty(self):
        self.assertEqual(matte._scan_edge(np.zeros((0, 5, 3)), 30.0, 0.9, 10), 0)


class IsFullFrameTest(unittest.TestCase):
    def test_cases(self):
        shape = (H, W, 3)
        self.assertTrue(matte.is_full_frame((0, 0, W, H), shape))
        self.assertTrue(matte.is_full_frame((0, 0, W - 1, H - 1), shape))   # even rounding of odd sizes
        self.assertFalse(matte.is_full_frame((0, 0, W - 2, H), shape))
        self.assertFalse(matte.is_full_frame((0, 1, W, H - 1), shape))
        self.assertFalse(matte.is_full_frame((2, 0, W - 2, H), shape))


class FakeCapture:
    def __init__(self, frames):
        self.frames, self.released, self.props = list(frames), False, {}

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


class CroppingCaptureTest(unittest.TestCase):
    def test_crops_and_delegates(self):
        frame = picture()
        inner = FakeCapture([frame, frame])
        cap = matte.CroppingCapture(inner, (4, 6, 20, 30))
        ok, out = cap.read()
        self.assertTrue(ok)
        np.testing.assert_array_equal(out, frame[6:36, 4:24])
        self.assertTrue(cap.set(cv2.CAP_PROP_POS_FRAMES, 3))
        self.assertEqual(cap.get(cv2.CAP_PROP_POS_FRAMES), 3)
        self.assertTrue(cap.isOpened())
        cap.read()
        self.assertEqual(cap.read(), (False, None))              # end of video passes through
        cap.release()
        self.assertTrue(inner.released)
        self.assertFalse(cap.isOpened())

    def test_real_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
            for k in range(20):
                writer.write(matted(top=8, bottom=8))
            writer.release()
            cap = matte.CroppingCapture(cv2.VideoCapture(path), (0, 8, W, 32))
            shapes = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                shapes.append(frame.shape)
            cap.release()
        self.assertEqual(shapes, [(32, W, 3)] * 20)


if __name__ == "__main__":
    unittest.main()
