"""Tests for opening videos and in-order frame access (tmwt/core/video_io.py)."""

import contextlib
import io
import os
import tempfile
import unittest

import cv2
import numpy as np

from tmwt.core import video_io
from tmwt.core.job import VideoJob
from tmwt.core.matte import CroppingCapture

W, H, FPS, N = 64, 48, 10, 20


def content_frame(k, rng):
    """A textured frame; frame k is distinguishable from its neighbours."""
    small = rng.integers(40, 220, (H // 8, W // 8, 3)).astype(np.uint8)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, str(k), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def write_video(path, frames, fps=FPS):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def read_all(cap):
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return frames
        frames.append(frame)


def close(a, b, tol=15.0):
    """Same frame content, allowing for JPEG loss (neighbouring frames differ by ~50)."""
    return a.shape == b.shape and np.abs(a.astype(float) - b.astype(float)).mean() < tol


class VideoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.content = [content_frame(k, rng) for k in range(N)]

    def tearDown(self):
        self.tmp.cleanup()

    def video(self, frames, name="v.mp4"):
        path = os.path.join(self.tmp.name, name)
        write_video(path, frames)
        return path


class ProbeVideoTest(VideoTestCase):
    def test_plain_video(self):
        info = quiet(video_io.probe_video, self.video(self.content))
        self.assertEqual(info.first_frame_idx, 0)
        self.assertIsNone(info.crop)
        self.assertAlmostEqual(info.fps, FPS)
        self.assertEqual(info.total_frames, N)
        self.assertEqual(info.first_frame.shape, (H, W, 3))

    def test_leading_black_frames_skipped(self):
        black = [np.zeros((H, W, 3), np.uint8)] * 3
        info = quiet(video_io.probe_video, self.video(black + self.content[:N - 3]))
        self.assertEqual(info.first_frame_idx, 3)
        self.assertGreaterEqual(info.first_frame.mean(), video_io.BLACK_FRAME_MEAN)

    def test_all_black(self):
        path = self.video([np.zeros((H, W, 3), np.uint8)] * 5)
        with self.assertRaisesRegex(video_io.VideoError, "black"):
            quiet(video_io.probe_video, path)

    def test_cannot_open(self):
        with self.assertRaises(video_io.VideoError):
            video_io.probe_video(os.path.join(self.tmp.name, "missing.mp4"))
        junk = os.path.join(self.tmp.name, "junk.mp4")
        with open(junk, "wb") as f:
            f.write(b"not a video")
        with self.assertRaises(video_io.VideoError):
            quiet(video_io.probe_video, junk)

    def test_letterbox(self):
        frames = []
        for img in self.content:
            f = np.zeros_like(img)
            f[8:40] = img[8:40]
            frames.append(f)
        info = quiet(video_io.probe_video, self.video(frames))
        self.assertEqual(info.crop, (0, 8, W, 32))
        self.assertEqual(info.first_frame.shape, (32, W, 3))

    def test_pillarbox(self):
        frames = []
        for img in self.content:
            f = np.zeros_like(img)
            f[:, 16:48] = img[:, 16:48]
            frames.append(f)
        info = quiet(video_io.probe_video, self.video(frames))
        self.assertEqual(info.crop, (16, 0, 32, H))
        self.assertEqual(info.first_frame.shape, (H, 32, 3))

    def test_matte_crop_disabled(self):
        frames = []
        for img in self.content:
            f = np.zeros_like(img)
            f[8:40] = img[8:40]
            frames.append(f)
        info = quiet(video_io.probe_video, self.video(frames), matte_crop=False)
        self.assertIsNone(info.crop)
        self.assertEqual(info.first_frame.shape, (H, W, 3))


class OpenVideoTest(VideoTestCase):
    def test_start_frame(self):
        path = self.video(self.content)
        all_frames = read_all(video_io.open_video(path))
        self.assertEqual(len(all_frames), N)
        cap = video_io.open_video(path, start_frame=5)
        ok, frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        np.testing.assert_array_equal(frame, all_frames[5])

    def test_crop(self):
        path = self.video(self.content)
        full = read_all(video_io.open_video(path))
        cap = video_io.open_video(path, crop=(4, 6, 20, 30), start_frame=2)
        self.assertIsInstance(cap, CroppingCapture)
        self.assertTrue(cap.isOpened())
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), FPS)
        frames = read_all(cap)
        cap.release()
        self.assertEqual(len(frames), N - 2)
        self.assertEqual(frames[0].shape, (30, 20, 3))
        np.testing.assert_array_equal(frames[0], full[2][6:36, 4:24])

    def test_missing_file_reads_nothing(self):
        cap = video_io.open_video(os.path.join(self.tmp.name, "missing.mp4"), crop=(0, 0, 8, 8))
        self.assertFalse(cap.isOpened())
        self.assertEqual(cap.read()[0], False)


class FrameSourceTest(VideoTestCase):
    def job(self, frames, crop=None):
        path = self.video(frames)
        job = VideoJob(path=path, output_path="", name="v.mp4")
        job.info = quiet(video_io.probe_video, path)
        if crop is not None:
            job.info.crop = crop
        return job

    def setUp(self):
        super().setUp()
        black = [np.zeros((H, W, 3), np.uint8)] * 2
        self.frames_job = self.job(black + self.content[:N - 2])
        # What the analysis sees: the video read in order from the first content frame.
        self.expected = read_all(self.frames_job.open_capture())
        self.assertEqual(len(self.expected), N - 2)

    def test_frames_in_order(self):
        src = video_io.FrameSource(self.frames_job)
        for k, want in enumerate(self.expected):
            got = src.get(k)
            np.testing.assert_array_equal(got, want)         # the newest frame is exact
        src.close()

    def test_random_access_matches_in_order(self):
        src = video_io.FrameSource(self.frames_job)
        for k in (10, 3, 0, 10, 11, 17, 5):
            self.assertTrue(close(src.get(k), self.expected[k]), f"frame {k}")
        # Stepping back gives the right frame, not a neighbour.
        self.assertFalse(close(src.get(4), self.expected[5]))
        src.close()

    def test_returned_frame_is_a_copy(self):
        src = video_io.FrameSource(self.frames_job)
        a = src.get(3)
        a[:] = 0
        self.assertTrue(close(src.get(3), self.expected[3]))
        src.close()

    def test_past_the_end(self):
        src = video_io.FrameSource(self.frames_job)
        self.assertIsNone(src.get(len(self.expected)))
        self.assertIsNone(src.get(1000))
        # Earlier frames are still available afterwards.
        self.assertTrue(close(src.get(len(self.expected) - 1), self.expected[-1]))
        self.assertTrue(close(src.get(0), self.expected[0]))
        src.close()

    def test_close(self):
        src = video_io.FrameSource(self.frames_job)
        self.assertIsNotNone(src.get(2))
        src.close()
        self.assertIsNone(src.get(0))
        src.close()                                           # closing twice is fine

    def test_cropped(self):
        job = self.job(self.content, crop=(8, 4, 32, 40))
        src = video_io.FrameSource(job)
        frame = src.get(0)
        self.assertEqual(frame.shape, (40, 32, 3))
        self.assertTrue(close(src.get(6), read_all(job.open_capture())[6]))
        src.close()


if __name__ == "__main__":
    unittest.main()
