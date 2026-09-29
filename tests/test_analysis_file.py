"""Tests for the analysis file format: save, load, fingerprint (tmwt/core/analysis_file.py)."""

import contextlib
import io
import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from tmwt.core import analysis_file as af
from tmwt.core import video_io
from tmwt.core.job import STATUS_FAILED, STATUS_OK, STATUS_PENDING, FrameResult, VideoJob
from tmwt.pose import pose_common as pc
from tmwt.pose.pose_common import Landmark

W, H, FPS, N = 64, 48, 10, 20


def write_video(path, n=N, black=0, seed=0):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for k in range(n):
        if k < black:
            img = np.zeros((H, W, 3), np.uint8)
        else:
            small = rng.integers(40, 220, (H // 8, W // 8, 3)).astype(np.uint8)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        writer.write(img)
    writer.release()


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


PROVENANCE = {"backend": "rtmlib", "model": "balanced", "device": "cpu", "max_people": 3,
              "model_strength": "performance",
              "pose_check": {"result": "fixed by heavier model",
                             "runs": [{"model_strength": "balanced", "flagged_frames": 2}]}}


class AnalysisFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self.tmp.name, "walk.mp4")
        write_video(self.video, black=2)

    def tearDown(self):
        self.tmp.cleanup()

    def processed_job(self):
        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        job.info = quiet(video_io.probe_video, self.video)
        job.aruco_finish = (40, 30)
        pose_a = [Landmark(0.01 * i, 0.02 * i, -0.5 * i, 0.9) for i in range(pc.NUM_LANDMARKS)]
        pose_a[3] = None
        pose_a[5].visibility = None                    # no visibility: stored as 1.0
        pose_b = [None] * pc.NUM_LANDMARKS
        pose_b[27] = Landmark(0.3, 0.7, 0.0, 0.0)      # visibility 0 is kept, not turned into 1
        Hm = np.array([[1.0, 0.1, 2.0], [0.0, 1.0, -3.0], [1e-4, 0.0, 1.0]])
        job.frames = [
            FrameResult(0, 0.2, Hm, people=[pose_a, pose_b]),
            FrameResult(1, 0.3, None, people=[]),       # tracking lost, nobody seen
            FrameResult(2, 0.4, np.eye(3), people=[pose_b]),
        ]
        return job

    def test_analysis_path(self):
        self.assertEqual(af.analysis_path(self.video),
                         os.path.join(self.tmp.name, af.ANALYSIS_DIR, "walk.mp4.npz"))

    def test_round_trip(self):
        saved = self.processed_job()
        path = af.save(saved, PROVENANCE)
        self.assertEqual(path, af.analysis_path(self.video))
        self.assertEqual(os.listdir(os.path.dirname(path)), ["walk.mp4.npz"])   # no .tmp left

        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        quiet(af.load, job)
        self.assertEqual(job.status, STATUS_PENDING)
        self.assertEqual(job.info.first_frame_idx, 2)
        self.assertIsNone(job.info.crop)
        self.assertEqual(job.info.fps, saved.info.fps)
        self.assertEqual(job.info.total_frames, N)
        self.assertEqual(job.info.first_frame.shape, (H, W, 3))
        self.assertEqual(job.aruco_finish, (40, 30))

        self.assertEqual(len(job.frames), 3)
        self.assertEqual([f.frame_idx for f in job.frames], [0, 1, 2])
        self.assertEqual([f.time_s for f in job.frames], [0.2, 0.3, 0.4])
        np.testing.assert_array_equal(job.frames[0].H, saved.frames[0].H)
        self.assertIsNone(job.frames[1].H)
        np.testing.assert_array_equal(job.frames[2].H, np.eye(3))
        self.assertEqual([len(f.people) for f in job.frames], [2, 0, 1])

        a, b = job.frames[0].people
        self.assertEqual(len(a), pc.NUM_LANDMARKS)
        self.assertIsNone(a[3])
        self.assertEqual((a[10].x, a[10].y, a[10].z, a[10].visibility), (0.1, 0.2, -5.0, 0.9))
        self.assertEqual(a[5].visibility, 1.0)
        self.assertEqual(sum(lm is not None for lm in b), 1)
        self.assertEqual((b[27].x, b[27].y, b[27].visibility), (0.3, 0.7, 0.0))

        meta = job.analysis_meta
        self.assertEqual(meta["format_version"], af.FORMAT_VERSION)
        self.assertEqual(meta["video"], "walk.mp4")
        self.assertEqual(meta["status"], "processed")
        self.assertEqual((meta["frame_w"], meta["frame_h"]), (W, H))
        for key, value in PROVENANCE.items():
            self.assertEqual(meta[key], value)
        self.assertEqual(job.model_strength, "performance")
        self.assertEqual(af.read_meta(self.video), meta)

    def test_round_trip_no_frames(self):
        saved = self.processed_job()
        saved.frames = []
        saved.aruco_finish = None
        af.save(saved, {})
        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        quiet(af.load, job)
        self.assertEqual(job.frames, [])
        self.assertIsNone(job.aruco_finish)

    def test_round_trip_with_crop(self):
        saved = self.processed_job()
        saved.info.crop = (0, 8, W, 32)
        saved.info.first_frame = saved.info.first_frame[8:40]
        af.save(saved, {})
        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        quiet(af.load, job)
        self.assertEqual(job.info.crop, (0, 8, W, 32))
        self.assertEqual(job.info.first_frame.shape, (32, W, 3))

    def test_failed_job(self):
        saved = VideoJob(path=self.video, output_path="", name="walk.mp4",
                         status=STATUS_FAILED, error="all frames are black")
        af.save(saved, {"model": "balanced"})
        meta = af.read_meta(self.video)
        self.assertEqual(meta["status"], STATUS_FAILED)
        self.assertIsNone(meta["fps"])
        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        af.load(job)
        self.assertEqual((job.status, job.error), (STATUS_FAILED, "all frames are black"))
        self.assertIsNone(job.info)
        self.assertEqual(job.frames, [])

    def test_save_overwrites(self):
        job = self.processed_job()
        af.save(job, {"model": "first"})
        job.status = STATUS_OK
        af.save(job, {"model": "second"})
        self.assertEqual(af.read_meta(self.video)["model"], "second")
        self.assertEqual(os.listdir(os.path.dirname(af.analysis_path(self.video))), ["walk.mp4.npz"])

    def test_read_meta_missing(self):
        self.assertIsNone(af.read_meta(self.video))

    def test_read_meta_garbage(self):
        path = af.analysis_path(self.video)
        os.makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            f.write(b"not an npz file at all")
        self.assertIsNone(af.read_meta(self.video))

    def test_read_meta_no_meta_entry_or_bad_json(self):
        path = af.analysis_path(self.video)
        os.makedirs(os.path.dirname(path))
        np.savez_compressed(path, times=np.zeros(3))
        self.assertIsNone(af.read_meta(self.video))
        np.savez_compressed(path, meta=np.array("{not json"))
        self.assertIsNone(af.read_meta(self.video))

    # A truncated .npz (e.g. an interrupted copy from the cluster).
    def test_read_meta_truncated_file(self):
        af.save(self.processed_job(), {})
        path = af.analysis_path(self.video)
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data[:len(data) // 2])
        self.assertIsNone(af.read_meta(self.video))

    # review_session.load_jobs only catches AnalysisFileError, so a damaged
    # file must raise that (not zipfile.BadZipFile) to not end the review.
    def test_load_truncated_file_raises_analysis_file_error(self):
        af.save(self.processed_job(), {})
        path = af.analysis_path(self.video)
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data[:len(data) // 2])
        with self.assertRaises(af.AnalysisFileError):
            af.load(VideoJob(path=self.video, output_path="", name="walk.mp4"))

    def test_load_missing_file(self):
        with self.assertRaisesRegex(af.AnalysisFileError, "not processed"):
            af.load(VideoJob(path=self.video, output_path="", name="walk.mp4"))

    def test_load_refuses_changed_video(self):
        af.save(self.processed_job(), {})
        write_video(self.video, black=2, seed=1)           # a different video, same name
        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        with self.assertRaisesRegex(af.AnalysisFileError, "differs"):
            af.load(job)
        self.assertEqual(job.frames, [])
        self.assertEqual(job.analysis_meta, {})

    def test_load_refuses_other_format_version(self):
        af.save(self.processed_job(), {})
        path = af.analysis_path(self.video)
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        meta = json.loads(str(arrays["meta"]))
        for version in (af.FORMAT_VERSION + 1, None):
            meta["format_version"] = version
            arrays["meta"] = np.array(json.dumps(meta))
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(af.AnalysisFileError, "not supported"):
                af.load(VideoJob(path=self.video, output_path="", name="walk.mp4"))

    def test_load_refuses_other_frame_size(self):
        job = self.processed_job()
        job.info.first_frame = np.zeros((H // 2, W, 3), np.uint8)
        af.save(job, {})
        with self.assertRaisesRegex(af.AnalysisFileError, "frame size"):
            quiet(af.load, VideoJob(path=self.video, output_path="", name="walk.mp4"))

    def test_fingerprint(self):
        fp = af.fingerprint(self.video)
        self.assertEqual(fp, af.fingerprint(self.video))           # deterministic
        with open(self.video, "ab") as f:                           # size changes
            f.write(b"\0")
        self.assertNotEqual(af.fingerprint(self.video), fp)

    def test_fingerprint_same_size_different_content(self):
        a, b = (os.path.join(self.tmp.name, n) for n in ("a.bin", "b.bin"))
        with open(a, "wb") as f:
            f.write(b"\1" * 100)
        with open(b, "wb") as f:
            f.write(b"\1" * 99 + b"\2")
        self.assertNotEqual(af.fingerprint(a), af.fingerprint(b))

    def test_fingerprint_large_file_uses_both_ends(self):
        chunk = af._FINGERPRINT_CHUNK
        path = os.path.join(self.tmp.name, "big.bin")
        data = bytearray(2 * chunk + 1000)
        with open(path, "wb") as f:
            f.write(data)
        fp = af.fingerprint(path)
        data[-1] = 7                                                  # last MiB changes
        with open(path, "wb") as f:
            f.write(data)
        self.assertNotEqual(af.fingerprint(path), fp)
        fp = af.fingerprint(path)
        data[chunk + 10] = 7                                          # middle: not hashed
        with open(path, "wb") as f:
            f.write(data)
        self.assertEqual(af.fingerprint(path), fp)


if __name__ == "__main__":
    unittest.main()


class DamagedFileTest(unittest.TestCase):
    """Files that aren't analysis files at all."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = os.path.join(self.tmp.name, "v.mp4")
        with open(self.video, "wb") as f:
            f.write(b"not really a video")
        self.path = af.analysis_path(self.video)
        os.makedirs(os.path.dirname(self.path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_plain_npy_array_is_not_an_analysis_file(self):
        with open(self.path, "wb") as f:
            np.save(f, np.zeros(3))
        self.assertIsNone(af.read_meta(self.video))

    def test_random_bytes(self):
        with open(self.path, "wb") as f:
            f.write(os.urandom(200))
        self.assertIsNone(af.read_meta(self.video))
