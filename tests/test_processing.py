"""Tests for processing a folder: finding videos, reusing analysis files, the pose re-check (tmwt/detection/processing.py)."""

import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from tmwt.core import analysis_file
from tmwt.core.job import STATUS_FAILED, VideoJob
from tmwt.detection import analysis, pose_check, processing


def touch(path, data=b"video bytes"):
    with open(path, "wb") as f:
        f.write(data)
    return path


class FindVideosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_videos_sorted_and_case_insensitive(self):
        for name in ("b.mp4", "a.MOV", "c.avi", "d.MkV", "e.wmv", "f.m4v", "notes.txt", "g.mp4.npz", "noext"):
            touch(os.path.join(self.dir, name))
        os.mkdir(os.path.join(self.dir, "tmwt_analysis"))
        touch(os.path.join(self.dir, "tmwt_analysis", "nested.mp4"))
        found = [os.path.basename(p) for p in processing.find_videos(self.dir)]
        self.assertEqual(found, ["a.MOV", "b.mp4", "c.avi", "d.MkV", "e.wmv", "f.m4v"])
        self.assertTrue(all(os.path.dirname(p) == self.dir for p in processing.find_videos(self.dir)))

    def test_empty_folder(self):
        self.assertEqual(processing.find_videos(self.dir), [])

    # A sub-folder named like a video isn't a video.
    def test_folder_named_like_a_video_is_ignored(self):
        os.mkdir(os.path.join(self.dir, "clips.mp4"))
        touch(os.path.join(self.dir, "real.mp4"))
        found = [os.path.basename(p) for p in processing.find_videos(self.dir)]
        self.assertEqual(found, ["real.mp4"])


class NeedsProcessingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = touch(os.path.join(self.tmp.name, "walk.mp4"), b"x" * 5000)
        self.wanted = processing.settings("rtmlib", "balanced", True)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, status="", error="", provenance=None):
        job = VideoJob(path=self.video, output_path="", name="walk.mp4")
        job.status, job.error = status, error
        analysis_file.save(job, dict(self.wanted if provenance is None else provenance))

    def test_settings(self):
        self.assertEqual(self.wanted, {"backend": "rtmlib", "model": "balanced",
                                       "matte_crop": True, "max_people": processing.people.MAX_PEOPLE})

    def test_not_processed(self):
        self.assertEqual(processing.needs_processing(self.video, self.wanted), (True, "not processed yet"))

    def test_reusable(self):
        self.save()
        self.assertEqual(processing.needs_processing(self.video, self.wanted), (False, ""))

    def test_unreadable_file_counts_as_not_processed(self):
        path = analysis_file.analysis_path(self.video)
        os.makedirs(os.path.dirname(path))
        touch(path, b"not an npz")
        self.assertEqual(processing.needs_processing(self.video, self.wanted), (True, "not processed yet"))

    def test_older_format(self):
        path = analysis_file.analysis_path(self.video)
        os.makedirs(os.path.dirname(path))
        np.savez_compressed(path, meta=np.array(json.dumps({"format_version": 0})))
        self.assertEqual(processing.needs_processing(self.video, self.wanted),
                         (True, "older analysis file format"))

    def test_failed(self):
        self.save(status=STATUS_FAILED, error="video is entirely black")
        needed, why = processing.needs_processing(self.video, self.wanted)
        self.assertTrue(needed)
        self.assertEqual(why, "previous attempt failed (video is entirely black)")

    def test_different_settings(self):
        self.save()
        for other in (processing.settings("rtmlib", "performance", True),
                      processing.settings("mediapipe", "balanced", True),
                      processing.settings("rtmlib", "balanced", False)):
            self.assertEqual(processing.needs_processing(self.video, other),
                             (True, "processed with different settings"))

    def test_video_changed(self):
        self.save()
        touch(self.video, b"y" * 5000)                  # same size, different content
        self.assertEqual(processing.needs_processing(self.video, self.wanted), (True, "video has changed"))
        touch(self.video, b"x" * 6000)                  # different size
        self.assertEqual(processing.needs_processing(self.video, self.wanted), (True, "video has changed"))


def flag(frame=30, landmark=33):
    return pose_check.Flag(frame, frame / 25.0, landmark, pose_check.FOOT_LENGTH)


class FakeBackend:
    """A pose backend with a heavier model and nothing else; its models are never loaded."""
    HEAVIER = {"balanced": "performance"}

    @staticmethod
    def heavier_model(mode):
        return FakeBackend.HEAVIER.get(mode)


class CheckPoseTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.retry_flags = []
        self.retry_fails = False
        patches = [
            mock.patch.object(processing, "load_pose_model", self.fake_load),
            mock.patch.object(analysis, "process_video", self.fake_process),
            mock.patch.object(analysis, "choose_subject", lambda job: object()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_load(self, backend, backend_name, model_path, ui=None):
        self.calls.append(("load", model_path))

    def fake_process(self, job, model_path, backend, matte_crop=True, on_progress=None):
        self.calls.append(("process", model_path))
        if self.retry_fails:
            job.status, job.error = STATUS_FAILED, "boom"
        else:
            job.pose_flags = list(self.retry_flags)

    def check(self, flags, backend=FakeBackend, model="balanced"):
        job = VideoJob(path="walk.mp4", output_path="", name="walk.mp4")
        job.pose_flags = flags
        with contextlib.redirect_stdout(io.StringIO()):
            out, meta = processing._check_pose(job, backend, "fake", model, True, None, 0, "1 of 1")
        return job, out, meta

    def test_no_flags(self):
        job, out, meta = self.check([])
        self.assertIs(out, job)
        self.assertEqual(self.calls, [])
        self.assertEqual(meta["model_strength"], "balanced")
        self.assertEqual(meta["pose_check"]["result"], "ok")
        self.assertEqual(len(meta["pose_check"]["runs"]), 1)
        self.assertEqual(meta["pose_check"]["runs"][0]["flagged_frames"], 0)

    def test_heavier_model_fixes_it(self):
        job, out, meta = self.check([flag()])
        self.assertIsNot(out, job)
        self.assertEqual(self.calls, [("load", "performance"), ("process", "performance")])
        self.assertEqual(meta["model_strength"], "performance")
        self.assertEqual(meta["pose_check"]["result"], "fixed by heavier model")
        runs = meta["pose_check"]["runs"]
        self.assertEqual([r["model_strength"] for r in runs], ["balanced", "performance"])
        self.assertEqual([r["flagged_frames"] for r in runs], [1, 0])

    def test_anomalies_remain_with_the_heavier_model(self):
        self.retry_flags = [flag(40, 31), flag(41, 31)]
        job, out, meta = self.check([flag()])
        self.assertIsNot(out, job)
        self.assertEqual(out.pose_flags, self.retry_flags)          # the heavier result is kept
        self.assertEqual(meta["model_strength"], "performance")
        self.assertEqual(meta["pose_check"]["result"], "anomalies remain")
        self.assertEqual(meta["pose_check"]["runs"][1]["flagged_frames"], 2)

    def test_no_heavier_model(self):
        job, out, meta = self.check([flag()], model="performance")
        self.assertIs(out, job)
        self.assertEqual(self.calls, [])
        self.assertEqual(meta["model_strength"], "performance")
        self.assertEqual(meta["pose_check"]["result"], "anomalies remain (no heavier model)")

    def test_backend_without_heavier_models(self):
        job, out, meta = self.check([flag()], backend=types.SimpleNamespace(),
                                    model="models/pose_landmarker_full.task")
        self.assertIs(out, job)
        self.assertEqual(meta["model_strength"], "pose_landmarker_full")
        self.assertEqual(meta["pose_check"]["result"], "anomalies remain (no heavier model)")

    def test_heavier_model_failing_keeps_the_first_result(self):
        self.retry_fails = True
        job, out, meta = self.check([flag()])
        self.assertIs(out, job)
        self.assertEqual(meta["model_strength"], "balanced")
        self.assertEqual(meta["pose_check"]["result"], "anomalies remain (heavier model failed: boom)")
        self.assertEqual(len(meta["pose_check"]["runs"]), 1)

    def test_nobody_seen_means_nothing_to_check(self):
        with mock.patch.object(analysis, "choose_subject", lambda job: None):
            job, out, meta = self.check([flag()])
        self.assertIs(out, job)
        self.assertEqual(meta["pose_check"]["result"], "ok")
        self.assertEqual(self.calls, [])

    # A heavier run that finds nobody isn't a fix: the first result is kept.
    def test_heavier_model_seeing_nobody_is_not_a_fix(self):
        seen = {"n": 0}

        def subject_first_run_only(job):
            seen["n"] += 1
            return object() if seen["n"] == 1 else None
        with mock.patch.object(analysis, "choose_subject", subject_first_run_only):
            job, out, meta = self.check([flag()])
        self.assertEqual(meta["pose_check"]["result"], "anomalies remain (heavier model found nobody)")
        self.assertIs(out, job)

    def test_failed_job_is_not_checked(self):
        job = VideoJob(path="walk.mp4", output_path="", name="walk.mp4", status=STATUS_FAILED)
        out, meta = processing._check_pose(job, FakeBackend, "fake", "balanced", True, None, 0, "")
        self.assertIs(out, job)
        self.assertEqual(meta, {})
        self.assertEqual(self.calls, [])


class ConsoleProgressTest(unittest.TestCase):
    def test_every_ten_percent_once(self):
        cb = processing._progress_callback(None, 0, "t", "s")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for k in range(1, 31):
                cb(k / 30, None, [])
            cb(1.0, None, [])
        self.assertEqual([int(line.strip()[:-1]) for line in out.getvalue().split()],
                         list(range(10, 101, 10)))


if __name__ == "__main__":
    unittest.main()
