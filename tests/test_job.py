"""Tests for the job / frame data model (tmwt/core/job.py)."""

import os
import unittest

from tmwt.core import job as job_mod
from tmwt.core.job import FrameResult, VideoJob
from tmwt.detection.pose_check import Flag


def make_job(**kw):
    return VideoJob(path="/videos/walk.mp4", output_path="/out/walk.csv", name="walk.mp4", **kw)


class DurationSpeedTest(unittest.TestCase):
    def test_duration_and_speed(self):
        job = make_job(walk_start=1.5, walk_end=6.5)
        self.assertAlmostEqual(job.duration, 5.0)
        self.assertAlmostEqual(job.speed, job_mod.COURSE_M / 5.0)

    def test_missing_start_or_end_gives_none(self):
        for start, end in ((None, 5.0), (1.0, None), (None, None)):
            job = make_job(walk_start=start, walk_end=end)
            self.assertIsNone(job.duration)
            self.assertIsNone(job.speed)

    def test_start_at_zero_is_not_missing(self):
        job = make_job(walk_start=0.0, walk_end=4.0)
        self.assertAlmostEqual(job.duration, 4.0)
        self.assertAlmostEqual(job.speed, 2.5)

    def test_zero_duration_has_no_speed(self):
        job = make_job(walk_start=3.0, walk_end=3.0)
        self.assertEqual(job.duration, 0.0)
        self.assertIsNone(job.speed)


class TrackTest(unittest.TestCase):
    def test_only_frames_with_ref_foot(self):
        frames = [FrameResult(k, k / 10, None) for k in range(4)]
        frames[1].ref_foot = (10, 20)
        frames[3].ref_foot = (0, 0)          # a falsy-looking position still counts
        job = make_job(frames=frames)
        self.assertEqual([f.frame_idx for f in job.track], [1, 3])

    def test_empty(self):
        self.assertEqual(make_job().track, [])


class ModelStrengthTest(unittest.TestCase):
    def test_prefers_model_strength(self):
        job = make_job(analysis_meta={"model_strength": "performance", "model": "m.onnx"})
        self.assertEqual(job.model_strength, "performance")

    def test_falls_back_to_model(self):
        self.assertEqual(make_job(analysis_meta={"model": "m.onnx"}).model_strength, "m.onnx")
        self.assertEqual(make_job(analysis_meta={"model_strength": "", "model": "m.onnx"}).model_strength,
                         "m.onnx")

    def test_unknown_without_meta(self):
        self.assertEqual(make_job().model_strength, "unknown")
        self.assertEqual(make_job(analysis_meta={"model_strength": None, "model": None}).model_strength,
                         "unknown")


class PoseFlaggedFramesTest(unittest.TestCase):
    def test_counts_distinct_frames(self):
        flags = [Flag(3, 0.3, 31, "foot_length"), Flag(3, 0.3, 33, "foot_length"),
                 Flag(7, 0.7, 25, "spike")]
        self.assertEqual(make_job(pose_flags=flags).pose_flagged_frames, 2)

    def test_none(self):
        self.assertEqual(make_job().pose_flagged_frames, 0)


class OutputFileTest(unittest.TestCase):
    def test_replaces_extension(self):
        job = make_job()
        self.assertEqual(job.output_file("_annotated.mp4"), "/out/walk_annotated.mp4")
        self.assertEqual(job.output_file(".csv"), job.output_path)

    def test_dots_in_folder_and_name(self):
        job = VideoJob(path="v.mp4", output_path=os.path.join("a.b", "walk.v2.csv"), name="v.mp4")
        self.assertEqual(job.output_file("_timing.json"), os.path.join("a.b", "walk.v2_timing.json"))


class DefaultsTest(unittest.TestCase):
    def test_frame_defaults(self):
        f = FrameResult(0, 0.0, None)
        self.assertEqual(f.people, [])
        self.assertEqual(f.pose_flags, {})
        self.assertEqual(f.pose_smoothed, set())
        self.assertIsNone(f.pose)
        self.assertIsNone(f.t_along)

    def test_frame_containers_are_per_instance(self):
        a, b = FrameResult(0, 0.0, None), FrameResult(1, 0.1, None)
        a.pose_flags[31] = "spike"
        a.pose_smoothed.add(31)
        a.people.append([])
        self.assertEqual(b.pose_flags, {})
        self.assertEqual(b.pose_smoothed, set())
        self.assertEqual(b.people, [])

    def test_job_defaults_are_per_instance(self):
        a, b = make_job(), make_job()
        a.frames.append(FrameResult(0, 0.0, None))
        a.pose_flags.append(Flag(0, 0.0, 31, "spike"))
        a.pose_edits.append(object())
        a.analysis_meta["model"] = "x"
        self.assertEqual((b.frames, b.pose_flags, b.pose_edits, b.analysis_meta), ([], [], [], {}))

    def test_job_initial_state(self):
        job = make_job()
        self.assertEqual(job.status, job_mod.STATUS_PENDING)
        self.assertEqual(job.review, job_mod.REVIEW_UNREVIEWED)
        self.assertEqual(job.endpoint_behavior, job_mod.END_FIRST_FOOT)
        self.assertIn(job.endpoint_behavior, job_mod.END_BEHAVIORS)
        self.assertFalse(job.saved)
        self.assertFalse(job.pose_confirmed)


if __name__ == "__main__":
    unittest.main()
