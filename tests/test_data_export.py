"""Tests for the per-video outputs and reading them back (tmwt/core/data_export.py)."""

import contextlib
import csv
import io
import os
import tempfile
import unittest

from tmwt.core import data_export as de
from tmwt.core.job import FrameResult, VideoJob, REVIEW_APPROVED
from tmwt.detection import pose_smoothing
from tmwt.detection.pose_check import Flag
from tmwt.pose import pose_common as pc
from tmwt.pose.pose_common import Landmark


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def full_pose(offset=0.0):
    return [Landmark(0.01 * i + offset, 0.5 + 0.001 * i, -0.1 * i) for i in range(pc.NUM_LANDMARKS)]


class CsvRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "sub", "walk.csv")   # folder created on write

    def tearDown(self):
        self.tmp.cleanup()

    def frames(self):
        a = FrameResult(0, 0.0, None, pose=full_pose(), body_px=(120, 340),
                        far_ep=(100, 50), near_ep=(110, 400), t_along=0.1234567891)
        a.pose_flags = {33: "foot_length", 25: "spike"}
        a.pose_smoothed = {29, 33}
        b = FrameResult(1, 0.04, None)                         # nothing known in this frame
        partial = [None] * pc.NUM_LANDMARKS
        partial[27] = Landmark(0.4, 0.8, 0.0)
        c = FrameResult(2, 0.123456, None, pose=partial)
        return [a, b, c]

    def test_header(self):
        quiet(de.write_frames_csv, self.path, [], 64, 48)
        with open(self.path, newline="") as f:
            self.assertEqual(next(csv.reader(f)), de.HEADERS)
        self.assertEqual(len(de.HEADERS), len(de.CORE_COLUMNS) + 3 * pc.NUM_LANDMARKS + 2)
        self.assertEqual(de.read_frames_csv(self.path), [])

    def test_round_trip(self):
        quiet(de.write_frames_csv, self.path, self.frames(), 640, 480)
        rows = de.read_frames_csv(self.path)
        self.assertEqual(len(rows), 3)
        a, b, c = rows

        self.assertEqual(a["frame"], 0.0)
        self.assertEqual((a["frame_w"], a["frame_h"]), (640.0, 480.0))
        self.assertEqual(de.row_point(a, "body"), (120, 340))
        self.assertEqual(de.row_point(a, "far_ep"), (100, 50))
        self.assertEqual(de.row_point(a, "near_ep"), (110, 400))
        self.assertAlmostEqual(a["t_along"], 0.123457)
        pose = de.row_pose(a)
        self.assertEqual(len(pose), pc.NUM_LANDMARKS)
        for i, lm in enumerate(full_pose()):
            self.assertAlmostEqual(pose[i].x, lm.x, places=6)
            self.assertAlmostEqual(pose[i].y, lm.y, places=6)
            self.assertAlmostEqual(pose[i].z, lm.z, places=6)
        self.assertEqual(de.row_flags(a), {33: "foot_length", 25: "spike"})
        self.assertEqual(de.row_smoothed(a), {29, 33})
        self.assertEqual(a["pose_flags"], "left_knee:spike;left_small_toe:foot_length")

        # Empty fields read back as None.
        self.assertIsNone(b["body_x"])
        self.assertIsNone(b["t_along"])
        self.assertIsNone(de.row_point(b, "body"))
        self.assertIsNone(de.row_pose(b))
        self.assertEqual(de.row_flags(b), {})
        self.assertEqual(de.row_smoothed(b), set())

        self.assertAlmostEqual(c["time_s"], 0.1235)
        pose = de.row_pose(c)
        self.assertEqual(sum(lm is not None for lm in pose), 1)
        self.assertAlmostEqual(pose[27].x, 0.4)

    def test_short_pose_is_padded(self):
        # A pose with fewer landmarks than the layout (e.g. an older 33-point
        # model) writes the missing ones as empty.
        f = FrameResult(0, 0.0, None, pose=full_pose()[:pc.MEDIAPIPE_LANDMARKS])
        quiet(de.write_frames_csv, self.path, [f], 64, 48)
        pose = de.row_pose(de.read_frames_csv(self.path)[0])
        self.assertIsNotNone(pose[pc.MEDIAPIPE_LANDMARKS - 1])
        self.assertIsNone(pose[pc.NUM_LANDMARKS - 1])


class RowHelpersTest(unittest.TestCase):
    def test_row_point_needs_both(self):
        self.assertIsNone(de.row_point({"body_x": 3.0, "body_y": None}, "body"))
        self.assertIsNone(de.row_point({}, "body"))
        self.assertEqual(de.row_point({"body_x": 0.0, "body_y": 0.0}, "body"), (0, 0))

    def test_older_csv_without_pose_columns(self):
        self.assertEqual(de.row_flags({}), {})
        self.assertEqual(de.row_smoothed({}), set())
        self.assertIsNone(de.row_pose({}))

    def test_row_pose_missing_z_defaults_to_zero(self):
        pose = de.row_pose({"lm_00_x": 0.1, "lm_00_y": 0.2, "lm_00_z": None})
        self.assertEqual((pose[0].x, pose[0].y, pose[0].z), (0.1, 0.2, 0.0))
        self.assertTrue(all(lm is None for lm in pose[1:]))

    def test_unknown_names_are_ignored(self):
        self.assertEqual(de.row_flags({"pose_flags": "nose:spike;left_heel:spike"}), {29: "spike"})
        self.assertEqual(de.row_smoothed({"pose_smoothed": "bogus;right_heel"}), {30})

    def test_output_paths(self):
        timing, annotated, skeleton = de.output_paths(os.path.join("out", "walk.csv"))
        self.assertEqual(timing, os.path.join("out", "walk_timing.json"))
        self.assertEqual(annotated, os.path.join("out", "walk_annotated.mp4"))
        self.assertEqual(skeleton, os.path.join("out", "walk_skeleton.mp4"))


class JobOutputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.job = VideoJob(path=os.path.join(self.tmp.name, "walk.mp4"),
                            output_path=os.path.join(self.tmp.name, "walk.csv"), name="walk.mp4")

    def tearDown(self):
        self.tmp.cleanup()

    def timing(self):
        de.write_timing(self.job)
        data = de.read_timing(self.job.output_path)
        self.assertIsNotNone(data)
        return data

    def test_read_timing_missing(self):
        self.assertIsNone(de.read_timing(self.job.output_path))

    def test_write_timing(self):
        job = self.job
        job.walk_start, job.walk_end = 1.0, 6.0
        job.endpoint_source, job.timing_source, job.timing_detail = "auto", "auto", "onset"
        job.timing_note = "start uncertain"
        job.review = REVIEW_APPROVED
        job.analysis_meta = {"model_strength": "performance",
                             "pose_check": {"result": "anomalies remain", "runs": []}}
        job.pose_flags = [Flag(2, 0.08, 33, "foot_length"), Flag(2, 0.08, 31, "foot_length"),
                          Flag(5, 0.2, 25, "spike")]
        job.pose_confirmed = True
        data = self.timing()
        self.assertEqual(data["video"], "walk.mp4")
        self.assertEqual(data["course_m"], 10.0)
        self.assertEqual((data["walk_start_s"], data["walk_end_s"]), (1.0, 6.0))
        self.assertEqual(data["duration_s"], 5.0)
        self.assertEqual(data["speed_mps"], 2.0)
        self.assertEqual(data["timing_note"], "start uncertain")
        self.assertEqual(data["review"], REVIEW_APPROVED)
        self.assertEqual(data["model_strength"], "performance")
        pc_ = data["pose_check"]
        self.assertEqual(pc_["result"], "2 frame(s) flagged; confirmed fine at review")
        self.assertEqual(pc_["processing"], "anomalies remain")
        self.assertEqual(pc_["flagged_frames"], 2)
        self.assertEqual(pc_["flagged_points"], 3)
        self.assertEqual(pc_["examples"][0], {"time_s": 0.08, "landmark": "left_small_toe",
                                              "kind": "foot_length"})
        self.assertEqual((pc_["smoothed_points"], pc_["smoothed_frames"], pc_["smoothing_method"]),
                         (0, 0, ""))

    def test_timing_note_only_for_auto_timing(self):
        self.job.timing_note = "start uncertain"
        self.job.timing_source = "manual"
        self.assertEqual(self.timing()["timing_note"], "")
        self.job.timing_source = ""
        self.assertEqual(self.timing()["timing_note"], "")

    def test_timing_without_meta_or_times(self):
        data = self.timing()
        self.assertIsNone(data["duration_s"])
        self.assertIsNone(data["speed_mps"])
        self.assertEqual(data["model_strength"], "unknown")
        self.assertEqual(data["pose_check"]["result"], "")
        self.assertEqual(data["pose_check"]["processing"], "")
        self.assertEqual(data["pose_check"]["flagged_frames"], 0)

    def test_timing_with_smoothing(self):
        job = self.job
        job.frames = [FrameResult(k, k * 0.04, None) for k in range(4)]
        job.analysis_meta = {"model": "m"}
        job.pose_flags = [Flag(1, 0.04, 33, "foot_length"), Flag(1, 0.04, 31, "foot_length"),
                          Flag(3, 0.12, 33, "spike")]
        job.pose_edits = [pose_smoothing.Edit(1, 33, "foot_length", Landmark(0, 0), Landmark(1, 1)),
                          pose_smoothing.Edit(1, 31, "foot_length", Landmark(0, 0), Landmark(1, 1)),
                          pose_smoothing.Edit(3, 33, "spike", Landmark(0, 0), Landmark(1, 1))]
        pc_ = self.timing()["pose_check"]
        self.assertEqual((pc_["smoothed_points"], pc_["smoothed_frames"]), (3, 2))
        self.assertEqual(pc_["smoothing_method"], pose_smoothing.METHOD)
        self.assertIn("3 point(s) smoothed", pc_["result"])

    def test_pose_corrections(self):
        job = self.job
        job.frames = [FrameResult(k + 10, k * 0.04, None) for k in range(4)]
        # Made out of order; written sorted by (frame, landmark).
        job.pose_edits = [
            pose_smoothing.Edit(3, 31, "spike", Landmark(0.5, 0.6, None), Landmark(0.55, 0.65, 0.1)),
            pose_smoothing.Edit(1, 33, "foot_length", Landmark(0.1, 0.2, 0.3), Landmark(0.15, 0.25, 0.35)),
            pose_smoothing.Edit(1, 29, "foot_length", Landmark(0.1, 0.2, 0.3), Landmark(0.1, 0.2, 0.3)),
        ]
        quiet(de.write_pose_corrections, job)
        path = job.output_file(de.CORRECTIONS_SUFFIX)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, de.CORRECTIONS_COLUMNS)
            rows = list(reader)
        self.assertEqual([(r["frame"], r["landmark"]) for r in rows],
                         [("11", "left_heel"), ("11", "left_small_toe"), ("13", "left_big_toe")])
        self.assertEqual(rows[1]["flag"], "foot_length")
        self.assertEqual(rows[1]["time_s"], "0.04")
        self.assertEqual((float(rows[1]["smoothed_x"]), float(rows[1]["smoothed_y"])), (0.15, 0.25))
        self.assertEqual(float(rows[2]["original_z"]), 0.0)          # None z written as 0

    def test_pose_corrections_removed_when_none(self):
        path = self.job.output_file(de.CORRECTIONS_SUFFIX)
        de.write_pose_corrections(self.job)                         # nothing, nothing to remove
        self.assertFalse(os.path.exists(path))
        with open(path, "w") as f:
            f.write("stale")
        de.write_pose_corrections(self.job)
        self.assertFalse(os.path.exists(path))

    def test_remove_outputs(self):
        job = self.job
        job.saved = True
        paths = [job.output_path, *de.output_paths(job.output_path), job.output_file(de.CORRECTIONS_SUFFIX)]
        for p in paths[:3]:                                          # some outputs missing
            with open(p, "w") as f:
                f.write("x")
        keep = os.path.join(self.tmp.name, "other.csv")
        with open(keep, "w") as f:
            f.write("x")
        quiet(de.remove_outputs, job)
        self.assertFalse(any(os.path.exists(p) for p in paths))
        self.assertTrue(os.path.exists(keep))
        self.assertFalse(job.saved)
        quiet(de.remove_outputs, job)                                # nothing left: still fine
        self.assertFalse(job.saved)


if __name__ == "__main__":
    unittest.main()
