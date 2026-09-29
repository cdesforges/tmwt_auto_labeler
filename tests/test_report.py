"""Tests for the labeling and processing reports (tmwt/core/report.py)."""

import contextlib
import csv
import io
import os
import tempfile
import unittest

from tmwt.core import analysis_file, report
from tmwt.core.job import (REVIEW_APPROVED, REVIEW_REJECTED, STATUS_FAILED, STATUS_NO_BODY,
                           STATUS_OK, VideoJob)
from tmwt.detection import pose_smoothing
from tmwt.detection.pose_check import Flag
from tmwt.pose.pose_common import Landmark


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def make_job(name="walk.mp4", **kw):
    defaults = dict(far_ep=(10, 10), near_ep=(10, 400), walk_start=1.0, walk_end=6.0, status=STATUS_OK)
    defaults.update(kw)
    return VideoJob(path=name, output_path=name + ".csv", name=name, **defaults)


FLAGS = [Flag(2, 0.1, 33, "foot_length"), Flag(2, 0.1, 31, "foot_length"), Flag(4, 0.2, 25, "spike")]
EDIT = pose_smoothing.Edit(2, 33, "foot_length", Landmark(0, 0), Landmark(1, 1))


class JobResultTest(unittest.TestCase):
    def test_rejected_wins(self):
        job = make_job(review=REVIEW_REJECTED, review_note="walker left", status=STATUS_FAILED)
        self.assertEqual(report.job_result(job), (report.REJECTED, "walker left"))

    def test_failed_and_no_body(self):
        for status in (STATUS_FAILED, STATUS_NO_BODY):
            job = make_job(status=status, error="boom", review=REVIEW_APPROVED)
            self.assertEqual(report.job_result(job), (report.FAILED, "boom"))

    def test_approved(self):
        job = make_job(review=REVIEW_APPROVED, review_note="fine", far_ep=None, walk_start=None)
        self.assertEqual(report.job_result(job), (report.APPROVED, "fine"))

    def test_no_endpoints(self):
        self.assertEqual(report.job_result(make_job(far_ep=None)), (report.FAILED, "rope endpoints not set"))
        result, reason = report.job_result(make_job(far_ep=None, endpoint_problem="no ArUco marker"))
        self.assertEqual(result, report.FAILED)
        self.assertEqual(reason, "no ArUco marker; rope endpoints need clicking at review")

    def test_missing_timing(self):
        self.assertEqual(report.job_result(make_job(walk_start=None)),
                         (report.FAILED, "no walk start detected"))
        self.assertEqual(report.job_result(make_job(walk_end=None)),
                         (report.FAILED, "no walk end detected"))
        self.assertEqual(report.job_result(make_job(walk_start=None, walk_end=None)),
                         (report.FAILED, "no walk start detected"))

    def test_unreviewed(self):
        self.assertEqual(report.job_result(make_job()), (report.UNREVIEWED, ""))

    def test_zero_duration_is_not_missing(self):
        self.assertEqual(report.job_result(make_job(walk_start=2.0, walk_end=2.0))[0], report.UNREVIEWED)


class PoseCheckTextTest(unittest.TestCase):
    META = {"model": "balanced"}

    def test_no_meta(self):
        self.assertEqual(report.pose_check_text(make_job(pose_flags=list(FLAGS))), "")

    def test_ok(self):
        self.assertEqual(report.pose_check_text(make_job(analysis_meta=self.META)), "ok")

    def test_flagged_not_confirmed(self):
        job = make_job(analysis_meta=self.META, pose_flags=list(FLAGS))
        self.assertEqual(report.pose_check_text(job), "2 frame(s) flagged; not confirmed")

    def test_confirmed(self):
        job = make_job(analysis_meta=self.META, pose_flags=list(FLAGS), pose_confirmed=True)
        self.assertEqual(report.pose_check_text(job), "2 frame(s) flagged; confirmed fine at review")

    def test_smoothed(self):
        job = make_job(analysis_meta=self.META, pose_flags=list(FLAGS), pose_edits=[EDIT],
                       pose_confirmed=True)
        self.assertEqual(report.pose_check_text(job), "2 frame(s) flagged; 1 point(s) smoothed at review")
        job.pose_confirmed = False
        self.assertEqual(report.pose_check_text(job),
                         "2 frame(s) flagged; 1 point(s) smoothed at review; not confirmed")

    def test_rejected(self):
        job = make_job(analysis_meta=self.META, pose_flags=list(FLAGS), review=REVIEW_REJECTED)
        self.assertEqual(report.pose_check_text(job), "2 frame(s) flagged; video removed at review")


class WriteReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_and_markdown(self):
        jobs = [
            make_job("a.mp4", review=REVIEW_APPROVED, review_note="left | right mixed",
                     analysis_meta={"model_strength": "performance"}, saved=True,
                     endpoint_source="auto", timing_source="auto"),
            make_job("b.mp4", far_ep=None),
            make_job("c.mp4", review=REVIEW_REJECTED),
        ]
        csv_path, md_path = quiet(report.write_report, jobs, self.tmp.name, "rtmlib balanced")
        self.assertEqual(csv_path, os.path.join(self.tmp.name, "labeling_report.csv"))
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, report.COLUMNS)
            rows = list(reader)
        self.assertEqual([r["result"] for r in rows], [report.APPROVED, report.FAILED, report.REJECTED])
        a = rows[0]
        self.assertEqual(a["reason"], "left | right mixed")        # CSV keeps the text as is
        self.assertEqual((a["start_s"], a["end_s"], a["duration_s"], a["speed_mps"]),
                         ("1.000", "6.000", "5.000", "2.00"))
        self.assertEqual(a["model_strength"], "performance")
        self.assertEqual(a["pose_check"], "ok")
        self.assertEqual(a["outputs_saved"], "yes")
        self.assertEqual(rows[1]["model_strength"], "")               # no analysis meta
        self.assertEqual(rows[1]["outputs_saved"], "no")

        with open(md_path) as f:
            md = f.read()
        self.assertIn("- Pose model: rtmlib balanced", md)
        self.assertIn("- Videos: 3", md)
        self.assertIn(f"  - {report.APPROVED}: 1", md)
        self.assertNotIn(f"  - {report.UNREVIEWED}:", md)              # zero counts are left out
        table = [line for line in md.splitlines() if line.startswith("| ")]
        self.assertEqual(len(table), 1 + len(jobs))                    # header + one per job
        for line in table:
            self.assertEqual(line.count("|"), len(report.COLUMNS) + 1)
        self.assertIn("left / right mixed", table[1])
        self.assertIn("—", table[2])                                   # empty cells shown as a dash

    def test_empty(self):
        csv_path, md_path = quiet(report.write_report, [], self.tmp.name, "x")
        with open(csv_path, newline="") as f:
            self.assertEqual(list(csv.reader(f)), [report.COLUMNS])
        with open(md_path) as f:
            self.assertIn("- Videos: 0", f.read())


class ProcessingReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def video(self, name, meta=None, failed=False):
        """A fake video file, with an analysis file saved for it unless meta is None."""
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(name.encode())
        if meta is not None:
            job = VideoJob(path=path, output_path="", name=name)
            if failed:
                job.status, job.error = STATUS_FAILED, "cannot open video"
            analysis_file.save(job, meta)
        return path

    def test_no_videos(self):
        self.assertIsNone(report.write_processing_report([]))

    def test_rows(self):
        run = lambda strength, n, ex: {"model_strength": strength, "flagged_frames": n,
                                       "flagged_points": n, "examples": ex}
        example = {"time_s": 1.234, "landmark": "left_heel", "kind": "foot_length"}
        videos = [
            self.video("ok.mp4", {"model": "balanced", "model_strength": "balanced",
                                  "pose_check": {"result": "ok", "runs": [run("balanced", 0, [])]}}),
            self.video("fixed.mp4", {"model": "balanced", "model_strength": "performance",
                                     "pose_check": {"result": "fixed by heavier model",
                                                    "runs": [run("balanced", 3, [example]),
                                                             run("performance", 0, [])]}}),
            self.video("remain.mp4", {"model": "balanced", "model_strength": "performance",
                                      "pose_check": {"result": "anomalies remain",
                                                     "runs": [run("balanced", 5, [example]),
                                                              run("performance", 2, [example, example])]}}),
            self.video("failed.mp4", {"model": "balanced"}, failed=True),
            self.video("old.mp4", {"model": "m.onnx"}),                 # no pose_check meta
            self.video("new.mp4"),                                       # not processed
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            csv_path, md_path = report.write_processing_report(videos)
        folder = os.path.join(self.tmp.name, analysis_file.ANALYSIS_DIR)
        self.assertEqual(os.path.dirname(csv_path), folder)
        self.assertEqual(os.path.dirname(md_path), folder)
        self.assertIn("Pose anomalies remain: remain.mp4", out.getvalue())

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, report.PROCESSING_COLUMNS)
            rows = {r["file"]: r for r in reader}
        self.assertEqual(list(rows), ["ok.mp4", "fixed.mp4", "remain.mp4", "failed.mp4", "old.mp4", "new.mp4"])

        self.assertEqual(rows["ok.mp4"]["status"], "processed")
        self.assertEqual(rows["ok.mp4"]["pose_check"], "ok")
        self.assertEqual(rows["ok.mp4"]["flagged_frames_first"], "0")

        fixed = rows["fixed.mp4"]
        self.assertEqual((fixed["model_requested"], fixed["model_strength"]), ("balanced", "performance"))
        self.assertEqual((fixed["flagged_frames_first"], fixed["flagged_frames_final"]), ("3", "0"))
        self.assertEqual(fixed["examples"], "")

        remain = rows["remain.mp4"]
        self.assertEqual((remain["flagged_frames_first"], remain["flagged_frames_final"]), ("5", "2"))
        self.assertEqual(remain["examples"], "left_heel foot_length at 1.23s; left_heel foot_length at 1.23s")

        failed = rows["failed.mp4"]
        self.assertEqual((failed["status"], failed["error"]), (STATUS_FAILED, "cannot open video"))
        self.assertEqual(failed["model_strength"], "balanced")          # falls back to the model

        old = rows["old.mp4"]
        self.assertEqual((old["pose_check"], old["model_strength"]), ("not checked", "m.onnx"))
        self.assertEqual(old["flagged_frames_first"], "")

        new = rows["new.mp4"]
        self.assertEqual((new["status"], new["pose_check"], new["model_strength"]), ("not processed", "", ""))

        with open(md_path) as f:
            md = f.read()
        self.assertIn("- Videos: 6", md)
        self.assertIn("pose anomalies fixed by re-running with the heavier model: 1", md)
        self.assertIn("(the reviewer is asked to confirm or remove): 1", md)
        table = [line for line in md.splitlines() if line.startswith("| ")]
        self.assertEqual(len(table), 1 + len(videos))
        for line in table:
            self.assertEqual(line.count("|"), len(report.PROCESSING_COLUMNS) + 1)

    def test_remain_variants_count(self):
        # "anomalies remain (no heavier model)" counts as remaining too.
        videos = [self.video("a.mp4", {"pose_check": {"result": "anomalies remain (no heavier model)",
                                                      "runs": []}})]
        _, md_path = quiet(report.write_processing_report, videos)
        with open(md_path) as f:
            self.assertIn("(the reviewer is asked to confirm or remove): 1", f.read())

    def test_pipes_escaped_in_markdown(self):
        videos = [self.video("a.mp4", {"model": "x|y"}, failed=True)]
        csv_path, md_path = quiet(report.write_processing_report, videos)
        with open(md_path) as f:
            row = [line for line in f.read().splitlines() if line.startswith("| a.mp4")][0]
        self.assertIn("x/y", row)
        self.assertEqual(row.count("|"), len(report.PROCESSING_COLUMNS) + 1)


if __name__ == "__main__":
    unittest.main()
