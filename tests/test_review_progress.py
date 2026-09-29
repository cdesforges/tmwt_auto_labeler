"""Tests for saving and resuming review progress (tmwt/session/review_progress.py)."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from tmwt.core.job import (REVIEW_APPROVED, REVIEW_REJECTED, REVIEW_UNREVIEWED, STATUS_FAILED,
                           STATUS_OK, FrameResult, VideoJob)
from tmwt.core.video_io import VideoInfo
from tmwt.detection import pose_check, pose_smoothing
from tmwt.detection.people import Track
from tmwt.pose import pose_common as pc
from tmwt.session import review_progress

FPS = 25.0


def pose(x=0.5, y=0.6):
    """A pose with a nose, both ankles and the left small toe."""
    p = [None] * pc.NUM_LANDMARKS
    p[0] = pc.Landmark(x, y - 0.4)
    p[27] = pc.Landmark(x - 0.02, y)
    p[28] = pc.Landmark(x + 0.02, y)
    p[33] = pc.Landmark(x - 0.01, y + 0.02)
    return p


def make_job(folder, name, n=10, fingerprint="fp", created="2026-01-01T10:00"):
    """A freshly 'loaded' job: analysed, automatic timing, nothing reviewed."""
    job = VideoJob(path=os.path.join(folder, name),
                   output_path=os.path.join(folder, "out", os.path.splitext(name)[0] + ".csv"),
                   name=name)
    job.info = VideoInfo(0, None, FPS, n, np.zeros((40, 30, 3), np.uint8))
    job.analysis_meta = {"video_fingerprint": fingerprint, "created": created}
    job.status = STATUS_OK
    for k in range(n):
        f = FrameResult(frame_idx=k, time_s=k / FPS, H=None)
        f.people = [pose(0.3), pose(0.7)]
        f.pose = f.people[0]
        job.frames.append(f)
    job.tracks = [Track(1, {k: 0 for k in range(n)}), Track(2, {k: 1 for k in range(n)})]
    job.subject = 1
    job.far_ep, job.near_ep = (10, 5), (12, 35)
    job.endpoint_source = "auto"
    job.walk_start, job.walk_end = 0.1, 0.3
    job.timing_source, job.timing_detail = "auto", "standstill"
    return job


class ReviewProgressTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def jobs(self, **kw):
        return [make_job(self.dir, "a.mp4", **kw), make_job(self.dir, "b.mp4", **kw),
                make_job(self.dir, "c.mp4", **kw)]

    def path(self):
        return os.path.join(self.dir, "tmwt_analysis", review_progress.FILE_NAME)

    # --- Where the file goes ---------------------------------------------------

    def test_progress_path_is_in_the_videos_analysis_folder(self):
        self.assertEqual(review_progress.progress_path(self.jobs()), self.path())

    def test_save_creates_the_analysis_folder(self):
        jobs = self.jobs()
        review_progress.save(jobs, 0)
        self.assertTrue(os.path.exists(self.path()))
        self.assertFalse(os.path.exists(self.path() + ".tmp"))

    # --- Round trip ------------------------------------------------------------

    def test_save_load_restore_round_trip(self):
        jobs = self.jobs()
        a, b, c = jobs
        a.review, a.review_note, a.saved = REVIEW_APPROVED, "", True
        a.far_ep, a.near_ep = (11.7, 6.2), (13, 36)      # floats are stored as ints
        a.endpoint_source, a.far_ep_is_standing_spot = "manual", True
        a.walk_start, a.walk_end = 0.12, 0.34
        a.timing_source, a.timing_detail = "manual", "marked during review"
        a.timing_note = "check the start"
        a.pose_confirmed = True
        b.review, b.review_note = REVIEW_REJECTED, "skipped at review"
        b.far_ep = b.near_ep = None
        b.walk_start = b.walk_end = None
        review_progress.save(jobs, 1)

        fresh = self.jobs()
        progress = review_progress.load(fresh)
        self.assertIsNotNone(progress)
        last = review_progress.restore(fresh, progress)
        self.assertEqual(last, 1)
        fa, fb, fc = fresh
        self.assertEqual((fa.review, fa.saved), (REVIEW_APPROVED, True))
        self.assertEqual((fa.far_ep, fa.near_ep), ((11, 6), (13, 36)))
        self.assertEqual((fa.endpoint_source, fa.far_ep_is_standing_spot), ("manual", True))
        self.assertEqual((fa.walk_start, fa.walk_end), (0.12, 0.34))
        self.assertEqual((fa.timing_source, fa.timing_detail), ("manual", "marked during review"))
        self.assertEqual(fa.timing_note, "check the start")
        self.assertTrue(fa.pose_confirmed)
        self.assertEqual((fb.review, fb.review_note), (REVIEW_REJECTED, "skipped at review"))
        self.assertIsNone(fb.far_ep)
        self.assertIsNone(fb.near_ep)
        self.assertIsNone(fb.walk_start)
        self.assertEqual(fc.review, REVIEW_UNREVIEWED)
        self.assertFalse(fc.saved)

    def test_restoring_endpoints_reprojects_them_into_every_frame(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        jobs[0].far_ep, jobs[0].near_ep = (1, 2), (3, 4)
        review_progress.save(jobs, 0)
        fresh = self.jobs()
        review_progress.restore(fresh, review_progress.load(fresh))
        self.assertEqual(fresh[0].frames[5].far_ep, (1, 2))
        self.assertEqual(fresh[0].frames[5].near_ep, (3, 4))

    def test_restore_changes_the_subject(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        jobs[0].subject = 2
        review_progress.save(jobs, 0)
        fresh = self.jobs()
        review_progress.restore(fresh, review_progress.load(fresh))
        self.assertEqual(fresh[0].subject, 2)
        self.assertIs(fresh[0].frames[3].pose, fresh[0].frames[3].people[1])
        self.assertIsNotNone(fresh[0].subject_start)

    def test_restore_ignores_a_subject_that_is_no_longer_tracked(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        jobs[0].subject = 9
        review_progress.save(jobs, 0)
        fresh = self.jobs()
        review_progress.restore(fresh, review_progress.load(fresh))
        self.assertEqual(fresh[0].subject, 1)
        self.assertEqual(fresh[0].review, REVIEW_APPROVED)

    def test_pose_edits_are_reapplied_to_freshly_loaded_poses(self):
        jobs = self.jobs()
        a = jobs[0]
        a.review = REVIEW_APPROVED
        a.frames[4].pose_flags = {33: pose_check.FOOT_LENGTH}
        original = a.frames[4].pose[33]
        edit = pose_smoothing.Edit(4, 33, pose_check.FOOT_LENGTH, original, pc.Landmark(0.25, 0.75, 0.5))
        pose_smoothing.apply(a, edit)
        a.pose_edits.append(edit)
        review_progress.save(jobs, 0)

        fresh = self.jobs()
        fresh[0].frames[4].pose_flags = {33: pose_check.FOOT_LENGTH}   # as the analysis file gives it
        review_progress.restore(fresh, review_progress.load(fresh))
        f = fresh[0].frames[4]
        self.assertEqual((f.pose[33].x, f.pose[33].y, f.pose[33].z), (0.25, 0.75, 0.5))
        self.assertEqual(f.pose_smoothed, {33})
        self.assertEqual(len(fresh[0].pose_edits), 1)
        restored = fresh[0].pose_edits[0]
        self.assertEqual((restored.frame, restored.landmark, restored.kind), (4, 33, pose_check.FOOT_LENGTH))
        self.assertAlmostEqual(restored.original.x, original.x)   # the detected point, for Unsmooth
        # Undoing puts the freshly loaded point back.
        pose_smoothing.undo_all(fresh[0])
        self.assertAlmostEqual(f.pose[33].x, original.x)
        self.assertEqual(f.pose_smoothed, set())

    def test_pose_edits_for_missing_frames_or_points_are_skipped(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        with open(self.path()) as f:
            data = json.load(f)
        data["videos"]["a.mp4"]["pose_edits"] = [[99, 33, 0.1, 0.1, 0.0], [-1, 33, 0.1, 0.1, 0.0],
                                                 [2, 5, 0.1, 0.1, 0.0], [3, 33, 0.2, 0.2, 0.0]]
        with open(self.path(), "w") as f:
            json.dump(data, f)
        fresh = self.jobs()
        fresh[0].frames[1].pose = None
        review_progress.restore(fresh, review_progress.load(fresh))
        self.assertEqual([(e.frame, e.landmark) for e in fresh[0].pose_edits], [(3, 33)])

    def test_failed_jobs_are_not_saved(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        jobs[2].status = STATUS_FAILED
        review_progress.save(jobs, 0)
        with open(self.path()) as f:
            self.assertEqual(sorted(json.load(f)["videos"]), ["a.mp4", "b.mp4"])

    def test_last_is_none_without_a_last_index(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, None)
        fresh = self.jobs()
        self.assertIsNone(review_progress.restore(fresh, review_progress.load(fresh)))

    # --- outputs_saved -----------------------------------------------------------

    def test_outputs_saved_is_recorded_and_carried_over(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        self.assertIsNone(review_progress.load(jobs)["outputs_saved"])
        review_progress.save(jobs, None, outputs_saved=True)
        stamp = review_progress.load(jobs)["outputs_saved"]
        self.assertIsNotNone(stamp)
        review_progress.save(jobs, 1)                    # a later save keeps it
        self.assertEqual(review_progress.load(jobs)["outputs_saved"], stamp)

    # --- What load keeps -----------------------------------------------------------

    def test_load_without_a_file_is_none(self):
        self.assertIsNone(review_progress.load(self.jobs()))

    def test_load_of_an_unreadable_file_is_none(self):
        os.makedirs(os.path.dirname(self.path()))
        with open(self.path(), "w") as f:
            f.write("{not json")
        self.assertIsNone(review_progress.load(self.jobs()))

    def test_nothing_decided_is_none(self):
        jobs = self.jobs()
        jobs[0].walk_start = 0.2      # edits, but no decision
        review_progress.save(jobs, 0)
        self.assertTrue(os.path.exists(self.path()))
        self.assertIsNone(review_progress.load(jobs))

    def test_another_format_version_is_ignored(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        with open(self.path()) as f:
            data = json.load(f)
        data["format_version"] = review_progress.FORMAT_VERSION + 1
        with open(self.path(), "w") as f:
            json.dump(data, f)
        self.assertIsNone(review_progress.load(jobs))

    def test_replaced_video_entry_is_dropped(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        jobs[1].review = REVIEW_REJECTED
        review_progress.save(jobs, 0)
        fresh = self.jobs()
        fresh[0].analysis_meta["video_fingerprint"] = "another video"
        progress = review_progress.load(fresh)
        self.assertEqual(sorted(progress["videos"]), ["b.mp4", "c.mp4"])
        review_progress.restore(fresh, progress)
        self.assertEqual(fresh[0].review, REVIEW_UNREVIEWED)
        self.assertEqual(fresh[1].review, REVIEW_REJECTED)

    def test_reprocessed_video_entry_is_dropped(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        jobs[1].review = REVIEW_REJECTED
        review_progress.save(jobs, 0)
        fresh = self.jobs()
        fresh[1].analysis_meta["created"] = "2026-02-02T10:00"
        self.assertNotIn("b.mp4", review_progress.load(fresh)["videos"])

    def test_only_decision_on_a_replaced_video_means_no_progress(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        fresh = self.jobs(fingerprint="new")
        self.assertIsNone(review_progress.load(fresh))

    def test_entries_for_videos_no_longer_there_or_failed_are_dropped(self):
        jobs = self.jobs()
        jobs[0].review = jobs[2].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        fresh = self.jobs()[1:]          # a.mp4 gone
        fresh[1].status = STATUS_FAILED  # c.mp4 failed to load
        self.assertIsNone(review_progress.load(fresh))

    # --- counts ----------------------------------------------------------------------

    def test_counts(self):
        jobs = self.jobs() + [make_job(self.dir, "d.mp4")]
        jobs[0].review = jobs[3].review = REVIEW_APPROVED
        jobs[1].review = REVIEW_REJECTED
        review_progress.save(jobs, 0)
        self.assertEqual(review_progress.counts(review_progress.load(jobs)), (2, 1, 1))

    # --- snapshot / put_back / clear ---------------------------------------------------

    def test_snapshot_without_a_file_is_none(self):
        self.assertIsNone(review_progress.snapshot(self.jobs()))

    def test_put_back_restores_the_exact_text(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        with open(self.path()) as f:
            before = f.read()
        snap = review_progress.snapshot(jobs)
        self.assertEqual(snap, before)
        jobs[0].review, jobs[1].review = REVIEW_REJECTED, REVIEW_APPROVED
        review_progress.save(jobs, 1, outputs_saved=True)
        review_progress.put_back(jobs, snap)
        with open(self.path()) as f:
            self.assertEqual(f.read(), before)

    def test_put_back_of_no_snapshot_deletes_the_file(self):
        jobs = self.jobs()
        snap = review_progress.snapshot(jobs)
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        review_progress.put_back(jobs, snap)
        self.assertFalse(os.path.exists(self.path()))
        review_progress.put_back(jobs, None)           # nothing there: no error
        self.assertFalse(os.path.exists(self.path()))

    def test_put_back_after_the_file_was_cleared(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        snap = review_progress.snapshot(jobs)
        review_progress.clear(jobs)          # "Start over"
        review_progress.put_back(jobs, snap)
        self.assertEqual(review_progress.snapshot(jobs), snap)

    def test_clear(self):
        jobs = self.jobs()
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        review_progress.clear(jobs)
        self.assertFalse(os.path.exists(self.path()))
        review_progress.clear(jobs)          # twice is fine
        self.assertIsNone(review_progress.load(jobs))


if __name__ == "__main__":
    unittest.main()
