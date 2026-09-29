"""Tests for the review session's helpers and flows (tmwt/session/review_session.py), without a window."""

import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from tmwt.core import data_export
from tmwt.core.job import (REVIEW_APPROVED, REVIEW_REJECTED, REVIEW_UNREVIEWED, STATUS_FAILED,
                           STATUS_INCOMPLETE, STATUS_NEEDS_INPUT, STATUS_NO_BODY, STATUS_OK,
                           FrameResult, VideoJob)
from tmwt.core.video_io import VideoInfo
from tmwt.session import review, review_progress, review_session
from tmwt.ui.events import JumpTo, QuitWithoutSaving, SaveAndQuit, WindowClosed
from tmwt.ui.sidebar import (APPROVED_MARK, DONE, FAILED, NEEDS_INPUT, SAVED_MARK, UNREVIEWED,
                             WAITING, WORKING)

FPS = 25.0


def make_job(folder, name, n=5, status=STATUS_OK, review_=REVIEW_UNREVIEWED):
    job = VideoJob(path=os.path.join(folder, name),
                   output_path=os.path.join(folder, "out", os.path.splitext(name)[0] + ".csv"),
                   name=name)
    job.info = VideoInfo(0, None, FPS, n, np.zeros((20, 10, 3), np.uint8))
    job.analysis_meta = {"video_fingerprint": "fp-" + name, "created": "2026-01-01T10:00"}
    job.frames = [FrameResult(frame_idx=k, time_s=k / FPS, H=None) for k in range(n)]
    job.status, job.review = status, review_
    if status != STATUS_FAILED:
        job.far_ep, job.near_ep = (5, 2), (5, 18)
        job.endpoint_source = "auto"
        job.walk_start, job.walk_end = 1.0, 4.0
        job.timing_source = "auto"
    else:
        job.frames, job.error = [], "cannot open video"
    return job


class FakeUI:
    """
    Just what review_session uses of LabelerUI. `answers` maps a message's
    title to the value to return (or an exception to raise, or a list of
    those, used in turn).
    """

    def __init__(self, n, answers=None, confirm=True):
        self.active = None
        self.review_targets = set()
        self.states = [WAITING] * n
        self.notes = [""] * n
        self.reviewed = {}
        self.in_review = False
        self.answers = dict(answers or {})
        self.confirm_answer = confirm
        self.titles = []
        self.progress = []

    def set_state(self, i, state, note=""):
        self.states[i], self.notes[i] = state, note

    def mark_reviewed(self, i, outcome):
        self.reviewed[i] = outcome

    def show_progress(self, title, subtitle, fraction, preview=None, force=False, cancellable=False):
        self.progress.append((title, fraction))

    def show_message(self, lines, specs, background=None):
        title = lines[0][0]
        self.titles.append(title)
        if title not in self.answers:
            raise AssertionError(f"unexpected screen {title!r}")
        answer = self.answers[title]
        if isinstance(answer, list):
            answer = answer.pop(0)
        if isinstance(answer, BaseException) or (isinstance(answer, type) and issubclass(answer, BaseException)):
            raise answer
        return answer

    def confirm(self, title, lines, yes, no="Go back", background=None):
        self.titles.append(title)
        return self.confirm_answer


def fake_save_job(calls):
    def save_job(job, on_progress=None):
        calls.append(job.name)
        if on_progress is not None:
            on_progress(1.0)
        job.saved = True
    return save_job


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.out = os.path.join(self.dir, "out")
        os.makedirs(self.out)
        self.saved, self.removed = [], []
        for p in (mock.patch("sys.stdout", new_callable=io.StringIO),   # quiet progress prints
                  mock.patch.object(data_export, "save_job", fake_save_job(self.saved)),
                  mock.patch.object(data_export, "remove_outputs",
                                    lambda job: self.removed.append(job.name))):
            p.start()
            self.addCleanup(p.stop)

    def jobs(self, n=3):
        return [make_job(self.dir, f"v{i}.mp4") for i in range(n)]

    def progress_file(self, jobs):
        return review_progress.progress_path(jobs)

    def report_written(self):
        return os.path.exists(os.path.join(self.out, "labeling_report.csv"))


# --- Helpers -------------------------------------------------------------------

class HelpersTest(Fixture):
    def test_reviewable_leaves_out_failed_jobs(self):
        jobs = self.jobs(4)
        jobs[1].status = STATUS_FAILED
        jobs[3].status = STATUS_NO_BODY          # can still be reviewed (skipped)
        self.assertEqual(review_session.reviewable(jobs), [0, 2, 3])

    def test_next_unreviewed_goes_forward(self):
        jobs = self.jobs(4)
        self.assertEqual(review_session.next_unreviewed(jobs, -1), 0)
        self.assertEqual(review_session.next_unreviewed(jobs, 1), 2)

    def test_next_unreviewed_wraps_round(self):
        jobs = self.jobs(4)
        jobs[2].review = jobs[3].review = REVIEW_APPROVED
        self.assertEqual(review_session.next_unreviewed(jobs, 2), 0)
        self.assertEqual(review_session.next_unreviewed(jobs, 3), 0)

    def test_next_unreviewed_skips_decided_and_failed(self):
        jobs = self.jobs(5)
        jobs[1].review = REVIEW_APPROVED
        jobs[2].review = REVIEW_REJECTED
        jobs[3].status = STATUS_FAILED
        self.assertEqual(review_session.next_unreviewed(jobs, 0), 4)
        self.assertEqual(review_session.next_unreviewed(jobs, 4), 0)

    def test_next_unreviewed_can_return_the_same_job(self):
        jobs = self.jobs(3)
        jobs[0].review = jobs[2].review = REVIEW_APPROVED
        self.assertEqual(review_session.next_unreviewed(jobs, 1), 1)

    def test_next_unreviewed_is_none_when_all_decided(self):
        jobs = self.jobs(3)
        jobs[0].review = jobs[1].review = REVIEW_APPROVED
        jobs[2].status = STATUS_FAILED
        self.assertIsNone(review_session.next_unreviewed(jobs, -1))
        self.assertIsNone(review_session.next_unreviewed([], -1))

    def test_analysis_state_ok(self):
        job = self.jobs(1)[0]
        self.assertEqual(review_session._analysis_state(job), (DONE, "auto  3.00s"))

    def test_analysis_state_needs_input(self):
        job = self.jobs(1)[0]
        job.status, job.endpoint_problem = STATUS_NEEDS_INPUT, "no rope"
        self.assertEqual(review_session._analysis_state(job), (NEEDS_INPUT, "needs endpoints: no rope"))
        job.subject_start = (3, 4)
        self.assertEqual(review_session._analysis_state(job), (NEEDS_INPUT, "needs finish point: no rope"))

    def test_analysis_state_incomplete_says_what_is_missing(self):
        job = self.jobs(1)[0]
        job.status, job.walk_end = STATUS_INCOMPLETE, None
        self.assertEqual(review_session._analysis_state(job),
                         (FAILED, "Timing incomplete: start 1.00s, end not found"))

    def test_analysis_state_failed_and_no_body_give_the_error(self):
        job = self.jobs(1)[0]
        job.status, job.error = STATUS_FAILED, "cannot open video"
        self.assertEqual(review_session._analysis_state(job), (FAILED, "cannot open video"))
        job.status, job.error = STATUS_NO_BODY, "no body detected in any frame"
        self.assertEqual(review_session._analysis_state(job), (FAILED, "no body detected in any frame"))

    def test_pose_models(self):
        jobs = self.jobs(4)
        jobs[0].analysis_meta.update(backend="rtmlib", model_strength="balanced", model="rtmpose-m", device="mps")
        jobs[1].analysis_meta.update(backend="rtmlib", model_strength="balanced", model="rtmpose-m", device="mps")
        jobs[2].analysis_meta.update(backend="mediapipe", model="heavy")        # no strength, no device
        # jobs[3]: no backend (e.g. failed) -> left out
        self.assertEqual(review_session._pose_models(jobs), "mediapipe: heavy (?), rtmlib: balanced (mps)")

    def test_pose_models_unknown(self):
        self.assertEqual(review_session._pose_models(self.jobs(2)), "unknown")


# --- save_outputs -----------------------------------------------------------------

class SaveOutputsTest(Fixture):
    def test_what_is_saved_and_removed(self):
        jobs = self.jobs(5)
        jobs[0].review = REVIEW_REJECTED                 # rejected: outputs removed, not saved
        jobs[1].review = REVIEW_UNREVIEWED               # automatic result: saved
        jobs[2].review = REVIEW_APPROVED                 # approved: saved
        jobs[3].far_ep = None                            # unreviewed, no endpoints: not saved
        jobs[4] = make_job(self.dir, "v4.mp4", status=STATUS_FAILED)   # no frames: not saved
        ui = FakeUI(5)
        ui.notes[2] = "approved  3.00s"
        review_session.save_outputs(jobs, ui)
        self.assertEqual(self.removed, ["v0.mp4"])
        self.assertEqual(self.saved, ["v1.mp4", "v2.mp4"])
        self.assertEqual((ui.states[1], ui.notes[1]), (UNREVIEWED, "saved (not reviewed)"))
        self.assertNotIn(1, ui.reviewed)
        self.assertEqual((ui.states[2], ui.notes[2]), (DONE, "approved  3.00s"))
        self.assertEqual(ui.reviewed[2], SAVED_MARK)
        self.assertEqual([t for t, _ in ui.progress], ["Saving v1.mp4", "Saving v2.mp4"])

    def test_unreviewed_video_with_incomplete_timing_is_not_saved(self):
        # The Review complete screen and the report call it unsaveable, so it isn't written.
        jobs = self.jobs(2)
        jobs[0].walk_end = None
        review_session.save_outputs(jobs, FakeUI(2))
        self.assertEqual(self.saved, ["v1.mp4"])

    def test_approved_without_endpoints_is_still_saved(self):
        jobs = self.jobs(1)
        jobs[0].review, jobs[0].far_ep = REVIEW_APPROVED, None
        review_session.save_outputs(jobs, FakeUI(1))
        self.assertEqual(self.saved, ["v0.mp4"])

    def test_without_a_window(self):
        jobs = self.jobs(2)
        jobs[1].review = REVIEW_REJECTED
        review_session.save_outputs(jobs, None)
        self.assertEqual(self.saved, ["v0.mp4"])
        self.assertEqual(self.removed, ["v1.mp4"])


# --- run_review -------------------------------------------------------------------

class RunReviewTest(Fixture):
    def test_reviews_in_order_and_saves_progress_after_each(self):
        jobs = self.jobs(3)
        order = []

        def review_job(job, ui, i):
            order.append(i)
            job.review = REVIEW_APPROVED

        ui = FakeUI(3)
        with mock.patch.object(review, "review_job", review_job):
            review_session.run_review(jobs, ui, 1)
        self.assertEqual(order, [1, 2, 0])
        self.assertEqual(ui.review_targets, set())
        self.assertEqual(review_progress.counts(review_progress.load(jobs)), (3, 0, 0))

    def test_quit_stops_the_review(self):
        jobs = self.jobs(3)
        order = []
        with mock.patch.object(review, "review_job", lambda job, ui, i: order.append(i) or review.QUIT):
            review_session.run_review(jobs, FakeUI(3), 0)
        self.assertEqual(order, [0])

    def test_only_one(self):
        jobs = self.jobs(3)
        order = []

        def review_job(job, ui, i):
            order.append(i)
            job.review = REVIEW_APPROVED

        with mock.patch.object(review, "review_job", review_job):
            review_session.run_review(jobs, FakeUI(3), 2, only_one=True)
        self.assertEqual(order, [2])

    def test_jump_leaves_the_video_undecided_and_switches(self):
        jobs = self.jobs(3)
        ui = FakeUI(3)
        ui.states[0], ui.notes[0] = DONE, "auto  3.00s"
        order = []

        seen = []

        def review_job(job, ui, i):
            order.append(i)
            if i == 2:
                seen.append((ui.states[0], ui.notes[0], jobs[0].review))
            ui.set_state(i, WORKING)
            if i == 0 and order.count(0) == 1:
                raise JumpTo(2)
            job.review = REVIEW_APPROVED

        with mock.patch.object(review, "review_job", review_job):
            review_session.run_review(jobs, ui, 0)
        self.assertEqual(order, [0, 2, 0, 1])
        self.assertEqual(seen, [(DONE, "auto  3.00s", REVIEW_UNREVIEWED)])   # as before the jump


# --- run ------------------------------------------------------------------------

class RunTest(Fixture):
    def approve_all(self, job, ui, i):
        job.review = REVIEW_APPROVED
        ui.set_state(i, DONE, f"approved  {job.duration:.2f}s")
        ui.mark_reviewed(i, APPROVED_MARK)

    def test_unattended_saves_and_writes_the_report(self):
        jobs = self.jobs(3)
        jobs[2] = make_job(self.dir, "v2.mp4", status=STATUS_FAILED)
        self.assertTrue(review_session.run(jobs, None, self.out))
        self.assertEqual(self.saved, ["v0.mp4", "v1.mp4"])
        self.assertTrue(self.report_written())
        self.assertFalse(os.path.exists(self.progress_file(jobs)))   # no review, no progress

    def test_full_review_saves_outputs_progress_and_report(self):
        jobs = self.jobs(2)
        ui = FakeUI(2, {"Analysis complete": "review", "Review complete": "save", "All done": "close"})
        with mock.patch.object(review, "review_job", self.approve_all):
            self.assertTrue(review_session.run(jobs, ui, self.out))
        self.assertEqual(self.saved, ["v0.mp4", "v1.mp4"])
        self.assertEqual(ui.reviewed, {0: SAVED_MARK, 1: SAVED_MARK})
        self.assertTrue(self.report_written())
        self.assertFalse(ui.in_review)
        progress = review_progress.load(jobs)
        self.assertIsNotNone(progress["outputs_saved"])
        self.assertTrue(all(e["saved"] for e in progress["videos"].values()))

    def test_save_all_without_reviewing(self):
        jobs = self.jobs(2)
        ui = FakeUI(2, {"Analysis complete": "skip", "All done": "close"})
        with mock.patch.object(review, "review_job", side_effect=AssertionError("no review")):
            self.assertTrue(review_session.run(jobs, ui, self.out))
        self.assertEqual(self.saved, ["v0.mp4", "v1.mp4"])
        self.assertEqual(ui.states, [UNREVIEWED, UNREVIEWED])

    def test_review_first_false_skips_the_review(self):
        jobs = self.jobs(2)
        ui = FakeUI(2, {"All done": "close"})
        self.assertTrue(review_session.run(jobs, ui, self.out, review_first=False))
        self.assertEqual(self.saved, ["v0.mp4", "v1.mp4"])

    def test_resume_continues_where_it_stopped(self):
        jobs = self.jobs(3)
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 1)
        fresh = self.jobs(3)
        order = []

        def review_job(job, ui, i):
            order.append(i)
            job.review = REVIEW_APPROVED

        ui = FakeUI(3, {"Continue your previous review?": "continue", "Review complete": "save",
                        "All done": "close"})
        with mock.patch.object(review, "review_job", review_job):
            review_session.run(fresh, ui, self.out)
        self.assertEqual(order, [1, 2])                  # v0 was approved already
        self.assertEqual(ui.reviewed[0], SAVED_MARK)

    def test_start_over_clears_the_progress(self):
        jobs = self.jobs(2)
        jobs[0].review = REVIEW_REJECTED
        review_progress.save(jobs, 0)
        fresh = self.jobs(2)
        ui = FakeUI(2, {"Continue your previous review?": "restart", "Analysis complete": "skip",
                        "All done": "close"}, confirm=True)
        review_session.run(fresh, ui, self.out)
        self.assertIn("Start over?", ui.titles)
        self.assertEqual(fresh[0].review, REVIEW_UNREVIEWED)
        self.assertEqual(self.saved, ["v0.mp4", "v1.mp4"])

    def test_save_and_quit_keeps_progress_and_writes_nothing(self):
        jobs = self.jobs(3)

        def review_job(job, ui, i):
            ui.active = i
            if i == 1:
                raise SaveAndQuit()
            job.review = REVIEW_APPROVED

        ui = FakeUI(3, {"Analysis complete": "review"})
        with mock.patch.object(review, "review_job", review_job):
            self.assertFalse(review_session.run(jobs, ui, self.out))
        self.assertFalse(ui.in_review)
        self.assertEqual((self.saved, self.removed), ([], []))
        self.assertFalse(self.report_written())
        progress = review_progress.load(self.jobs(3))
        self.assertEqual(review_progress.counts(progress), (1, 0, 2))
        self.assertEqual(progress["last"], "v1.mp4")

    def test_quit_without_saving_puts_the_progress_back(self):
        jobs = self.jobs(3)
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        with open(self.progress_file(jobs)) as f:
            before = f.read()
        fresh = self.jobs(3)

        def review_job(job, ui, i):
            if i == 2:
                raise QuitWithoutSaving()
            job.review = REVIEW_REJECTED                 # saved to the file, then discarded

        ui = FakeUI(3, {"Continue your previous review?": "continue"})
        with mock.patch.object(review, "review_job", review_job):
            self.assertFalse(review_session.run(fresh, ui, self.out))
        with open(self.progress_file(jobs)) as f:
            self.assertEqual(f.read(), before)
        self.assertEqual((self.saved, self.removed), ([], []))
        self.assertFalse(self.report_written())

    def test_quit_without_saving_a_first_review_leaves_no_progress(self):
        jobs = self.jobs(2)

        def review_job(job, ui, i):
            if i == 1:
                raise QuitWithoutSaving()
            job.review = REVIEW_APPROVED

        ui = FakeUI(2, {"Analysis complete": "review"})
        with mock.patch.object(review, "review_job", review_job):
            self.assertFalse(review_session.run(jobs, ui, self.out))
        self.assertFalse(os.path.exists(self.progress_file(jobs)))

    def test_window_closed_saves_progress_and_reraises(self):
        jobs = self.jobs(3)

        def review_job(job, ui, i):
            ui.active = i
            if i == 2:
                raise WindowClosed()
            job.review = REVIEW_APPROVED

        ui = FakeUI(3, {"Analysis complete": "review"})
        with mock.patch.object(review, "review_job", review_job):
            with self.assertRaises(WindowClosed):
                review_session.run(jobs, ui, self.out)
        self.assertFalse(ui.in_review)
        self.assertEqual(self.saved, [])
        self.assertFalse(self.report_written())
        progress = review_progress.load(self.jobs(3))
        self.assertEqual(review_progress.counts(progress), (2, 0, 1))
        self.assertEqual(progress["last"], "v2.mp4")

    def test_save_and_quit_on_the_review_complete_screen(self):
        jobs = self.jobs(2)
        ui = FakeUI(2, {"Analysis complete": "review", "Review complete": SaveAndQuit()})
        with mock.patch.object(review, "review_job", self.approve_all):
            self.assertFalse(review_session.run(jobs, ui, self.out))
        self.assertEqual(self.saved, [])
        progress = review_progress.load(self.jobs(2))
        self.assertEqual(review_progress.counts(progress), (2, 0, 0))
        self.assertIsNone(progress["last"])

    # Closing the window on "Continue your previous review?" (before restore)
    # must not save the freshly loaded, all-unreviewed jobs over the progress.
    def test_window_closed_at_the_resume_prompt_keeps_the_earlier_progress(self):
        jobs = self.jobs(2)
        jobs[0].review = REVIEW_APPROVED
        review_progress.save(jobs, 0)
        fresh = self.jobs(2)
        ui = FakeUI(2, {"Continue your previous review?": WindowClosed()})
        with self.assertRaises(WindowClosed):
            review_session.run(fresh, ui, self.out)
        progress = review_progress.load(self.jobs(2))
        self.assertIsNotNone(progress)
        self.assertEqual(review_progress.counts(progress), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
