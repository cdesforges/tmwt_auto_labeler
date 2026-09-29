"""Tests for reviewing one video (tmwt/session/review.py, review_playback.py, flagged_points.py), without a window."""

import io
import unittest
from unittest import mock

import numpy as np

from tmwt.core.job import (REVIEW_APPROVED, REVIEW_REJECTED, REVIEW_UNREVIEWED, STATUS_NO_BODY,
                           STATUS_OK, FrameResult, VideoJob)
from tmwt.core.video_io import VideoInfo
from tmwt.detection import people, pose_check
from tmwt.measurement import timing
from tmwt.pose.pose_common import FLAGGED_COLOR, SMOOTHED_COLOR
from tmwt.session import flagged_points, review, review_playback
from tmwt.session.review_playback import PlaybackResult
from tmwt.ui.sidebar import APPROVED_MARK, DONE, FAILED, REJECTED_MARK

FPS = 25.0


def make_job(n=20):
    job = VideoJob(path="v.mp4", output_path="v.csv", name="v.mp4")
    job.info = VideoInfo(0, None, FPS, n, np.zeros((20, 10, 3), np.uint8))
    job.frames = [FrameResult(frame_idx=k, time_s=k / FPS, H=None) for k in range(n)]
    job.status = STATUS_OK
    job.far_ep, job.near_ep = (5, 2), (5, 18)
    job.endpoint_source = "auto"
    job.walk_start, job.walk_end = 1.0, 6.0
    job.timing_source, job.timing_detail = "auto", "standstill"
    return job


def flag(job, k, idx=33):
    job.pose_flags.append(pose_check.Flag(k, k / FPS, idx, pose_check.FOOT_LENGTH))
    job.frames[k].pose_flags[idx] = pose_check.FOOT_LENGTH


def result(value, start=1.0, end=6.0, k=0):
    return PlaybackResult(value, start, end, "last-frame", k)


class FakeUI:
    """What review.py uses of LabelerUI. `answers`: message title -> value or list of values."""

    def __init__(self, answers=None, menu=(), endpoints=()):
        self.active = None
        self.states, self.notes, self.reviewed = {}, {}, {}
        self.answers = dict(answers or {})
        self.menu = list(menu)
        self.endpoints = list(endpoints)
        self.titles, self.specs, self.menus = [], [], []

    def set_state(self, i, state, note=""):
        self.states[i], self.notes[i] = state, note

    def mark_reviewed(self, i, outcome):
        self.reviewed[i] = outcome

    def show_message(self, lines, specs, background=None):
        title = lines[0][0]
        self.titles.append(title)
        self.specs.append(specs)
        answer = self.answers[title]
        return answer.pop(0) if isinstance(answer, list) else answer

    def ask_menu(self, background, title, lines, options, note=None):
        self.menus.append((options, note))
        return self.menu.pop(0)

    def pick_endpoints(self, frame, reason=None, start=None, finish=None, auto_start=None):
        return self.endpoints.pop(0)


class Base(unittest.TestCase):
    def setUp(self):
        p = mock.patch("sys.stdout", new_callable=io.StringIO)
        p.start()
        self.addCleanup(p.stop)

    def patch(self, target, attr, **kw):
        p = mock.patch.object(target, attr, **kw)
        m = p.start()
        self.addCleanup(p.stop)
        return m


# --- Pure helpers ------------------------------------------------------------------

class HelpersTest(Base):
    def test_summary_lines_complete(self):
        job = make_job()
        self.assertEqual(review.summary_lines(job), [
            "Endpoints: auto    Timing: auto (standstill)",
            "Start 1.00s    End 6.00s    Duration 5.00s (2.00 m/s)"])

    def test_summary_lines_incomplete(self):
        job = make_job()
        job.endpoint_source = job.timing_source = job.timing_detail = ""
        job.walk_end = None
        self.assertEqual(review.summary_lines(job), [
            "Endpoints: none    Timing: none",
            "Timing incomplete: start 1.00s, end not found"])
        job.walk_start, job.walk_end = None, 6.0
        self.assertEqual(review.summary_lines(job)[1], "Timing incomplete: start not found, end 6.00s")

    def test_apply_marks_missing_mark(self):
        job = make_job()
        for start, end in ((None, 5.0), (2.0, None), (None, None)):
            self.assertIn("both a start and a stop", review._apply_marks(job, start, end, "x"))
        self.assertEqual((job.walk_start, job.walk_end, job.timing_source), (1.0, 6.0, "auto"))

    def test_apply_marks_stop_before_start(self):
        job = make_job()
        self.assertIn("before the start", review._apply_marks(job, 4.0, 3.0, "x"))
        self.assertIn("before the start", review._apply_marks(job, 4.0, 4.0, "x"))   # zero-length
        self.assertEqual((job.walk_start, job.walk_end, job.timing_source), (1.0, 6.0, "auto"))

    def test_apply_marks_valid_makes_timing_manual(self):
        job = make_job()
        self.assertIsNone(review._apply_marks(job, 0.0, 3.0, "marked during review"))
        self.assertEqual((job.walk_start, job.walk_end), (0.0, 3.0))
        self.assertEqual((job.timing_source, job.timing_detail), ("manual", "marked during review"))

    def test_skip_reason(self):
        job = make_job()
        self.assertEqual(review._skip_reason(job), "skipped at review")
        flag(job, 5)
        self.assertEqual(review._skip_reason(job), "pose detection anomalies")
        job.pose_confirmed = True
        self.assertEqual(review._skip_reason(job), "skipped at review")

    def menu_values(self, job):
        ui = FakeUI(menu=["back"])
        review._menu(job, ui, None)
        return [value for _, _, value, _ in ui.menus[0][0]]

    def test_menu_basic_options(self):
        self.assertEqual(self.menu_values(make_job()), ["endpoints", "skip", "back", "quit"])

    def test_menu_flagged_option_only_with_flags(self):
        job = make_job()
        flag(job, 5)
        self.assertEqual(self.menu_values(job), ["endpoints", "skip", "flags", "back", "quit"])

    def test_menu_person_option_only_with_several_people(self):
        job = make_job()
        with mock.patch.object(people, "people_on_screen", return_value=(3, [("a", None), ("b", None)])):
            self.assertEqual(self.menu_values(job), ["endpoints", "skip", "person", "back", "quit"])
        with mock.patch.object(people, "people_on_screen", return_value=(None, [])):
            self.assertNotIn("person", self.menu_values(job))

    def test_menu_passes_the_note(self):
        ui = FakeUI(menu=["back"])
        review._menu(make_job(), ui, None, "why")
        self.assertEqual(ui.menus[0][1], "why")

    def test_final_confirm_complete(self):
        ui = FakeUI({"Confirm this video?": "next"})
        self.assertEqual(review._final_confirm(make_job(), ui, None), "next")
        self.assertEqual([v for _, v, _ in ui.specs[0]], ["next", "back"])

    def test_final_confirm_incomplete_says_what_is_missing(self):
        job = make_job()
        job.walk_end = None
        ui = FakeUI({"Walk end not found": "skip"})
        self.assertEqual(review._final_confirm(job, ui, None), "skip")
        # Time manually (back to the playback), Reselect points, Skip this file
        self.assertEqual([v for _, v, _ in ui.specs[0]], ["back", "reselect", "skip"])
        self.assertEqual(ui.specs[0][0][0], "Time manually")

    # --- what's missing -------------------------------------------------------------

    def test_missing_times(self):
        job = make_job()
        self.assertEqual(review.missing_times(job), [])
        job.walk_start = None
        self.assertEqual(review.missing_times(job), ["start"])
        job.walk_end = None
        self.assertEqual(review.missing_times(job), ["start", "end"])
        job.walk_start = 1.0
        self.assertEqual(review.missing_times(job), ["end"])

    def test_incomplete_notice_names_the_dots(self):
        job = make_job()
        self.assertIsNone(review.incomplete_notice(job))
        job.walk_end = None
        self.assertIn("Walk end not found", review.incomplete_notice(job))
        self.assertIn("red dot", review.incomplete_notice(job))
        self.assertNotIn("green dot", review.incomplete_notice(job))
        job.walk_start = None
        note = review.incomplete_notice(job)
        self.assertIn("Walk start and end not found", note)
        self.assertIn("green and red dots", note)

    def test_incomplete_notice_fits_the_playback(self):
        import cv2
        from tmwt.ui.widgets import FONT, MAIN_W
        job = make_job()
        for start, end in ((None, 6.0), (1.0, None), (None, None)):
            job.walk_start, job.walk_end = start, end
            width = cv2.getTextSize(review.incomplete_notice(job), FONT, 0.5, 1)[0][0] + 16
            self.assertLess(width, MAIN_W)

    def titles_and_text(self, job):
        lines = review.missing_timing_lines(job)
        return lines[0][0], " ".join(t for t, _ in lines[1:])

    def test_missing_start_at_the_standing_spot_gives_the_reason(self):
        job = make_job()
        job.walk_start, job.far_ep_is_standing_spot = None, True
        job.timing_note = timing.NOTE_NO_START.format(reason="the walk starts too soon after the recording begins")
        title, text = self.titles_and_text(job)
        self.assertEqual(title, "Walk start not found")
        self.assertIn("the walk starts too soon", text)
        self.assertIn("drag the", text)                            # the already-walking hint
        self.assertNotIn("finish line", text)

    def test_missing_start_with_a_start_line(self):
        job = make_job()
        job.walk_start, job.far_ep_is_standing_spot = None, False
        title, text = self.titles_and_text(job)
        self.assertIn("crossing the start line", text)

    def test_missing_end_explains_the_finish_line(self):
        job = make_job()
        job.walk_end = None
        title, text = self.titles_and_text(job)
        self.assertEqual(title, "Walk end not found")
        self.assertIn("finish line", text)
        self.assertNotIn("Start:", text)

    def test_missing_both(self):
        job = make_job()
        job.walk_start = job.walk_end = None
        self.assertEqual(self.titles_and_text(job)[0], "Walk start and end not found")


# --- review_job ----------------------------------------------------------------------

class ReviewJobTest(Base):
    def setUp(self):
        super().setUp()
        self.play = self.patch(review, "playback")
        self.flagged = self.patch(review, "review_flagged_points")
        self.alert = self.patch(review, "alert_pose_flags")

    def plays(self, *results):
        self.play.side_effect = list(results)

    def start_ks(self):
        return [c.kwargs.get("start_k") for c in self.play.call_args_list]

    def test_confirm_then_next_approves(self):
        job = make_job()
        self.plays(result("confirm", k=40))
        ui = FakeUI({"Confirm this video?": "next"})
        self.assertIsNone(review.review_job(job, ui, 3))
        self.assertEqual((job.review, job.saved), (REVIEW_APPROVED, False))
        self.assertEqual((ui.states[3], ui.notes[3]), (DONE, "approved  5.00s"))
        self.assertEqual(ui.reviewed[3], APPROVED_MARK)
        self.assertEqual(job.timing_source, "auto")      # marks unchanged: timing stays automatic
        self.assertEqual(ui.active, 3)

    def test_approving_again_clears_saved(self):
        job = make_job()
        job.review, job.saved = REVIEW_APPROVED, True
        self.plays(result("confirm"))
        review.review_job(job, FakeUI({"Confirm this video?": "next"}), 0)
        self.assertFalse(job.saved)

    def test_back_resumes_the_playback_at_the_same_frame(self):
        job = make_job()
        self.plays(result("confirm", k=42), result("confirm", k=42))
        ui = FakeUI({"Confirm this video?": ["back", "next"]})
        review.review_job(job, ui, 0)
        self.assertEqual(self.start_ks(), [None, 42])
        self.assertEqual(job.review, REVIEW_APPROVED)

    def test_new_marks_become_manual_timing(self):
        job = make_job()
        self.plays(result("confirm", start=2.0, end=5.0))
        review.review_job(job, FakeUI({"Confirm this video?": "next"}), 0)
        self.assertEqual((job.walk_start, job.walk_end, job.timing_source), (2.0, 5.0, "manual"))

    def test_bad_marks_go_back_to_the_playback_with_why(self):
        job = make_job()
        self.plays(result("confirm", start=5.0, end=2.0, k=7), result("confirm", k=7))
        ui = FakeUI({"Confirm this video?": "next"})
        review.review_job(job, ui, 0)
        self.assertEqual(ui.titles, ["Confirm this video?"])       # only after the second playback
        second = self.play.call_args_list[1].kwargs
        self.assertEqual(second["start_k"], 7)
        self.assertIn("before the start", second["notice"])
        self.assertEqual((job.walk_start, job.walk_end), (1.0, 6.0))

    def test_incomplete_timing_can_be_skipped_at_the_final_confirm(self):
        job = make_job()
        job.walk_end = None
        self.plays(result("confirm", end=None))
        ui = FakeUI({"Walk end not found": "skip"})
        review.review_job(job, ui, 0)
        self.assertEqual((job.review, job.review_note), (REVIEW_REJECTED, "skipped at review"))
        self.assertEqual(ui.reviewed[0], REJECTED_MARK)

    def test_menu_skip_rejects(self):
        job = make_job()
        job.saved = True
        self.plays(result("menu"))
        ui = FakeUI(menu=["skip"])
        self.assertIsNone(review.review_job(job, ui, 2))
        self.assertEqual((job.review, job.review_note, job.saved), (REVIEW_REJECTED, "skipped at review", False))
        self.assertEqual((ui.states[2], ui.notes[2]), (FAILED, "rejected: skipped at review"))
        self.assertEqual(ui.reviewed[2], REJECTED_MARK)

    def test_menu_skip_with_unconfirmed_flags_gives_pose_anomalies(self):
        job = make_job()
        flag(job, 5)
        self.plays(result("menu"))
        review.review_job(job, FakeUI(menu=["skip"]), 0)
        self.assertEqual(job.review_note, "pose detection anomalies")
        self.alert.assert_called_once()
        self.flagged.assert_called_once()

    def test_menu_quit(self):
        job = make_job()
        self.plays(result("menu"))
        self.assertEqual(review.review_job(job, FakeUI(menu=["quit"]), 0), review.QUIT)
        self.assertEqual(job.review, REVIEW_UNREVIEWED)

    def test_menu_back_resumes_where_it_was(self):
        job = make_job()
        self.plays(result("menu", k=9), result("confirm", k=9))
        review.review_job(job, FakeUI({"Confirm this video?": "next"}, menu=["back"]), 0)
        self.assertEqual(self.start_ks(), [None, 9])

    def test_menu_flags_opens_flagged_points_mode(self):
        job = make_job()
        flag(job, 5)
        job.pose_confirmed = True
        self.plays(result("menu", k=9), result("confirm", k=9))
        review.review_job(job, FakeUI({"Confirm this video?": "next"}, menu=["flags"]), 0)
        self.assertEqual(self.flagged.call_count, 2)     # on opening, then from the menu
        self.assertEqual(self.start_ks(), [None, 9])

    def test_menu_marks_changed_are_applied_before_the_menu(self):
        job = make_job()
        self.plays(result("menu", start=2.0, end=3.0), result("confirm", start=2.0, end=3.0))
        review.review_job(job, FakeUI({"Confirm this video?": "next"}, menu=["back"]), 0)
        self.assertEqual((job.walk_start, job.walk_end, job.timing_source), (2.0, 3.0, "manual"))

    def test_menu_new_endpoints_replay_from_the_start(self):
        job = make_job()
        self.plays(result("menu", k=9), result("confirm", start=0.5, end=4.5))

        def update(j):
            j.walk_start, j.walk_end, j.timing_source = 0.5, 4.5, "auto"

        self.patch(timing, "update_timing", side_effect=update)
        ui = FakeUI({"Confirm this video?": "next"}, menu=["endpoints"], endpoints=[((1, 1), (2, 2), True)])
        review.review_job(job, ui, 0)
        self.assertEqual(self.start_ks(), [None, None])
        self.assertEqual((job.far_ep, job.near_ep, job.endpoint_source), ((1, 1), (2, 2), "manual"))
        self.assertEqual((job.walk_start, job.walk_end, job.timing_source), (0.5, 4.5, "auto"))

    def test_reselect_points_from_the_final_confirm(self):
        job = make_job()
        job.walk_end = None
        self.plays(result("confirm", end=None, k=7), result("confirm", start=0.5, end=4.5))

        def update(j):
            j.walk_start, j.walk_end, j.timing_source = 0.5, 4.5, "auto"

        self.patch(timing, "update_timing", side_effect=update)
        ui = FakeUI({"Walk end not found": "reselect", "Confirm this video?": "next"},
                    endpoints=[((1, 1), (2, 2), True)])
        review.review_job(job, ui, 0)
        self.assertEqual(job.near_ep, (2, 2))
        self.assertEqual(self.start_ks(), [None, None])            # replays from the start
        self.assertEqual(job.review, REVIEW_APPROVED)

    def test_incomplete_timing_shows_what_to_mark_in_the_playback(self):
        job = make_job()
        job.walk_start = None
        self.plays(result("menu"))
        ui = FakeUI(menu=["quit"])
        review.review_job(job, ui, 0)
        notice = self.play.call_args_list[0].kwargs["notice"]
        self.assertIn("Walk start not found", notice)

    def test_timing_still_incomplete_after_picking_offers_time_manually(self):
        job = make_job()
        job.far_ep = job.near_ep = None
        self.plays(result("menu"))

        def update(j):
            j.walk_start, j.walk_end = 1.0, None

        self.patch(timing, "update_timing", side_effect=update)
        ui = FakeUI({"Walk end not found": "manual"}, menu=["quit"], endpoints=[((1, 1), (2, 2), True)])
        review.review_job(job, ui, 0)
        self.assertEqual([v for _, v, _ in ui.specs[0]], ["reselect", "manual", "skip"])
        self.assertIn("red dot", self.play.call_args_list[0].kwargs["notice"])

    def test_confirm_with_unconfirmed_flags_goes_to_flagged_points_first(self):
        job = make_job()
        flag(job, 5)
        calls = []

        def flagged_mode(j, ui):
            calls.append(len(self.play.call_args_list))
            if len(calls) == 2:
                j.pose_confirmed = True

        self.flagged.side_effect = flagged_mode
        self.plays(result("confirm", k=12), result("confirm", k=12))
        ui = FakeUI({"Confirm this video?": "next"})
        review.review_job(job, ui, 0)
        self.assertEqual(calls, [0, 1])          # on opening, then after the first Confirm
        self.assertEqual(self.start_ks(), [None, 12])
        self.assertEqual(ui.titles, ["Confirm this video?"])
        self.assertEqual(job.review, REVIEW_APPROVED)

    def test_confirmed_flags_go_straight_to_the_final_confirm(self):
        job = make_job()
        flag(job, 5)
        self.flagged.side_effect = lambda j, ui: setattr(j, "pose_confirmed", True)
        self.plays(result("confirm"))
        review.review_job(job, FakeUI({"Confirm this video?": "next"}), 0)
        self.assertEqual(self.flagged.call_count, 1)
        self.assertEqual(job.review, REVIEW_APPROVED)

    def test_no_body_only_offers_skip(self):
        job = make_job()
        job.status = STATUS_NO_BODY
        ui = FakeUI({"No body detected": "skip"})
        self.assertIsNone(review.review_job(job, ui, 1))
        self.assertEqual([v for _, v, _ in ui.specs[0]], ["skip"])
        self.assertEqual((job.review, job.review_note), (REVIEW_REJECTED, "no body detected"))
        self.play.assert_not_called()

    def test_endpoint_picking_cancelled_plays_with_a_note(self):
        job = make_job()
        job.far_ep = job.near_ep = None
        job.endpoint_problem = "no rope"
        self.plays(result("menu"))
        review.review_job(job, FakeUI(menu=["skip"], endpoints=[None]), 0)
        self.assertIn("Rope endpoints not set", self.play.call_args_list[0].kwargs["notice"])
        self.assertEqual(job.review, REVIEW_REJECTED)


# --- review_playback --------------------------------------------------------------------

class PlaybackHelpersTest(Base):
    def test_m_key_only_on_the_mark_it_sets_next(self):
        start = review_playback._mark_button("start", "start")
        stop = review_playback._mark_button("stop", "start")
        self.assertEqual(start, ("Mark walk start", "mark_start", review_playback._MARK_KEYS, "mark_start"))
        self.assertEqual(stop, ("Mark walk stop", "mark_stop", (), "mark_stop"))
        self.assertEqual(review_playback._mark_button("stop", "stop")[2], review_playback._MARK_KEYS)
        self.assertEqual(review_playback._mark_button("start", "stop")[2], ())

    def test_mark_text(self):
        self.assertEqual(review_playback._mark_text(None), "--")
        self.assertEqual(review_playback._mark_text(0.0), "0.00s")
        self.assertEqual(review_playback._mark_text(3.456), "3.46s")

    def test_pose_highlights(self):
        job = make_job(6)
        job.frames[1].pose_flags = {33: "x"}                                   # unsmoothed
        job.frames[2].pose_flags, job.frames[2].pose_smoothed = {33: "x"}, {33}   # smoothed
        job.frames[4].pose_flags, job.frames[4].pose_smoothed = {31: "x", 33: "x"}, {33}   # half
        self.assertEqual(review_playback.pose_highlights(job), {FLAGGED_COLOR: [1, 4], SMOOTHED_COLOR: [2, 4]})

    def test_pose_highlights_empty(self):
        self.assertEqual(review_playback.pose_highlights(make_job(3)), {FLAGGED_COLOR: [], SMOOTHED_COLOR: []})


# --- flagged_points ---------------------------------------------------------------------

class RetimeTest(Base):
    def test_retime_recomputes_automatic_timing_with_endpoints(self):
        job = make_job()
        with mock.patch.object(timing, "update_timing") as update:
            flagged_points._retime(job)
        update.assert_called_once_with(job)

    def test_retime_keeps_manual_timing(self):
        job = make_job()
        job.timing_source = "manual"
        with mock.patch.object(timing, "update_timing") as update:
            flagged_points._retime(job)
        update.assert_not_called()

    def test_retime_needs_both_endpoints(self):
        for missing in ("far_ep", "near_ep"):
            job = make_job()
            setattr(job, missing, None)
            with mock.patch.object(timing, "update_timing") as update:
                flagged_points._retime(job)
            update.assert_not_called()


class PickSubjectTest(Base):
    """_pick_subject: choosing the walker among several people."""

    class Cap:
        def set(self, *a):
            pass

        def read(self):
            return True, np.zeros((20, 10, 3), np.uint8)

        def release(self):
            pass

    def setUp(self):
        super().setUp()
        self.job = make_job()
        self.job.subject = 1
        self.tracks = [mock.Mock(id=1), mock.Mock(id=2)]
        self.patch(people, "people_on_screen", return_value=(3, [(t, "pose") for t in self.tracks]))
        self.patch(self.job.__class__, "open_capture", return_value=self.Cap())
        self.set_subject = self.patch(people, "set_subject")
        self.patch(people, "subject_start", return_value=(5, 5))
        self.update = self.patch(timing, "update_timing")

    def ui(self, choice):
        ui = mock.Mock()
        ui.pick_person.return_value = choice
        return ui

    def test_current_subject_is_chosen_to_begin_with(self):
        ui = self.ui(None)
        review._pick_subject(self.job, ui)
        self.assertEqual(ui.pick_person.call_args.kwargs["selected"], 0)

    def test_same_person_changes_nothing(self):
        # e.g. confirming the walker already tracked: manual timing must survive
        self.job.timing_source = "manual"
        self.assertFalse(review._pick_subject(self.job, self.ui(0)))
        self.set_subject.assert_not_called()
        self.update.assert_not_called()

    def test_cancel_changes_nothing(self):
        self.assertFalse(review._pick_subject(self.job, self.ui(None)))
        self.update.assert_not_called()

    def test_another_person_becomes_the_subject(self):
        self.job.pose_confirmed = True
        self.assertTrue(review._pick_subject(self.job, self.ui(1)))
        self.set_subject.assert_called_once_with(self.job, self.tracks[1])
        self.assertFalse(self.job.pose_confirmed)
        self.update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
