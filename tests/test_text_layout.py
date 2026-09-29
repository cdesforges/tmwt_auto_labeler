"""
Text layout: no text collides with other text, buttons or the screen's edges.

The screens' real text is collected by running the real code with a recording
stand-in for the window (so new wording is checked automatically), then
measured against the layouts the screens use (base_window.message_layout,
labeler_ui.menu_layout, widgets.button_row, the top bar, badges, the info
panel and the sidebar).
"""

import io
import unittest
from unittest import mock

import cv2
import numpy as np

from tests.test_base_window import FakeWindow
from tmwt.core.job import (REVIEW_APPROVED, REVIEW_REJECTED, STATUS_FAILED, STATUS_INCOMPLETE,
                           STATUS_NEEDS_INPUT, STATUS_NO_BODY, STATUS_OK, FrameResult, VideoJob)
from tmwt.core.video_io import VideoInfo
from tmwt.detection import people, pose_check, pose_smoothing
from tmwt.measurement import onset, timing
from tmwt.session import flagged_points, review, review_playback, review_session
from tmwt.ui import annotate, base_window, sidebar as sb
from tmwt.ui.base_window import MESSAGE_LINE, MESSAGE_TITLE, message_layout
from tmwt.ui.labeler_ui import MENU_BUTTON_W, MENU_LINE, MENU_TITLE, LabelerUI, menu_layout
from tmwt.ui.pickers import EndpointPicking
from tmwt.ui.player import Player
from tmwt.ui.top_bar import TopBar
from tmwt.ui.widgets import (BTN_H, CONFIRM, FONT, MAIN_H, MAIN_W, TEXT_MARGIN, TextButton,
                             bar_buttons, button_row)

LONG_NAME = "REACH_AlbCtrl_10m_participant_0042_visit_3_hallway_B_2026-09-29_take_2.mp4"


def size(text, scale, thickness=1):
    """(width, height above the baseline, depth below it) of `text` in pixels."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h, base


def make_job(name="walk.mp4", n=100):
    job = VideoJob(path=name, output_path=name + ".csv", name=name)
    job.info = VideoInfo(0, None, 25.0, n, np.zeros((720, 404, 3), np.uint8))
    job.frames = [FrameResult(frame_idx=k, time_s=k / 25.0, H=None) for k in range(n)]
    job.status = STATUS_OK
    job.far_ep, job.near_ep = (200, 300), (200, 700)
    job.far_ep_is_standing_spot = True
    job.endpoint_source = "auto start, manual finish"
    job.walk_start, job.walk_end = 12.34, 123.45
    job.timing_source, job.timing_detail = "auto", "first right foot movement (on/past start line)"
    return job


class Recorder:
    """
    Replaces a LabelerUI's show_message / ask_menu: records what would be shown
    and answers from `answers` (title -> value, or a list of values in turn),
    else with the first button.
    """

    def __init__(self, ui, answers=None):
        self.messages, self.menus = [], []
        self.answers = dict(answers or {})
        ui.show_message = self.show_message
        ui.ask_menu = self.ask_menu

    def show_message(self, lines, specs, background=None, dim=None):
        self.messages.append((lines, specs))
        answer = self.answers.get(lines[0][0], specs[0][1])
        return answer.pop(0) if isinstance(answer, list) else answer

    def ask_menu(self, background, title, lines, options, note=None):
        self.menus.append((title, lines, options, note))
        return "back"


class LayoutTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(base_window, "Window", FakeWindow)
        patcher.start()
        self.addCleanup(patcher.stop)
        out = mock.patch("sys.stdout", new_callable=io.StringIO)
        out.start()
        self.addCleanup(out.stop)
        self.ui = LabelerUI([LONG_NAME, "walk.mp4"], heading="control_vids")
        self.rec = Recorder(self.ui)

    # --- checks ---------------------------------------------------------------------

    def assert_fits_width(self, text, scale, thickness, room, what):
        w = size(text, scale, thickness)[0]
        self.assertLessEqual(w, room, f"{what}: {text!r} is {w}px wide, room for {room}px")

    def assert_buttons_fit(self, buttons, what):
        self.assertGreaterEqual(buttons[0].x, 0, what)
        self.assertLessEqual(buttons[-1].x + buttons[-1].w, MAIN_W, what)
        for a, b in zip(buttons, buttons[1:]):
            self.assertLess(a.x + a.w, b.x, f"{what}: {a.text!r} overlaps {b.text!r}")
        for b in buttons:
            self.assertLessEqual(b.y + b.h, MAIN_H, what)
            if isinstance(b, TextButton):
                self.assert_fits_width(b.text, 0.6, 1, b.w - 16, f"{what} button")

    def assert_message_fits(self, lines, specs):
        title = lines[0][0]
        baselines, buttons_y = message_layout(len(lines))
        tops, bottoms = [], []
        for k, ((text, _), y) in enumerate(zip(lines, baselines)):
            scale, thickness = MESSAGE_TITLE if k == 0 else MESSAGE_LINE
            self.assert_fits_width(text, scale, thickness, MAIN_W - 2 * TEXT_MARGIN, f"message {title!r}")
            _, h, base = size(text or "x", scale, thickness)
            tops.append(y - h)
            bottoms.append(y + base)
        self.assertGreaterEqual(tops[0], 0, f"message {title!r} starts above the screen")
        for k in range(len(lines) - 1):
            self.assertLess(bottoms[k], tops[k + 1], f"message {title!r}: lines {k} and {k + 1} overlap")
        self.assertLess(bottoms[-1], buttons_y, f"message {title!r}: text runs into its buttons")
        self.assert_buttons_fit(button_row(specs, buttons_y), f"message {title!r}")

    def assert_menu_fits(self, title, lines, options, note):
        layout = menu_layout(len(lines), len(options))
        self.assertGreaterEqual(layout.title - size(title, *MENU_TITLE)[1], 0)
        self.assert_fits_width(title, *MENU_TITLE, MAIN_W - 2 * TEXT_MARGIN, "menu title")
        for text in lines:
            self.assert_fits_width(text, *MENU_LINE, MAIN_W - 2 * TEXT_MARGIN, "menu line")
        if lines:
            self.assertLess(layout.lines[-1] + size(lines[-1], *MENU_LINE)[2], layout.options[0],
                            "menu lines run into the options")
        for key_label, text, _, _ in options:
            # KeyedButton: the shortcut at +16, the label at +80
            self.assert_fits_width(f"[{key_label}]", 0.6, 2, 80 - 16 - 4, "menu shortcut")
            self.assert_fits_width(text, 0.6, 1, MENU_BUTTON_W - 80 - 12, "menu option")
        last_bottom = layout.options[-1] + BTN_H
        self.assertLessEqual(last_bottom, MAIN_H, "menu options run off the screen")
        if note:
            self.assert_fits_width(note, *MENU_LINE, MAIN_W - 2 * TEXT_MARGIN, "menu note")
            self.assertLess(last_bottom, layout.note - size(note, *MENU_LINE)[1], "note overlaps options")
            self.assertLessEqual(layout.note + size(note, *MENU_LINE)[2], MAIN_H, "note off the screen")


# --- message screens -------------------------------------------------------------------

class MessageScreensTest(LayoutTestCase):
    def check_recorded(self):
        self.assertTrue(self.rec.messages, "nothing was recorded")
        for lines, specs in self.rec.messages:
            self.assert_message_fits(lines, specs)

    def test_quit_dialog_and_its_confirmation(self):
        self.ui.in_review = True
        self.rec.answers = {"Quit the review?": ["discard", "cancel"], "Quit without saving?": "no"}
        self.ui._ask_quit()
        self.assertEqual([m[0][0][0] for m in self.rec.messages], ["Quit the review?", "Quit without saving?",
                                                                    "Quit the review?"])
        self.check_recorded()

    def test_resume_prompt_and_start_over(self):
        progress = {"saved": "2026-09-29T18:02", "outputs_saved": "2026-09-29T18:02",
                    "videos": {f"v{i}": {"review": r} for i, r in
                               enumerate(["approved"] * 120 + ["rejected"] * 30 + ["unreviewed"] * 50)}}
        self.rec.answers = {"Continue your previous review?": "restart", "Start over?": "yes"}
        review_session.ask_resume(self.ui, progress)
        self.check_recorded()

    def test_analysis_complete(self):
        jobs = []
        for status in (STATUS_OK, STATUS_NEEDS_INPUT, STATUS_INCOMPLETE, STATUS_NO_BODY, STATUS_FAILED):
            for _ in range(123):
                j = make_job()
                j.status = status
                jobs.append(j)
        self.rec.answers = {"Analysis complete": "skip"}
        review_session.ask_to_review(self.ui, jobs)
        self.check_recorded()

    def test_review_complete_with_every_kind_of_result(self):
        jobs = []
        for review_state in (REVIEW_APPROVED, REVIEW_REJECTED, "unreviewed"):
            for _ in range(123):
                j = make_job()
                j.review = review_state
                jobs.append(j)
        j = make_job()
        j.status, j.error = STATUS_FAILED, "x"
        jobs.append(j)
        review_session.confirm_save(jobs, self.ui)
        self.check_recorded()

    def test_final_confirmation_complete(self):
        job = make_job()
        job.pose_flags = [pose_check.Flag(k, k / 25, 33, "foot_length") for k in range(40)]
        job.pose_edits = [pose_smoothing.Edit(k, 33, "foot_length", None, None) for k in range(40)]
        job.timing_note = timing.NOTE_SHORT_STANDSTILL
        review._final_confirm(job, self.ui, None)
        self.check_recorded()

    def test_final_confirmation_with_missing_times(self):
        for start, end, standing in ((None, 5.0, True), (None, 5.0, False), (1.0, None, True),
                                     (None, None, True)):
            job = make_job()
            job.walk_start, job.walk_end, job.far_ep_is_standing_spot = start, end, standing
            job.timing_note = timing.NOTE_NO_START.format(reason=onset.START_TOO_SOON)
            review._final_confirm(job, self.ui, None)
        self.check_recorded()

    def test_no_timing_after_picking_points(self):
        job = make_job(LONG_NAME)
        self.ui.pick_endpoints = lambda *a, **k: ((10, 10), (20, 20), True)

        def no_timing(j):
            j.walk_start = j.walk_end = None
            j.timing_note = timing.NOTE_NO_START.format(reason="neither foot left its standstill band")
        self.rec.answers = {"Walk start and end not found": "skip"}
        with mock.patch.object(timing, "update_timing", no_timing):
            review._set_endpoints(job, self.ui, "reason")
        self.check_recorded()

    def test_pose_anomaly_notice(self):
        job = make_job()
        job.pose_flags = [pose_check.Flag(k, 123.45 + k, idx, "foot_length")
                          for k, idx in enumerate((33, 34, 29, 30, 31))]
        job.pose_edits = [pose_smoothing.Edit(0, 33, "foot_length", None, None)]
        job.analysis_meta = {"model_strength": "pose_landmarker_heavy",
                             "pose_check": {"result": "anomalies remain (heavier model found nobody)",
                                            "runs": [{}, {}]}}
        flagged_points.alert_pose_flags(job, self.ui)
        self.check_recorded()

    def test_no_body_detected(self):
        job = make_job(LONG_NAME)
        job.status = STATUS_NO_BODY
        review.review_job(job, self.ui, 0)
        self.check_recorded()

    def test_all_done_summary(self):
        jobs = [make_job() for _ in range(4)]
        jobs[1].review = REVIEW_APPROVED
        jobs[2].review = REVIEW_REJECTED
        review_session.show_summary(self.ui, jobs, "/a/very/long/path/to/the/output/folder/labeling_report.csv")
        self.check_recorded()


# --- menus -----------------------------------------------------------------------------

class MenuTest(LayoutTestCase):
    def test_review_menu_with_every_option_and_a_note(self):
        job = make_job()
        job.pose_flags = [pose_check.Flag(1, 0.04, 33, "foot_length")]
        with mock.patch.object(people, "people_on_screen", return_value=(0, [1, 2])):
            review._menu(job, self.ui, None, note="The stop mark was before the start mark. Timing unchanged.")
        (title, lines, options, note), = self.rec.menus
        self.assertEqual(len(options), 6)
        self.assert_menu_fits(title, lines, options, note)

    def test_the_most_options_a_menu_can_hold(self):
        options = [(str(k), "Option", str(k), ()) for k in range(9)]
        self.assert_menu_fits("Menu", ["a", "b"], options, "a note")


# --- button bars -----------------------------------------------------------------------

class ButtonBarTest(unittest.TestCase):
    def assert_bar_fits(self, specs, what):
        LayoutTestCase.assert_buttons_fit(self, bar_buttons(specs), what)

    assert_fits_width = LayoutTestCase.assert_fits_width

    def test_playback_bars(self):
        for paused in (False, True):
            for m_next in ("start", "stop"):
                player = Player([0.0, 0.04])
                player.paused = paused
                specs = ([review_playback._mark_button(w, m_next) for w in ("start", "stop")]
                         + player.transport() + [CONFIRM, review_playback._MENU])
                self.assert_bar_fits(specs, "playback bar")

    def test_flagged_points_bars(self):
        transport = Player([0.0, 0.04]).transport()
        for toggle in (("Smooth points", "smooth", ()), ("Unsmooth", "unsmooth", ())):
            specs = ([flagged_points._PREV_FLAG] + transport
                     + [flagged_points._NEXT_FLAG, toggle, CONFIRM])
            self.assert_bar_fits(specs, "flagged-points bar")

    def test_endpoint_picker_bars_in_every_state(self):
        for start, finish, auto in (((1, 1), None, (1, 1)), (None, None, (1, 1)), ((5, 5), (9, 9), (1, 1)),
                                    ((5, 5), (9, 9), None), ((1, 1), None, (2, 2))):
            self.assert_bar_fits(EndpointPicking(start, finish, auto).specs(), "endpoint picker")

    def test_person_picker_bar(self):
        self.assert_bar_fits([CONFIRM, ("Cancel", "cancel", ())], "person picker")

    def test_picker_prompts_fit(self):
        for prompt in EndpointPicking.PROMPTS.values():
            self.assert_fits_width(prompt, 0.65, 2, MAIN_W - 32, "endpoint prompt")
        self.assert_fits_width("Person 12 chosen (green): Confirm, or click someone else", 0.65, 2,
                               MAIN_W - 32, "person prompt")


# --- badges and notices over the playback --------------------------------------------------

class BadgeTest(unittest.TestCase):
    assert_fits_width = LayoutTestCase.assert_fits_width

    def test_playback_notices_and_alerts(self):
        job = make_job()
        texts = [review_playback._STOP_BEFORE_START, timing.NOTE_SHORT_STANDSTILL,
                 "Rope endpoints not set: use the menu (Esc) to set them, or to skip the video"]
        for reason in (onset.START_TOO_SOON, "neither foot left its standstill band",
                       "no sustained forward movement found", "longest sustained advance only 0.42 m"):
            texts.append(timing.NOTE_NO_START.format(reason=reason))
        for start, end in ((None, 5.0), (1.0, None), (None, None)):
            job.walk_start, job.walk_end = start, end
            texts.append(review.incomplete_notice(job))
        for text in texts:
            # alerts: scale 0.6, thickness 2; notices 0.5 / 1; plus the badge's padding
            self.assert_fits_width(text, 0.6 if text == review_playback._STOP_BEFORE_START else 0.5,
                                   2 if text == review_playback._STOP_BEFORE_START else 1,
                                   MAIN_W - 2 * TEXT_MARGIN - 16, "notice")

    def test_top_left_badges(self):
        labels = [f"EDITED  start {review_playback._mark_text(123.45)}  stop "
                  f"{review_playback._mark_text(None)}  (M: start)",
                  "FLAGGED POINTS  1234 frame(s)  1234 smoothed (yellow)"]
        for label in labels:
            self.assert_fits_width(label, 0.55, 1, MAIN_W - 12 - 16 - 12, "badge")


# --- the info panel next to the frame --------------------------------------------------------

class InfoPanelTest(unittest.TestCase):
    NAMES = list(pose_check.LANDMARK_NAMES.values())

    def panel(self, height=720, **kw):
        args = dict(model_strength="pose_landmarker_heavy", flagged=self.NAMES[:6], smoothed=self.NAMES[6:])
        args.update(kw)
        return annotate.draw_info_panel(height, 123.45, 12345, -12.345, 0.5, 2.0, title="TMWT Labeler",
                                        subtitle=LONG_NAME, **args)

    def test_nothing_is_drawn_past_the_right_edge(self):
        # long model names, file names and point names are shortened, not cut off by the edge
        self.assertFalse((self.panel()[:, -3:] > 60).any())

    def test_flag_lists_stop_above_the_bottom(self):
        for height in (360, 404, 480, 720):
            with self.subTest(height=height):
                blank = self.panel(height, flagged=(), smoothed=())
                rows = np.where((blank != self.panel(height)).any(axis=(1, 2)))[0]
                if rows.size:
                    self.assertLessEqual(rows.max(), height - 30 + 6, "flag list runs into the bottom")

    def test_a_tall_panel_lists_every_point(self):
        img = self.panel(720)
        blank = self.panel(720, flagged=(), smoothed=())
        rows = np.where((blank != img).any(axis=(1, 2)))[0]
        self.assertGreater(rows.max() - rows.min(), 12 * 20)     # all twelve names, not "+N more"


# --- sidebar -----------------------------------------------------------------------------------

class SidebarTest(unittest.TestCase):
    def test_long_names_and_notes_stay_in_their_rows(self):
        bar = sb.Sidebar(0, 0, MAIN_H, [LONG_NAME] * 3)
        bar.set_state(0, sb.DONE, "rejected: pose detection anomalies (a long, long explanation)")
        bar.mark_reviewed(1, sb.SAVED_MARK)
        img = bar.render(None, None)
        self.assertEqual(img.shape[1], sb.SIDEBAR_W)
        # nothing drawn in the rightmost few pixels (the text is shortened before the edge)
        self.assertFalse((img[:, -4:] > 200).any())


# --- top bar -------------------------------------------------------------------------------------

class TopBarTest(unittest.TestCase):
    TITLES = ("", "v6", "control_vids", "control_vids_cluster_version_test3_" * 5, "Études_é_" * 20)

    @staticmethod
    def drawn(img, base):
        """(first, last) column where `img` differs from `base`, or None."""
        c = np.where((img != base).any(axis=(0, 2)))[0]
        return (int(c.min()), int(c.max())) if c.size else None

    def regions(self, width, title, save):
        """Columns of the title, the buttons and the logos on a real bar (same logos throughout)."""
        def render(title="", save=False, logos=True):
            bar = TopBar(width, title) if logos else TopBar(width, title, logo_paths=())
            bar.show_save = save
            return bar.render(None, None), bar
        full, bar = render(title, save)
        no_title, _ = render("", save)
        no_logos, _ = render("", save, logos=False)
        bare, _ = render("", False, logos=False)
        buttons_only = render("", save, logos=False)[0]
        empty = np.full_like(bare, bare[:, -1:].copy())          # the plain bar background
        empty[-1] = bare[-1]                                      # keep the bottom rule
        return (self.drawn(full, no_title),                      # the title
                self.drawn(buttons_only, empty),                 # the buttons
                self.drawn(no_title, no_logos),                  # the logos
                bar)

    def test_nothing_overlaps_at_any_width(self):
        for width in (100, 160, 320, 480, 640, 960, 1280, 1920):
            for title in self.TITLES:
                for save in (False, True):
                    t, b, l, bar = self.regions(width, title, save)
                    with self.subTest(width=width, title=title[:20], save=save):
                        for a_name, a, c_name, c in (("title", t, "buttons", b), ("title", t, "logos", l),
                                                     ("buttons", b, "logos", l)):
                            if a is not None and c is not None:
                                self.assertTrue(a[1] < c[0] or c[1] < a[0], f"{a_name} {a} overlaps {c_name} {c}")
                        for region in (t, b, l):
                            if region is not None:
                                self.assertGreaterEqual(region[0], 0)
                                self.assertLess(region[1], width)

    def test_logos_drop_out_before_they_would_collide(self):
        wide, narrow, tiny = TopBar(1280), TopBar(160), TopBar(100)
        self.assertEqual(len(wide.shown_logos()), 2)
        self.assertLess(len(narrow.shown_logos()), 2)
        self.assertEqual(tiny.shown_logos(), [])
        self.assertEqual(tiny.render(None, None).shape[1], 100)   # still draws, just without logos
        self.assertGreaterEqual(TopBar(1280).title_width(), 400)   # room left for a real title


if __name__ == "__main__":
    unittest.main()
