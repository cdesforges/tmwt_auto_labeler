"""Tests for playback: frame lookup, pacing and transport controls (tmwt/ui/player.py)."""

import types
import unittest
from unittest import mock

from tmwt.ui import player as player_module
from tmwt.ui.player import (PlaybackClock, Player, nearest_frame, seek_index, seek_state)
from tmwt.ui.seek_bar import Marker, SeekState
from tmwt.ui.widgets import GREEN, KEY_SPACE, RED, YELLOW
from tmwt.ui.window import KEY_LEFT, KEY_RIGHT

TIMES = [i / 25 for i in range(100)]       # 25 fps, 3.96 s


class FakeClock:
    """Stands in for the `time` module in player.py: perf_counter() returns `now`."""

    def __init__(self, now=100.0):
        self.now = now

    def perf_counter(self):
        return self.now


def fake_time(clock):
    return mock.patch.object(player_module, "time", types.SimpleNamespace(perf_counter=clock.perf_counter))


class FrameLookupTest(unittest.TestCase):
    def test_nearest_frame(self):
        self.assertEqual(nearest_frame(TIMES, 1.0), 25)
        self.assertEqual(nearest_frame(TIMES, 1.015), 25)
        self.assertEqual(nearest_frame(TIMES, 1.025), 26)
        self.assertEqual(nearest_frame(TIMES, -5), 0)
        self.assertEqual(nearest_frame(TIMES, 50), 99)

    def test_uneven_times(self):
        times = [0.0, 0.1, 0.5, 0.52]
        self.assertEqual(nearest_frame(times, 0.29), 1)
        self.assertEqual(nearest_frame(times, 0.31), 2)

    def test_seek_index(self):
        self.assertEqual(seek_index(TIMES, 0.0), 0)
        self.assertEqual(seek_index(TIMES, 1.0), 99)
        self.assertEqual(seek_index(TIMES, 0.5), nearest_frame(TIMES, TIMES[-1] / 2))

    def test_seek_index_with_offset_times(self):
        times = [10 + t for t in TIMES]
        self.assertEqual(seek_index(times, 0.0), 0)
        self.assertEqual(seek_index(times, 1.0), 99)

    def test_single_frame_clip(self):
        self.assertEqual(seek_index([2.0], 0.7), 0)
        state = seek_state([2.0], 0, [(2.0, GREEN)])
        self.assertEqual(state.fraction, 0.0)
        self.assertEqual(state.playhead_fill, GREEN)


class SeekStateTest(unittest.TestCase):
    def test_position_and_text(self):
        state = seek_state(TIMES, 50)
        self.assertAlmostEqual(state.fraction, TIMES[50] / TIMES[-1])
        self.assertEqual(state.text, " 2.00 /  3.96 s")
        self.assertEqual(state.markers, [])
        self.assertEqual(state.stretches, [])

    def test_times_not_starting_at_zero(self):
        times = [5 + t for t in TIMES]
        state = seek_state(times, 0, [(5.0, GREEN), (5 + TIMES[-1], RED)])
        self.assertEqual(state.fraction, 0.0)
        self.assertEqual([m.fraction for m in state.markers], [0.0, 1.0])
        self.assertTrue(state.text.startswith(" 0.00"))

    def test_markers_only_within_the_clip(self):
        marks = [(-0.1, GREEN, "start"), (0.0, GREEN), (2.0, RED, "stop"), (TIMES[-1] + 0.01, RED)]
        state = seek_state(TIMES, 0, marks)
        self.assertEqual(len(state.markers), 2)
        self.assertEqual(state.markers[0], Marker(0.0, GREEN, None))
        self.assertAlmostEqual(state.markers[1].fraction, 2.0 / TIMES[-1])

    def test_draggable_ids(self):
        state = seek_state(TIMES, 0, [(1.0, GREEN, "start"), (2.0, RED)])
        self.assertEqual([m.id for m in state.markers], ["start", None])

    def test_fill_only_for_marks_shown(self):
        # A mark outside the clip whose nearest frame is k doesn't fill the playhead.
        self.assertIsNone(seek_state(TIMES, 99, [(TIMES[-1] + 0.01, RED)]).playhead_fill)

    def test_first_mark_on_the_frame_wins(self):
        state = seek_state(TIMES, 25, [(1.0, GREEN, "start"), (1.01, RED, "stop")])
        self.assertEqual(state.playhead_fill, GREEN)

    def test_highlight_stretches_per_colour(self):
        state = seek_state(TIMES, 0, highlights={GREEN: [5, 3, 4], YELLOW: [4, 5], RED: []})
        greens = [s for s in state.stretches if s[2] == GREEN]
        yellows = [s for s in state.stretches if s[2] == YELLOW]
        self.assertEqual(len(greens), 1)                        # unsorted, but consecutive
        self.assertEqual(len(yellows), 1)                       # colours don't merge with each other
        self.assertAlmostEqual(greens[0][0], TIMES[3] / TIMES[-1])
        self.assertAlmostEqual(greens[0][1], TIMES[6] / TIMES[-1])
        self.assertFalse([s for s in state.stretches if s[2] == RED])

    def test_gap_splits_a_stretch(self):
        state = seek_state(TIMES, 0, highlights={GREEN: [1, 2, 4]})
        self.assertEqual(len(state.stretches), 2)

    def test_last_frame_highlight(self):
        (a, b, _), = seek_state(TIMES, 0, highlights={GREEN: [99]}).stretches
        self.assertEqual((a, b), (1.0, 1.0))


class PlaybackClockTest(unittest.TestCase):
    def test_paces_to_video_time(self):
        clock, wall = PlaybackClock(), FakeClock(100.0)
        with fake_time(wall):
            self.assertEqual(clock.ms_until(2.0), 0.0)          # the first frame is shown at once
            wall.now = 100.1
            self.assertAlmostEqual(clock.ms_until(2.5), 400.0)
            wall.now = 101.0
            self.assertAlmostEqual(clock.ms_until(2.5), -500.0)  # running late

    def test_restart_re_anchors(self):
        clock, wall = PlaybackClock(), FakeClock(100.0)
        with fake_time(wall):
            clock.ms_until(0.0)
            wall.now = 150.0                                     # e.g. a long pause
            clock.restart()
            self.assertEqual(clock.ms_until(3.0), 0.0)
            wall.now = 150.04
            self.assertAlmostEqual(clock.ms_until(3.08), 40.0)


class TransportTest(unittest.TestCase):
    def test_label_and_icon_follow_the_pause_state(self):
        p = Player(TIMES)
        back, toggle, forward = p.transport()
        # The arrow keys are the frame-step buttons' own keys (so their tooltips name them).
        self.assertEqual(back[1:], ("back", (KEY_LEFT,), "prev_frame"))
        self.assertEqual(forward[1:], ("forward", (KEY_RIGHT,), "next_frame"))
        self.assertEqual((toggle[0], toggle[3]), ("Pause", "pause"))
        p.paused = True
        toggle = p.transport()[1]
        self.assertEqual((toggle[0], toggle[3]), ("Play", "play"))

    def test_space_toggles(self):
        self.assertEqual(Player(TIMES).transport()[1][1:3], ("toggle", (KEY_SPACE,)))


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.p = Player(TIMES)

    def test_toggle(self):
        self.assertTrue(self.p._apply("toggle"))
        self.assertTrue(self.p.paused)
        self.assertTrue(self.p._apply("toggle"))
        self.assertFalse(self.p.paused)

    def test_stepping_pauses_and_clamps(self):
        self.assertTrue(self.p._apply("back"))
        self.assertEqual(self.p.k, 0)                            # clamped at the start
        self.assertTrue(self.p.paused)
        self.p.paused = False
        self.p._apply("forward")
        self.assertEqual(self.p.k, 1)
        self.assertTrue(self.p.paused)
        self.p.k = self.p.last
        self.p._apply("forward")
        self.assertEqual(self.p.k, self.p.last)                  # clamped at the end
        self.p._apply("back")
        self.assertEqual(self.p.k, self.p.last - 1)

    def test_other_values_are_not_transport(self):
        for value in (None, "confirm", "menu", "mark"):
            self.assertFalse(self.p._apply(value))
        self.assertEqual((self.p.k, self.p.paused), (0, False))

    def test_seek_while_playing_resumes_after(self):
        self.assertTrue(self.p._apply(("seek", 0.5)))
        self.assertEqual(self.p.k, seek_index(TIMES, 0.5))
        self.assertTrue(self.p.paused)                            # holds still while dragging
        self.p._apply(("seek", 0.8))
        self.assertTrue(self.p.paused)
        self.p._apply(("seek_end", 0.25))
        self.assertEqual(self.p.k, seek_index(TIMES, 0.25))
        self.assertFalse(self.p.paused)                           # was playing: plays on
        self.assertIsNone(self.p._resume_after_seek)

    def test_seek_while_paused_stays_paused(self):
        self.p.paused = True
        self.p._apply(("seek", 0.5))
        self.p._apply(("seek_end", 0.6))
        self.assertTrue(self.p.paused)

    def test_seek_end_alone(self):
        # A click on the track (press and release in one go) gives only seek_end.
        self.p._apply(("seek_end", 1.0))
        self.assertEqual(self.p.k, self.p.last)
        self.assertFalse(self.p.paused)

    def test_mark_drag_and_drop(self):
        self.p._apply(("mark_drag", "start", 0.3))
        self.assertEqual(self.p.k, seek_index(TIMES, 0.3))
        self.assertTrue(self.p.paused)
        self.p._apply(("mark_drop", "start", 0.4))
        self.assertEqual(self.p.k, seek_index(TIMES, 0.4))
        self.assertFalse(self.p.paused)

    def test_a_later_drag_starts_afresh(self):
        self.p._apply(("seek", 0.5))
        self.p._apply(("seek_end", 0.5))
        self.p._apply("toggle")                                   # now paused
        self.p._apply(("seek", 0.1))
        self.p._apply(("seek_end", 0.1))
        self.assertTrue(self.p.paused)


class AdvanceTest(unittest.TestCase):
    def test_plays_forward(self):
        p = Player(TIMES)
        self.assertTrue(p.advance())
        self.assertEqual(p.k, 1)

    def test_paused_stays(self):
        p = Player(TIMES)
        p.paused = True
        self.assertTrue(p.advance())
        self.assertEqual(p.k, 0)

    def test_stops_at_the_end(self):
        p = Player(TIMES)
        p.k = p.last
        self.assertFalse(p.advance())
        self.assertEqual(p.k, p.last)

    def test_holds_once_after_a_transport_action(self):
        p = Player(TIMES)
        ui = FakeUI(["toggle"])
        p.show(ui, None, [])                    # now paused, held
        ui.values.append("toggle")
        p.show(ui, None, [])                    # playing again, held
        self.assertTrue(p.advance())
        self.assertEqual(p.k, 0)                # the hold keeps this frame one more redraw
        self.assertTrue(p.advance())
        self.assertEqual(p.k, 1)

    def test_seek_to_the_end_while_playing_holds_then_stops(self):
        p = Player(TIMES)
        p.show(FakeUI([("seek_end", 1.0)]), None, [])
        self.assertTrue(p.advance())            # shows the last frame first
        self.assertFalse(p.advance())


class FakeUI:
    """LabelerUI.show_frame stand-in: returns scripted values and records its calls."""

    def __init__(self, values=(), open_dialog_on=()):
        self.values = list(values)
        self.open_dialog_on = set(open_dialog_on)   # call numbers during which a dialog is shown
        self.dialogs_shown = 0
        self.calls = []

    def show_frame(self, img, wait_ms, specs, label=None, hotkeys=None, seek=None, alert=None,
                   notice=None):
        self.calls.append(dict(img=img, wait_ms=wait_ms, specs=specs, label=label, hotkeys=hotkeys,
                               seek=seek, alert=alert, notice=notice))
        if len(self.calls) - 1 in self.open_dialog_on:
            self.dialogs_shown += 1
        value = self.values.pop(0) if self.values else None
        return value, 12.5


class ShowTest(unittest.TestCase):
    def test_passes_everything_to_show_frame(self):
        p = Player(TIMES, highlights={YELLOW: [3]})
        p.k = 10
        ui = FakeUI()
        img = object()
        specs = [("Confirm", "confirm", ())] + p.transport()
        result = p.show(ui, img, specs, label="Review", marks=[(1.0, GREEN, "start")],
                        hotkeys={ord("m"): "menu"}, alert="oops", notice="note")
        self.assertEqual(result, (None, 12.5))
        (call,) = ui.calls
        self.assertIs(call["img"], img)
        self.assertEqual(call["specs"], specs)
        self.assertEqual((call["label"], call["alert"], call["notice"]), ("Review", "oops", "note"))
        self.assertEqual(call["hotkeys"], {ord("m"): "menu"})
        self.assertIsInstance(call["seek"], SeekState)
        self.assertEqual(call["seek"], seek_state(TIMES, 10, [(1.0, GREEN, "start")], {YELLOW: [3]}))

    def test_no_hotkeys_given_means_none(self):
        ui = FakeUI()
        Player(TIMES).show(ui, None, [])
        self.assertEqual(ui.calls[0]["hotkeys"], {})

    def test_caller_values_are_returned(self):
        p = Player(TIMES)
        self.assertEqual(p.show(FakeUI(["confirm"]), None, []), ("confirm", 12.5))
        self.assertEqual(p.k, 0)

    def test_transport_values_are_applied_not_returned(self):
        p = Player(TIMES)
        self.assertEqual(p.show(FakeUI(["forward"]), None, []), (None, 12.5))
        self.assertEqual(p.k, 1)
        self.assertTrue(p.paused)

    def test_dropped_mark(self):
        p = Player(TIMES)
        ui = FakeUI([("mark_drag", "stop", 0.2), ("mark_drop", "stop", 0.5)])
        self.assertEqual(p.show(ui, None, []), (None, 12.5))
        value, _ = p.show(ui, None, [])
        k = seek_index(TIMES, 0.5)
        self.assertEqual(value, ("mark", "stop", TIMES[k]))
        self.assertEqual(p.k, k)

    def test_wait_times(self):
        wall = FakeClock(100.0)
        with fake_time(wall):
            p = Player(TIMES)
            ui = FakeUI()
            p.show(ui, None, [])
            self.assertEqual(ui.calls[-1]["wait_ms"], 0.0)          # playing: paced by the clock
            p.k = 5
            p.show(ui, None, [])
            self.assertAlmostEqual(ui.calls[-1]["wait_ms"], 200.0)
            p.paused = True
            p.show(ui, None, [])
            self.assertEqual(ui.calls[-1]["wait_ms"], player_module._PAUSED_MS)
            ui.values.append(("seek", 0.5))
            p.show(ui, None, [])
            p.show(ui, None, [])
            self.assertEqual(ui.calls[-1]["wait_ms"], player_module._DRAGGING_MS)

    def test_restarts_the_clock_after_a_dialog(self):
        p = Player(TIMES)
        p._clock = mock.Mock(wraps=PlaybackClock())
        ui = FakeUI(open_dialog_on={1})
        p.show(ui, None, [])
        p._clock.restart.assert_not_called()
        p.show(ui, None, [])
        p._clock.restart.assert_called_once()
        p.show(ui, None, [])
        p._clock.restart.assert_called_once()

    def test_playing_on_after_a_dialog_does_not_skip(self):
        # After a long dialog the next frame's wait is measured from now, not
        # from before the dialog (which would be very negative).
        wall = FakeClock(100.0)
        with fake_time(wall):
            p = Player(TIMES)
            ui = FakeUI(open_dialog_on={0})
            p.show(ui, None, [])
            wall.now = 160.0
            p.advance()
            p.show(ui, None, [])
            self.assertEqual(ui.calls[-1]["wait_ms"], 0.0)


class TimeOnScreenTest(unittest.TestCase):
    def test_frame_on_screen_at_a_past_moment(self):
        wall = FakeClock(10.0)
        with fake_time(wall):
            p = Player(TIMES)
            p.paused = True
            ui = FakeUI()
            for k, t in ((0, 10.0), (1, 11.0), (2, 12.0)):
                p.k, wall.now = k, t
                p.show(ui, None, [])
        self.assertEqual(p.time_on_screen(11.5), TIMES[1])
        self.assertEqual(p.time_on_screen(12.0), TIMES[2])
        self.assertEqual(p.time_on_screen(11.0), TIMES[1])
        self.assertEqual(p.time_on_screen(99.0), TIMES[2])
        self.assertEqual(p.time_on_screen(5.0), TIMES[0])        # before the history: the oldest

    def test_nothing_shown_yet(self):
        p = Player(TIMES)
        p.k = 7
        self.assertEqual(p.time_on_screen(123.0), TIMES[7])

    def test_history_is_bounded(self):
        wall = FakeClock(0.0)
        with fake_time(wall):
            p = Player(TIMES)
            p.paused = True
            ui = FakeUI()
            for n in range(player_module._SHOWN_HISTORY + 50):
                wall.now = float(n)
                p.k = n % len(TIMES)
                p.show(ui, None, [])
        self.assertEqual(len(p._shown), player_module._SHOWN_HISTORY)


if __name__ == "__main__":
    unittest.main()
