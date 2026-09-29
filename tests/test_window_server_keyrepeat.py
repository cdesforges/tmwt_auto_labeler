"""
Tests for held arrow keys repeating in the window process
(tmwt/ui/window_server.py, _KeyRepeat), with SDL's dummy video driver and a
patched clock: no window is opened.

window_server must never share a process with OpenCV (their bundled SDL2
builds collide on macOS), and other test modules import OpenCV. So under a
normal test run, KeyRepeatInCleanProcessTest re-runs KeyRepeatTest in a fresh
interpreter that only imports pygame; KeyRepeatTest itself runs only there.
"""

import importlib.util
import os
import subprocess
import sys
import unittest
from unittest import mock

_CHILD_ENV = "TMWT_KEYREPEAT_TEST_CHILD"
_IN_CHILD = os.environ.get(_CHILD_ENV) == "1"
_HAVE_PYGAME = importlib.util.find_spec("pygame") is not None
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _IN_CHILD:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    import pygame
    from tmwt.ui import window_server
    from tmwt.ui.window import KEY_LEFT, KEY_RIGHT


class FakeTime:
    """monotonic() and time() for window_server, set by the test."""

    def __init__(self, now=1000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def time(self):
        return self.now + 5_000_000.0


@unittest.skipUnless(_IN_CHILD, "runs in its own process, without OpenCV (see KeyRepeatInCleanProcessTest)")
class KeyRepeatTest(unittest.TestCase):
    def setUp(self):
        self.assertNotIn("cv2", sys.modules)
        self.clock = FakeTime()
        patcher = mock.patch.object(window_server, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.repeat = window_server._KeyRepeat()

    def press(self, key):
        self.repeat.handle(pygame.event.Event(pygame.KEYDOWN, key=key))

    def release(self, key):
        self.repeat.handle(pygame.event.Event(pygame.KEYUP, key=key))

    def at(self, t):
        """Repeats due `t` seconds after the start."""
        self.clock.now = 1000.0 + t
        return self.repeat.due()

    def test_no_repeat_before_the_delay(self):
        self.press(pygame.K_LEFT)
        self.assertEqual(self.at(0.0), [])
        self.assertEqual(self.at(window_server._KeyRepeat.DELAY_S - 0.001), [])

    def test_one_repeat_per_interval_after_the_delay(self):
        delay, interval = window_server._KeyRepeat.DELAY_S, window_server._KeyRepeat.INTERVAL_S
        self.press(pygame.K_RIGHT)
        first = self.at(delay)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["key"], KEY_RIGHT)
        self.assertEqual(first[0]["repeat"], self.clock.time())
        self.assertEqual(self.at(delay + interval / 2), [])
        self.assertEqual(len(self.at(delay + interval)), 1)
        self.assertEqual(self.at(delay + interval * 1.5), [])
        self.assertEqual(len(self.at(delay + interval * 2)), 1)

    def test_a_late_check_sends_one_repeat_not_a_backlog(self):
        self.press(pygame.K_LEFT)
        self.assertEqual(len(self.at(2.0)), 1)

    def test_after_a_late_repeat_the_next_waits_an_interval(self):
        interval = window_server._KeyRepeat.INTERVAL_S
        self.press(pygame.K_LEFT)
        self.assertEqual(len(self.at(2.0)), 1)
        self.assertEqual(self.at(2.005), [])
        self.assertEqual(len(self.at(2.0 + interval)), 1)

    def test_stops_on_release(self):
        self.press(pygame.K_LEFT)
        self.assertEqual(len(self.at(0.5)), 1)
        self.release(pygame.K_LEFT)
        self.assertEqual(self.at(1.0), [])

    def test_release_of_another_key_keeps_repeating(self):
        self.press(pygame.K_LEFT)
        self.release(pygame.K_a)
        self.assertEqual(len(self.at(0.5)), 1)

    def test_both_arrows(self):
        self.press(pygame.K_LEFT)
        self.press(pygame.K_RIGHT)
        self.assertEqual(sorted(m["key"] for m in self.at(0.5)), [KEY_LEFT, KEY_RIGHT])
        self.release(pygame.K_LEFT)
        self.assertEqual([m["key"] for m in self.at(1.0)], [KEY_RIGHT])

    def test_pressing_again_restarts_the_delay(self):
        self.press(pygame.K_LEFT)
        self.clock.now = 1000.2
        self.press(pygame.K_LEFT)
        self.assertEqual(self.at(0.4), [])
        self.assertEqual(len(self.at(0.5)), 1)

    def test_cleared_on_focus_loss(self):
        self.press(pygame.K_LEFT)
        self.press(pygame.K_RIGHT)
        self.repeat.handle(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        self.assertEqual(self.at(1.0), [])

    def test_only_arrow_keys_repeat(self):
        for key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE, pygame.K_a, pygame.K_UP):
            self.press(key)
        self.assertEqual(self.at(5.0), [])

    def test_other_events_are_ignored(self):
        self.press(pygame.K_LEFT)
        self.repeat.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(0, 0)))
        self.assertEqual(len(self.at(0.5)), 1)


@unittest.skipIf(_IN_CHILD, "this is the child process")
@unittest.skipUnless(_HAVE_PYGAME, "pygame isn't installed")
class KeyRepeatInCleanProcessTest(unittest.TestCase):
    def test_key_repeat_in_a_clean_process(self):
        env = dict(os.environ, **{_CHILD_ENV: "1", "SDL_VIDEODRIVER": "dummy",
                                  "PYGAME_HIDE_SUPPORT_PROMPT": "1"})
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v",
                               "tests.test_window_server_keyrepeat.KeyRepeatTest"],
                              cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("skipped", proc.stderr)
        self.assertIn("OK", proc.stderr)


if __name__ == "__main__":
    unittest.main()
