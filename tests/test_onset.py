"""Tests for hindsight walk-start detection with little standstill (tmwt/measurement/onset.py)."""

import unittest

import numpy as np

from tmwt.measurement import onset

FPS = 30.0


def walk(still_s, walk_s=6.0, speed=1.2, noise=0.01, seed=0):
    """
    Synthetic ankle distances: standing still for `still_s`, then walking at
    `speed` (the feet alternate, each moving half the time).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, still_s + walk_s, 1 / FPS)
    moving = np.clip(t - still_s, 0, None)
    left = speed * moving + 0.15 * np.sin(np.pi * moving / 0.55) * (moving > 0)
    right = speed * moving - 0.15 * np.sin(np.pi * moving / 0.55) * (moving > 0)
    left += rng.normal(0, noise, len(t))
    right += rng.normal(0, noise, len(t))
    return t, left, right, (left + right) / 2


class OnsetTest(unittest.TestCase):
    def test_long_standstill_is_not_flagged(self):
        t, l, r, m = walk(still_s=3.0)
        start, info = onset.find_walk_onset(t, l, r, m)
        self.assertAlmostEqual(start, 3.0, delta=0.2)
        self.assertFalse(info["short_standstill"])

    def test_short_standstill_still_finds_a_start_and_flags_it(self):
        # Like SV_10MWRT_string: the walk begins under a second into the video.
        t, l, r, m = walk(still_s=0.8)
        start, info = onset.find_walk_onset(t, l, r, m)
        self.assertIsNotNone(start, info.get("reason"))
        self.assertAlmostEqual(start, 0.8, delta=0.35)
        self.assertTrue(info["short_standstill"])

    def test_no_standstill_says_why(self):
        t, l, r, m = walk(still_s=0.0)
        start, info = onset.find_walk_onset(t, l, r, m)
        self.assertIsNone(start)
        self.assertEqual(info["reason"], onset.START_TOO_SOON)


if __name__ == "__main__":
    unittest.main()
