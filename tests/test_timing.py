"""Tests for line-crossing geometry and smoothing (metric.t_along, timing.smooth_t_along)."""

import unittest

from tmwt.measurement import metric, timing


class TAlongTest(unittest.TestCase):
    # A rope tilted in the frame, like REACH_AlbCtrl_10m_1: start (176, 392),
    # finish (112, 682).
    FAR, NEAR = (176, 392), (112, 682)

    def test_endpoints(self):
        self.assertAlmostEqual(metric.t_along(self.FAR, self.FAR, self.NEAR), 0.0)
        self.assertAlmostEqual(metric.t_along(self.NEAR, self.FAR, self.NEAR), 1.0)

    def test_finish_line_is_horizontal(self):
        # A foot beside the rope, level with the finish point, is on the finish
        # line however far to the side it is.
        for x in (0, 112, 250, 404):
            self.assertAlmostEqual(metric.t_along((x, 682), self.FAR, self.NEAR), 1.0)

    def test_tilted_rope_regression(self):
        # The toe that was counted late on REACH_AlbCtrl_10m_1: right of the
        # rope and already past the finish tape's level. At right angles to
        # the tilted rope it read as short of the line.
        toe = (230, 690)
        self.assertLess(metric.rope_fraction(toe, self.FAR, self.NEAR), 1.0)
        self.assertGreater(metric.t_along(toe, self.FAR, self.NEAR), 1.0)

    def test_sideways_rope_falls_back_to_projection(self):
        far, near = (0, 100), (300, 120)
        self.assertAlmostEqual(metric.t_along((150, 400), far, near),
                               metric.rope_fraction((150, 400), far, near))

    def test_degenerate(self):
        self.assertIsNone(metric.t_along((5, 5), (10, 10), (10, 10)))


class SmoothingTest(unittest.TestCase):
    def test_no_lag_on_a_steady_walk(self):
        # A constant-speed walk: zero-lag smoothing crosses the line when the
        # raw series does; a forwards-only EMA crosses later.
        times = [i / 25 for i in range(100)]
        raw = [t / 3.0 for t in times]                     # reaches 1.0 at 3.0 s
        crossing = timing.find_crossing(times, timing.smooth_t_along(raw), 1.0)
        lagged = timing.find_crossing(times, timing._ema(raw), 1.0)
        self.assertAlmostEqual(crossing, 3.0, delta=0.005)
        self.assertGreater(lagged, crossing + 0.005)

    def test_gaps_stay_gaps(self):
        out = timing.smooth_t_along([0.1, None, 0.3, None])
        self.assertIsNone(out[1])
        self.assertIsNone(out[3])
        self.assertIsNotNone(out[0])

    def test_empty(self):
        self.assertEqual(timing.smooth_t_along([]), [])


if __name__ == "__main__":
    unittest.main()
