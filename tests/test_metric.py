"""Tests for line fitting, the vanishing point and perspective distances (tmwt/measurement/metric.py)."""

import unittest

import numpy as np

from tmwt.measurement import metric

# A pinhole camera held level at 1.5 m, looking straight down the course (the
# Z axis): the walking direction's vanishing point is the principal point.
F, CX, CY = 500.0, 320.0, 300.0
CAMERA_HEIGHT = 1.5


def project(x, y, z):
    """World point (x sideways, y down from the camera, z forward) -> image pixels."""
    return (CX + F * x / z, CY + F * y / z)


def walk(n=40, x=0.3, z_from=12.0, z_to=2.0, noise=0.0, seed=0):
    """Foot and head (1.7 m tall person) image tracks of a straight walk toward the camera."""
    rng = np.random.default_rng(seed)
    feet, heads = [], []
    for z in np.linspace(z_from, z_to, n):
        fx, fy = project(x, CAMERA_HEIGHT, z)
        hx, hy = project(x, CAMERA_HEIGHT - 1.7, z)
        if noise:
            fx, fy, hx, hy = np.array((fx, fy, hx, hy)) + rng.normal(0, noise, 4)
        feet.append((fx, fy))
        heads.append((hx, hy))
    return feet, heads


class FitLineTest(unittest.TestCase):
    def test_collinear_points(self):
        pts = [(x, 2 * x + 1) for x in range(10)]
        line, straightness = metric.fit_line(pts)
        a, b, c = line
        self.assertAlmostEqual(np.hypot(a, b), 1.0)           # unit normal
        for x, y in pts:
            self.assertAlmostEqual(a * x + b * y + c, 0.0, places=9)
        self.assertAlmostEqual(straightness, 0.0, places=9)

    def test_vertical_line(self):
        line, _ = metric.fit_line([(5, y) for y in range(6)])
        a, b, c = line
        self.assertAlmostEqual(abs(a), 1.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(-c / a, 5.0)

    def test_scatter_is_less_straight(self):
        rng = np.random.default_rng(1)
        pts = [(x, 0.5 * x + rng.normal(0, 3)) for x in range(50)]
        _, noisy = metric.fit_line(pts)
        _, clean = metric.fit_line([(x, 0.5 * x) for x in range(50)])
        self.assertGreater(noisy, clean)
        self.assertLess(noisy, 1.0)

    def test_two_points_is_enough(self):
        line, straightness = metric.fit_line([(0, 0), (10, 10)])
        self.assertIsNotNone(line)
        self.assertAlmostEqual(line @ (5, 5, 1), 0.0)

    def test_degenerate(self):
        self.assertEqual(metric.fit_line([]), (None, None))
        self.assertEqual(metric.fit_line([(3, 4)]), (None, None))
        self.assertEqual(metric.fit_line([(3, 4)] * 5), (None, None))   # all the same point


class VanishingPointTest(unittest.TestCase):
    def test_straight_walk_recovers_the_vanishing_point(self):
        feet, heads = walk()
        V, info = metric.estimate_vanishing_point(feet, heads)
        self.assertIsNotNone(V, info.get("reason"))
        np.testing.assert_allclose(V, (CX, CY), atol=1e-6)
        self.assertEqual(info["n_points"], 40)
        self.assertLess(info["foot_straightness"], 1e-9)
        np.testing.assert_allclose(info["V"], V)

    def test_noisy_walk_is_close(self):
        feet, heads = walk(n=200, noise=0.5)
        V, _ = metric.estimate_vanishing_point(feet, heads)
        self.assertLess(np.hypot(V[0] - CX, V[1] - CY), 5.0)

    def test_walk_off_centre(self):
        # Walking diagonally across the view: the vanishing point moves sideways.
        feet, heads = [], []
        for z in np.linspace(12, 2, 30):
            x = 1.0 + 0.1 * z                        # direction (0.1, 0, 1) -> V at CX + 0.1 F
            feet.append(project(x, CAMERA_HEIGHT, z))
            heads.append(project(x, CAMERA_HEIGHT - 1.7, z))
        V, _ = metric.estimate_vanishing_point(feet, heads)
        np.testing.assert_allclose(V, (CX + 0.1 * F, CY), atol=1e-6)

    def test_too_few_points(self):
        feet, heads = walk(n=metric.MIN_TRACK_POINTS - 1)
        V, info = metric.estimate_vanishing_point(feet, heads)
        self.assertIsNone(V)
        self.assertIn(str(metric.MIN_TRACK_POINTS - 1), info["reason"])

    def test_uses_the_shorter_track(self):
        feet, heads = walk(n=30)
        V, info = metric.estimate_vanishing_point(feet, heads[:5])
        self.assertIsNone(V)
        self.assertEqual(info["n_points"], 5)
        V, info = metric.estimate_vanishing_point(feet, heads[:20])
        self.assertIsNotNone(V)
        self.assertEqual(info["n_points"], 20)

    def test_standing_still_is_degenerate(self):
        V, info = metric.estimate_vanishing_point([(100, 400)] * 20, [(100, 100)] * 20)
        self.assertIsNone(V)
        self.assertEqual(info["reason"], "degenerate line fit")

    def test_parallel_lines(self):
        # Walking across the view at a constant depth: the lines never meet.
        feet = [(x, 400.0) for x in range(0, 200, 10)]
        heads = [(x, 100.0) for x in range(0, 200, 10)]
        V, info = metric.estimate_vanishing_point(feet, heads)
        self.assertIsNone(V)
        self.assertIn("parallel", info["reason"])

    def test_empty(self):
        V, info = metric.estimate_vanishing_point([], [])
        self.assertIsNone(V)
        self.assertEqual(info["n_points"], 0)


class MetricAlongTest(unittest.TestCase):
    # A 10 m course on the floor straight down the view, from 12 m to 2 m away.
    FAR = project(0.0, CAMERA_HEIGHT, 12.0)
    NEAR = project(0.0, CAMERA_HEIGHT, 2.0)
    V = (CX, CY)

    def along(self, point):
        return metric.metric_along(point, self.FAR, self.NEAR, self.V, 10.0)

    def test_endpoints(self):
        self.assertAlmostEqual(self.along(self.FAR), 0.0)
        self.assertAlmostEqual(self.along(self.NEAR), 10.0)

    def test_perspective_is_corrected(self):
        # Halfway in metres is far from halfway in pixels.
        mid = project(0.0, CAMERA_HEIGHT, 7.0)
        self.assertAlmostEqual(self.along(mid), 5.0, places=6)
        self.assertLess(metric.rope_fraction(mid, self.FAR, self.NEAR), 0.2)
        for z in (11.0, 9.5, 4.0, 2.5):
            self.assertAlmostEqual(self.along(project(0.0, CAMERA_HEIGHT, z)), 12.0 - z, places=6)

    def test_behind_the_start_and_past_the_finish(self):
        self.assertAlmostEqual(self.along(project(0.0, CAMERA_HEIGHT, 14.0)), -2.0, places=6)
        self.assertAlmostEqual(self.along(project(0.0, CAMERA_HEIGHT, 1.5)), 10.5, places=6)

    def test_sideways_offset_is_ignored(self):
        # A point beside the rope at the same image height reads the same distance.
        x, y = project(0.0, CAMERA_HEIGHT, 7.0)
        self.assertAlmostEqual(self.along((x + 40, y)), self.along((x, y)), places=6)

    def test_monotonic_along_the_line(self):
        ys = np.linspace(self.FAR[1] - 20, self.NEAR[1] + 50, 60)
        d = [self.along((CX, y)) for y in ys]
        self.assertTrue(all(b > a for a, b in zip(d, d[1:])))

    def test_none_at_the_vanishing_point(self):
        self.assertIsNone(self.along(self.V))

    def test_none_for_coincident_endpoints(self):
        self.assertIsNone(metric.metric_along((1, 1), (5, 5), (5, 5), (0, 0), 10.0))


if __name__ == "__main__":
    unittest.main()
