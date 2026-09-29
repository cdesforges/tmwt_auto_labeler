"""Tests for the pose backend factory and the shared pose layout helpers (tmwt/pose/)."""

import pathlib
import types
import unittest

import numpy as np

from tmwt.pose import pose_backend, pose_common as pc, pose_rtmlib


class PoseBackendTest(unittest.TestCase):
    def test_heavier_model_uses_the_backends_function(self):
        backend = types.SimpleNamespace(heavier_model=lambda m: m + "+")
        self.assertEqual(pose_backend.heavier_model(backend, "small"), "small+")

    def test_heavier_model_without_the_function(self):
        self.assertIsNone(pose_backend.heavier_model(types.SimpleNamespace(), "balanced"))

    def test_model_strength(self):
        self.assertEqual(pose_backend.model_strength("balanced"), "balanced")
        self.assertEqual(pose_backend.model_strength("performance"), "performance")
        self.assertEqual(pose_backend.model_strength("models/pose_landmarker_full.task"),
                         "pose_landmarker_full")
        self.assertEqual(pose_backend.model_strength(pathlib.Path("/a/b/heavy.v2.onnx")), "heavy.v2")

    def test_unknown_backend(self):
        for name in ("", "openpose", "RTMLIB", None):
            with self.assertRaises(ValueError):
                pose_backend.get_backend(name)

    def test_rtmlib_backend(self):
        self.assertIs(pose_backend.get_backend("rtmlib"), pose_rtmlib)


class RtmlibHeavierModelTest(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(pose_rtmlib.heavier_model("lightweight"), "balanced")
        self.assertEqual(pose_rtmlib.heavier_model("balanced"), "performance")
        self.assertIsNone(pose_rtmlib.heavier_model("performance"))
        self.assertIsNone(pose_rtmlib.heavier_model("unknown"))

    def test_through_the_factory(self):
        self.assertEqual(pose_backend.heavier_model(pose_rtmlib, "balanced"), "performance")
        self.assertIsNone(pose_backend.heavier_model(pose_rtmlib, pose_rtmlib.DEFAULT_MODEL_PATH + "x"))


def halpe(n=26, score=0.9):
    """Halpe-26 keypoints at (10 * i, 5 * i) pixels, all with `score`."""
    return [(10.0 * i, 5.0 * i) for i in range(n)], [score] * n


class Halpe26Test(unittest.TestCase):
    W, H = 400.0, 200.0

    def test_mapping(self):
        kps, scores = halpe()
        pose = pc.halpe26_to_landmarks(kps, scores, self.W, self.H)
        self.assertEqual(len(pose), pc.NUM_LANDMARKS)
        for src, dst in pc.HALPE26_TO_LAYOUT.items():
            self.assertAlmostEqual(pose[dst].x, 10.0 * src / self.W)
            self.assertAlmostEqual(pose[dst].y, 5.0 * src / self.H)
            self.assertAlmostEqual(pose[dst].visibility, 0.9)
        filled = {i for i, lm in enumerate(pose) if lm is not None}
        self.assertEqual(filled, set(pc.HALPE26_TO_LAYOUT.values()))
        # Named checks of the foot points.
        self.assertAlmostEqual(pose[pc.LEFT_ANKLE_IDX].x, 150 / self.W)       # halpe 15
        self.assertAlmostEqual(pose[pc.LEFT_BIG_TOE_IDX].x, 200 / self.W)     # halpe 20
        self.assertAlmostEqual(pose[pc.RIGHT_SMALL_TOE_IDX].x, 230 / self.W)  # halpe 23
        self.assertAlmostEqual(pose[pc.RIGHT_HEEL_IDX].x, 250 / self.W)       # halpe 25

    def test_low_scores_are_missing(self):
        kps, scores = halpe()
        scores[15] = pc.MIN_KEYPOINT_SCORE - 0.01
        scores[16] = pc.MIN_KEYPOINT_SCORE                  # exactly at the threshold is kept
        pose = pc.halpe26_to_landmarks(kps, scores, self.W, self.H)
        self.assertIsNone(pose[pc.LEFT_ANKLE_IDX])
        self.assertIsNotNone(pose[pc.RIGHT_ANKLE_IDX])

    def test_numpy_input_and_extra_columns(self):
        kps = np.array([(10.0 * i, 5.0 * i, 0.5) for i in range(26)], dtype=np.float32)
        scores = np.full(26, 0.8, dtype=np.float32)
        pose = pc.halpe26_to_landmarks(kps, scores, self.W, self.H)
        self.assertIsInstance(pose[0].x, float)
        self.assertAlmostEqual(pose[pc.LEFT_ANKLE_IDX].y, 75 / self.H, places=6)
        self.assertEqual(pose[pc.LEFT_ANKLE_IDX].z, 0.0)

    def test_too_few_keypoints(self):
        kps, scores = halpe(n=17)                           # COCO-17: no feet
        pose = pc.halpe26_to_landmarks(kps, scores, self.W, self.H)
        self.assertIsNotNone(pose[pc.LEFT_ANKLE_IDX])
        for idx in pc.TOE_IDXS + (pc.LEFT_HEEL_IDX, pc.RIGHT_HEEL_IDX):
            self.assertIsNone(pose[idx])
        self.assertEqual(pc.halpe26_to_landmarks([], [], self.W, self.H), [None] * pc.NUM_LANDMARKS)


def pose_with(points):
    pose = [None] * pc.NUM_LANDMARKS
    for i, (x, y) in points.items():
        pose[i] = pc.Landmark(x, y)
    return pose


class AnkleMidpointTest(unittest.TestCase):
    SHAPE = (200, 400, 3)

    def test_two_ankles(self):
        pose = pose_with({27: (0.25, 0.5), 28: (0.5, 0.75)})
        self.assertEqual(pc.ankle_midpoint(pose, self.SHAPE), (150, 125))

    def test_one_ankle(self):
        self.assertEqual(pc.ankle_midpoint(pose_with({28: (0.5, 0.75)}), self.SHAPE), (200, 150))
        self.assertEqual(pc.ankle_midpoint(pose_with({27: (0.25, 0.5)}), self.SHAPE[:2]), (100, 100))

    def test_no_ankles(self):
        self.assertIsNone(pc.ankle_midpoint(pose_with({0: (0.5, 0.1)}), self.SHAPE))

    def test_landmark_px(self):
        self.assertEqual(pc.landmark_px(pc.Landmark(0.25, 0.5), 400, 200), (100.0, 100.0))
        self.assertIsNone(pc.landmark_px(None, 400, 200))


class DrawPoseTest(unittest.TestCase):
    # Left ankle (27) at (20, 50) and left big toe (31) at (80, 50) on a 100x100
    # image: one connection (27-31) between them.
    POINTS = {27: (0.2, 0.5), 31: (0.8, 0.5)}
    GREEN = (0, 255, 0)

    def draw(self, **kwargs):
        img = np.zeros((100, 100, 3), np.uint8)
        pc.draw_pose(img, pose_with(self.POINTS), **kwargs)
        return img

    def colour(self, img, x, y):
        return tuple(int(v) for v in img[y, x])

    def test_plain(self):
        img = self.draw()
        self.assertEqual(self.colour(img, 20, 50), self.GREEN)
        self.assertEqual(self.colour(img, 80, 50), self.GREEN)
        self.assertEqual(self.colour(img, 50, 50), self.GREEN)      # the line
        self.assertEqual(self.colour(img, 50, 20), (0, 0, 0))

    def test_highlight(self):
        img = self.draw(highlight={31})
        self.assertEqual(self.colour(img, 80, 50), pc.FLAGGED_COLOR)
        self.assertEqual(self.colour(img, 80, 56), pc.FLAGGED_COLOR)   # bigger point (radius 7)
        self.assertEqual(self.colour(img, 50, 50), pc.FLAGGED_COLOR)   # the line to it
        self.assertEqual(self.colour(img, 20, 50), self.GREEN)
        self.assertEqual(self.colour(self.draw(), 80, 56), (0, 0, 0))  # normal radius is 4

    def test_smoothed_takes_precedence(self):
        img = self.draw(highlight={31}, smoothed={31})
        self.assertEqual(self.colour(img, 80, 50), pc.SMOOTHED_COLOR)
        self.assertEqual(self.colour(img, 50, 50), pc.SMOOTHED_COLOR)
        self.assertEqual(self.colour(img, 20, 50), self.GREEN)

    def test_custom_colour(self):
        img = self.draw(color=(255, 0, 0))
        self.assertEqual(self.colour(img, 20, 50), (255, 0, 0))

    def test_face_points_other_than_the_nose_are_skipped(self):
        img = np.zeros((100, 100, 3), np.uint8)
        pc.draw_pose(img, pose_with({0: (0.2, 0.2), 2: (0.8, 0.2)}))
        self.assertEqual(self.colour(img, 20, 20), self.GREEN)
        self.assertEqual(self.colour(img, 80, 20), (0, 0, 0))

    def test_empty_pose_draws_nothing(self):
        img = np.zeros((100, 100, 3), np.uint8)
        pc.draw_pose(img, [None] * pc.NUM_LANDMARKS)
        self.assertEqual(int(img.sum()), 0)


if __name__ == "__main__":
    unittest.main()
