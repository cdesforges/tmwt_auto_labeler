"""Tests for smoothing flagged pose points (tmwt/detection/pose_smoothing.py)."""

import unittest

from tmwt.core.job import FrameResult, VideoJob
from tmwt.detection import pose_check, pose_smoothing
from tmwt.pose import pose_common as pc

FPS = 25.0


def walking_job(n=100):
    """
    A subject walking down the frame: the left ankle moves 0.004 per frame,
    with the left small toe always 0.02 to its side and 0.01 below it; the
    left knee sits 0.1 above the ankle.
    """
    job = VideoJob(path="x.mp4", output_path="", name="x.mp4")
    for k in range(n):
        f = FrameResult(frame_idx=k, time_s=k / FPS, H=None)
        pose = [None] * pc.NUM_LANDMARKS
        ay = 0.3 + 0.004 * k
        pose[27] = pc.Landmark(0.5, ay)            # left ankle
        pose[33] = pc.Landmark(0.52, ay + 0.01)    # left small toe
        pose[25] = pc.Landmark(0.5, ay - 0.1)      # left knee
        f.pose = pose
        job.frames.append(f)
    return job


def flag(job, k, idx, kind=pose_check.FOOT_LENGTH):
    job.frames[k].pose_flags[idx] = kind


class SmoothingTest(unittest.TestCase):
    def test_toe_is_rebuilt_on_its_ankle(self):
        job = walking_job()
        for k in (50, 51, 52):                     # a thrown toe for three frames
            job.frames[k].pose[33] = pc.Landmark(0.9, 0.1)
            flag(job, k, 33)
        edits, failed = pose_smoothing.smooth_flagged(job)
        self.assertEqual((len(edits), failed), (3, 0))
        for k in (50, 51, 52):
            toe, ankle = job.frames[k].pose[33], job.frames[k].pose[27]
            self.assertAlmostEqual(toe.x - ankle.x, 0.02, places=6)
            self.assertAlmostEqual(toe.y - ankle.y, 0.01, places=6)
            self.assertEqual(job.frames[k].pose_smoothed, {33})
        self.assertEqual(edits[0].original.x, 0.9)   # the original is kept

    def test_other_points_are_interpolated_in_position(self):
        job = walking_job()
        job.frames[40].pose[25] = pc.Landmark(0.1, 0.9)   # knee spike
        flag(job, 40, 25, pose_check.SPIKE)
        pose_smoothing.smooth_flagged(job)
        knee = job.frames[40].pose[25]
        self.assertAlmostEqual(knee.x, 0.5, places=6)
        self.assertAlmostEqual(knee.y, 0.3 + 0.004 * 40 - 0.1, places=6)

    def test_no_good_frames_nearby_leaves_the_point(self):
        job = walking_job()
        for k in range(30, 70):                     # flagged for 1.6 s: too long to bridge
            flag(job, k, 25, pose_check.SPIKE)
        edits, failed = pose_smoothing.smooth_flagged(job)
        self.assertTrue(failed > 0)
        middle = job.frames[50]
        self.assertNotIn(25, middle.pose_smoothed)

    def test_undo_puts_the_originals_back(self):
        job = walking_job()
        job.frames[50].pose[33] = pc.Landmark(0.9, 0.1)
        flag(job, 50, 33)
        edits, _ = pose_smoothing.smooth_flagged(job)
        pose_smoothing.undo(job, edits)
        self.assertEqual(job.frames[50].pose[33].x, 0.9)
        self.assertEqual(job.frames[50].pose_smoothed, set())
        self.assertEqual(job.pose_edits, [])

    def test_smoothing_twice_does_not_redo_points(self):
        job = walking_job()
        job.frames[50].pose[33] = pc.Landmark(0.9, 0.1)
        flag(job, 50, 33)
        pose_smoothing.smooth_flagged(job)
        edits, failed = pose_smoothing.smooth_flagged(job)
        self.assertEqual((edits, failed), ([], 0))


if __name__ == "__main__":
    unittest.main()
