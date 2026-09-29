"""Tests for the pose plausibility check (tmwt/detection/pose_check.py)."""

import unittest

from tmwt.core.job import FrameResult
from tmwt.detection import pose_check
from tmwt.pose import pose_common as pc

W, H = 400, 800
FPS = 25.0


def standing_pose(x=200.0, feet_y=600.0, scale=1.0):
    """A plausible pose (pixels -> normalized), legs about 200 px long at scale 1."""
    pts = {
        23: (x - 20, feet_y - 200 * scale), 24: (x + 20, feet_y - 200 * scale),     # hips
        25: (x - 20, feet_y - 100 * scale), 26: (x + 20, feet_y - 100 * scale),     # knees
        27: (x - 20, feet_y), 28: (x + 20, feet_y),                                 # ankles
        29: (x - 22, feet_y + 5), 30: (x + 22, feet_y + 5),                         # heels
        31: (x - 20, feet_y + 25), 32: (x + 20, feet_y + 25),                       # big toes
        33: (x - 28, feet_y + 22), 34: (x + 28, feet_y + 22),                       # small toes
    }
    pose = [None] * pc.NUM_LANDMARKS
    for i, (px, py) in pts.items():
        pose[i] = pc.Landmark(px / W, py / H)
    return pose


def frames(n, poses=None):
    out = []
    for k in range(n):
        f = FrameResult(frame_idx=k, time_s=k / FPS, H=None)
        f.pose = poses[k] if poses else standing_pose()
        out.append(f)
    return out


def move(pose, idx, dx, dy):
    p = list(pose)
    p[idx] = pc.Landmark(p[idx].x + dx / W, p[idx].y + dy / H)
    return p


class PoseCheckTest(unittest.TestCase):
    def test_plausible_poses_are_not_flagged(self):
        self.assertEqual(pose_check.check(frames(100), (H, W)), [])

    def test_toe_thrown_across_the_floor_is_flagged(self):
        fs = frames(100)
        fs[50].pose = move(fs[50].pose, 33, 200, 0)     # left small toe 200 px away
        flags = pose_check.check(fs, (H, W))
        self.assertIn((50, 33, pose_check.FOOT_LENGTH), [(f.frame, f.landmark, f.kind) for f in flags])

    def test_foreshortened_shin_is_not_a_false_alarm(self):
        # A knee lifted toward the camera makes the shin look tiny for a frame;
        # the foot is compared with the typical leg, so that's fine.
        fs = frames(100)
        fs[50].pose = move(fs[50].pose, 25, 0, 95)       # knee almost on the ankle
        self.assertEqual(pose_check.check(fs, (H, W)), [])

    def test_one_frame_spike_is_flagged(self):
        fs = frames(100)
        fs[50].pose = move(fs[50].pose, 27, 250, 0)      # ankle jumps away for one frame
        kinds = {(f.frame, f.landmark): f.kind for f in pose_check.check(fs, (H, W))}
        self.assertEqual(kinds.get((50, 27)), pose_check.SPIKE)

    def test_edges_of_the_video_are_not_checked(self):
        fs = frames(100)
        for k in (2, 97):                                # within 1 s of the start / end
            fs[k].pose = move(fs[k].pose, 31, 250, 0)
        self.assertEqual(pose_check.check(fs, (H, W)), [])

    def test_describe_and_parse_round_trip(self):
        flags = {33: pose_check.FOOT_LENGTH, 27: pose_check.SPIKE}
        text = pose_check.describe(flags)
        self.assertEqual(text, "left_ankle:spike;left_small_toe:foot_length")
        self.assertEqual(pose_check.parse(text), flags)
        self.assertEqual(pose_check.parse(""), {})

    def test_summary(self):
        fs = frames(100)
        fs[40].pose = move(fs[40].pose, 31, 200, 0)
        fs[60].pose = move(fs[60].pose, 32, -200, 0)
        s = pose_check.summary(pose_check.check(fs, (H, W)))
        self.assertEqual(s["flagged_frames"], 2)
        self.assertEqual(s["examples"][0]["landmark"], "left_big_toe")


if __name__ == "__main__":
    unittest.main()
