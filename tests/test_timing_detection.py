"""Tests for line crossings and walk start / end detection (tmwt/measurement/timing.py)."""

import contextlib
import io
import unittest

import numpy as np

from tmwt.core.job import END_ANKLE_MIDPOINT, FrameResult, VideoJob
from tmwt.core.video_io import VideoInfo
from tmwt.detection import people, pose_check
from tmwt.measurement import onset, timing
from tmwt.pose import pose_common as pc

FPS = 25.0


def quiet(fn, *args, **kwargs):
    """Call fn with its console output suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def make_job(w, h, poses, H=np.eye(3)):
    """A job whose one person has `poses[k]` (normalized landmarks) in frame k."""
    job = VideoJob(path="x.mp4", output_path="", name="x.mp4")
    job.info = VideoInfo(first_frame_idx=0, crop=None, fps=FPS, total_frames=len(poses),
                         first_frame=np.zeros((h, w, 3), np.uint8))
    for k, pose in enumerate(poses):
        job.frames.append(FrameResult(frame_idx=k, time_s=k / FPS, H=H,
                                      people=[pose] if pose is not None else []))
    return job


def pose_from_px(points, w, h):
    pose = [None] * pc.NUM_LANDMARKS
    for i, (x, y) in points.items():
        pose[i] = pc.Landmark(x / w, y / h)
    return pose


# --- A flat walk down a vertical rope, linear in pixels ---------------------
# Rope from (200, 100) to (200, 700) in a 400x800 frame; the ankles move down
# 100 px/s from y=100 at t=0. Toes sit ahead of the ankles: left big toe +30 px,
# right big toe +20, small toes +10. So y=700 (the finish line) is crossed by
# the left big toe at 5.7 s, the right big toe at 5.8 s and the ankles at 6 s.

W1, H1 = 400, 800
FAR1, NEAR1 = (200, 100), (200, 700)
TOE_LEAD = {31: 30, 32: 20, 33: 10, 34: 10}


def linear_job(seconds=8.0, toes=True, ankles=True, behavior=None):
    poses = []
    for k in range(int(seconds * FPS)):
        y = 100 + 100 * k / FPS
        pts = {}
        if ankles:
            pts.update({27: (190, y), 28: (210, y)})
        if toes:
            pts.update({i: (190 if i % 2 else 210, y + d) for i, d in TOE_LEAD.items()})
        poses.append(pose_from_px(pts, W1, H1))
    job = make_job(W1, H1, poses, H=None)
    for f, pose in zip(job.frames, poses):
        f.pose = pose
        f.body_px = pc.ankle_midpoint(pose, (H1, W1)) if ankles else None
    job.far_ep, job.near_ep = FAR1, NEAR1
    if behavior:
        job.endpoint_behavior = behavior
    timing.apply_endpoints(job)
    return job


def crossings(job, level=timing.NEAR_T, after=None):
    return timing._Crossings(job, [f.time_s for f in job.frames]).crossing(level, after)


class FindCrossingTest(unittest.TestCase):
    TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_interpolates_between_samples(self):
        self.assertAlmostEqual(timing.find_crossing(self.TIMES, [0.0, 0.2, 0.6, 1.0, 1.4], 0.5), 1.75)

    def test_exactly_on_the_level_counts(self):
        self.assertAlmostEqual(timing.find_crossing(self.TIMES, [0.0, 0.5, 1.0, 1.0, 1.0], 0.5), 1.0)

    def test_first_crossing_wins(self):
        values = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.assertAlmostEqual(timing.find_crossing(self.TIMES, values, 0.5), 0.5)

    def test_after(self):
        values = [0.0, 1.0, 0.0, 1.0, 0.0]
        self.assertAlmostEqual(timing.find_crossing(self.TIMES, values, 0.5, after=2.5), 2.5)
        self.assertIsNone(timing.find_crossing(self.TIMES, values, 0.5, after=3.5))

    def test_gaps_are_bridged(self):
        values = [0.0, None, None, 1.0, 1.2]
        self.assertAlmostEqual(timing.find_crossing(self.TIMES, values, 0.5), 1.5)

    def test_never_crossing(self):
        self.assertIsNone(timing.find_crossing(self.TIMES, [0.0, 0.1, 0.2, 0.3, 0.4], 0.5))
        self.assertIsNone(timing.find_crossing(self.TIMES, [None] * 5, 0.5))
        self.assertIsNone(timing.find_crossing([], [], 0.5))

    def test_only_from_below(self):
        # Starting past the level and walking back over it isn't a crossing.
        self.assertIsNone(timing.find_crossing(self.TIMES, [1.0, 0.8, 0.6, 0.4, 0.2], 0.5))
        # Nor is starting already past it.
        self.assertIsNone(timing.find_crossing(self.TIMES, [0.6, 0.7, 0.8, 0.9, 1.0], 0.5))


class CrossingTiersTest(unittest.TestCase):
    def test_first_toe_crosses(self):
        self.assertAlmostEqual(crossings(linear_job()), 5.7, delta=0.01)

    def test_ankles_when_no_toes(self):
        self.assertAlmostEqual(crossings(linear_job(toes=False)), 6.0, delta=0.01)

    def test_midpoint_when_no_feet_seen(self):
        job = linear_job(toes=False)
        for f in job.frames:
            f.pose = [None] * pc.NUM_LANDMARKS    # the midpoint (t_smooth) is all that's left
        self.assertAlmostEqual(crossings(job), 6.0, delta=0.01)

    def test_ankle_midpoint_ignores_the_toes(self):
        job = linear_job(behavior=END_ANKLE_MIDPOINT)
        self.assertAlmostEqual(crossings(job), 6.0, delta=0.01)

    def test_start_line(self):
        # Everybody starts on the far line (t=0) and no one crosses it from below.
        self.assertIsNone(crossings(linear_job(), level=timing.FAR_T))
        self.assertAlmostEqual(crossings(linear_job(), level=0.5), 3.0 - 0.3, delta=0.01)

    def test_flagged_toe_is_left_out(self):
        job = linear_job()
        for f in job.frames[100:]:
            f.pose_flags = {31: pose_check.SPIKE}
        self.assertAlmostEqual(crossings(job), 5.8, delta=0.01)

    def test_smoothed_toe_counts_again(self):
        job = linear_job()
        for f in job.frames[100:]:
            f.pose_flags = {31: pose_check.SPIKE}
            f.pose_smoothed = {31}
        self.assertAlmostEqual(crossings(job), 5.7, delta=0.01)

    def test_position_at_is_the_leading_toe(self):
        job = linear_job()
        lines = timing._Crossings(job, [f.time_s for f in job.frames])
        self.assertAlmostEqual(lines.position_at(3.0), (400 + 30 - 100) / 600, delta=0.005)
        self.assertIsNone(lines.position_at(-1.0))

    def test_apply_endpoints_without_a_body(self):
        job = linear_job(ankles=False)
        self.assertTrue(all(f.t_along is None and f.t_smooth is None for f in job.frames))
        self.assertEqual(job.frames[0].far_ep, FAR1)


class ApplyEndpointsTest(unittest.TestCase):
    def test_homography_moves_the_endpoints(self):
        job = linear_job(seconds=1.0)
        shift = np.array([[1, 0, 5], [0, 1, 10], [0, 0, 1]], dtype=np.float64)
        job.frames[0].H = shift
        timing.apply_endpoints(job)
        self.assertEqual(job.frames[0].far_ep, (205, 110))
        self.assertEqual(job.frames[0].near_ep, (205, 710))
        self.assertEqual(job.frames[1].far_ep, FAR1)       # no homography: unchanged
        # The body point is measured against that frame's endpoints.
        self.assertAlmostEqual(job.frames[0].t_along, (100 - 110) / 600)


# --- A walk filmed with a level pinhole camera -------------------------------
# The subject stands 12 m from the camera, then walks toward it at 1.2 m/s; the
# course is 12 m -> 2 m, 0.3 m to one side. Big toes are 0.2 m ahead of the ankles.

W2, H2 = 640, 960
F, CX, CY = 500.0, 320.0, 300.0
SPEED = 1.2
Z_START = 12.0
X_OFF = 0.3        # the course runs 0.3 m to the side of the camera


def project(x, y, z):
    return (CX + F * x / z, CY + F * y / z)


def camera_pose(z):
    x = X_OFF
    pts = {0: project(x, -0.2, z),                                        # nose
           23: project(x - 0.1, 0.6, z), 24: project(x + 0.1, 0.6, z),    # hips
           25: project(x - 0.1, 1.05, z), 26: project(x + 0.1, 1.05, z),  # knees
           27: project(x - 0.1, 1.5, z), 28: project(x + 0.1, 1.5, z),    # ankles
           31: project(x - 0.1, 1.5, z - 0.2),                            # big toes
           32: project(x + 0.1, 1.5, z - 0.2)}
    return pose_from_px(pts, W2, H2)


def camera_job(still_s=2.0, walk_s=9.5):
    zs = [Z_START - SPEED * max(0.0, k / FPS - still_s)
          for k in range(int((still_s + walk_s) * FPS))]
    job = make_job(W2, H2, [camera_pose(z) for z in zs])
    track = people.Track(id=0, detections={k: 0 for k in range(len(zs))})
    people.set_subject(job, track)
    job.far_ep = people.subject_start(job)
    job.near_ep = tuple(int(round(v)) for v in project(X_OFF, 1.5, 2.0))
    job.far_ep_is_standing_spot = True
    return job


def finish_time(still_s):
    """When a big toe reaches 2 m from the camera."""
    return still_s + (Z_START - 2.2) / SPEED


class DetectWalkTimesTest(unittest.TestCase):
    def test_fixture_is_plausible(self):
        job = camera_job()
        self.assertEqual(job.pose_flags, [])
        self.assertEqual(len(job.track), len(job.frames))

    def test_distances_are_perspective_corrected(self):
        job = camera_job()
        dist, units = timing._distance_function(job, job.track)
        self.assertEqual(units, "perspective-corrected")
        self.assertAlmostEqual(dist(project(X_OFF, 1.5, 7.0)), 5.0, delta=0.1)   # endpoints are whole pixels
        dist, units = quiet(timing._distance_function, job, job.track[:5])
        self.assertEqual(units, "image-space")
        self.assertAlmostEqual(dist(job.near_ep), 10.0)

    def test_steady_walk(self):
        job = camera_job()
        quiet(timing.update_timing, job)
        self.assertAlmostEqual(job.walk_start, 2.0, delta=0.2)
        self.assertAlmostEqual(job.walk_end, finish_time(2.0), delta=0.03)
        self.assertEqual(job.timing_source, "auto")
        self.assertIn("foot movement", job.timing_detail)
        self.assertEqual(job.timing_note, "")

    def test_ankle_midpoint_finishes_later(self):
        job = camera_job()
        job.endpoint_behavior = END_ANKLE_MIDPOINT
        quiet(timing.update_timing, job)
        self.assertAlmostEqual(job.walk_end, 2.0 + (Z_START - 2.0) / SPEED, delta=0.03)

    def test_short_standstill_is_noted(self):
        job = camera_job(still_s=0.8)
        quiet(timing.update_timing, job)
        self.assertIsNotNone(job.walk_start)
        self.assertEqual(job.timing_note, timing.NOTE_SHORT_STANDSTILL)

    def test_no_standstill_means_no_start(self):
        job = camera_job(still_s=0.0)
        quiet(timing.update_timing, job)
        self.assertIsNone(job.walk_start)
        self.assertIsNone(job.walk_end)                  # the end needs a start
        self.assertEqual(job.timing_note, timing.NOTE_NO_START.format(reason=onset.START_TOO_SOON))

    def test_no_track_means_no_start(self):
        job = camera_job()
        for f in job.frames:
            f.ref_foot = None
        timing.apply_endpoints(job)
        start, end, detail, note = quiet(timing.detect_walk_times, job)
        self.assertIsNone(start)
        self.assertTrue(note.startswith("Start not found: "))

    def test_clicked_start_line_behind_the_subject_is_crossed(self):
        job = camera_job()
        job.far_ep = tuple(int(round(v)) for v in project(X_OFF, 1.5, 10.0))
        job.far_ep_is_standing_spot = False
        quiet(timing.update_timing, job)
        # The first big toe reaches 10 m from the camera.
        self.assertAlmostEqual(job.walk_start, 2.0 + (Z_START - 10.2) / SPEED, delta=0.05)
        self.assertEqual(job.timing_detail, "start-line crossing")
        self.assertEqual(job.timing_note, "")

    def test_clicked_start_line_already_passed(self):
        job = camera_job()
        job.far_ep_is_standing_spot = False              # far_ep is where they stand
        quiet(timing.update_timing, job)
        self.assertAlmostEqual(job.walk_start, 2.0, delta=0.2)
        self.assertTrue(job.timing_detail.endswith("(on/past start line)"))

    def test_clicked_start_line_decides_when_the_movement_is_unclear(self):
        job = camera_job(still_s=0.0)
        job.far_ep = tuple(int(round(v)) for v in project(X_OFF, 1.5, 10.0))
        job.far_ep_is_standing_spot = False
        quiet(timing.update_timing, job)
        self.assertAlmostEqual(job.walk_start, (Z_START - 10.2) / SPEED, delta=0.05)
        self.assertEqual(job.timing_detail, "start-line crossing")
        self.assertEqual(job.timing_note, "")
        self.assertIsNotNone(job.walk_end)

    def test_never_finishing(self):
        job = camera_job(walk_s=5.0)                    # stops recording at 6 m
        quiet(timing.update_timing, job)
        self.assertIsNotNone(job.walk_start)
        self.assertIsNone(job.walk_end)


if __name__ == "__main__":
    unittest.main()
