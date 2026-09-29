"""Tests for following people and choosing the walking subject (tmwt/detection/people.py)."""

import unittest

import numpy as np

from tmwt.core.job import FrameResult, VideoJob
from tmwt.core.video_io import VideoInfo
from tmwt.detection import people, pose_smoothing
from tmwt.pose import pose_common as pc

W, H = 800, 800
SHAPE = (H, W, 3)
FPS = 25.0


def person(x, feet_y, scale=1.0, nose=True):
    """A plausible standing pose, about 340 px tall at scale 1, feet at feet_y."""
    s = scale
    pts = {
        23: (x - 20 * s, feet_y - 200 * s), 24: (x + 20 * s, feet_y - 200 * s),   # hips
        25: (x - 20 * s, feet_y - 100 * s), 26: (x + 20 * s, feet_y - 100 * s),   # knees
        27: (x - 20 * s, feet_y), 28: (x + 20 * s, feet_y),                       # ankles
        29: (x - 22 * s, feet_y + 5 * s), 30: (x + 22 * s, feet_y + 5 * s),       # heels
        31: (x - 20 * s, feet_y + 25 * s), 32: (x + 20 * s, feet_y + 25 * s),     # big toes
    }
    if nose:
        pts[0] = (x, feet_y - 340 * s)
    pose = [None] * pc.NUM_LANDMARKS
    for i, (px, py) in pts.items():
        pose[i] = pc.Landmark(px / W, py / H)
    return pose


def frames_of(people_per_frame):
    return [FrameResult(frame_idx=k, time_s=k / FPS, H=np.eye(3), people=list(p))
            for k, p in enumerate(people_per_frame)]


def job_of(frames):
    job = VideoJob(path="x.mp4", output_path="", name="x.mp4")
    job.info = VideoInfo(first_frame_idx=0, crop=None, fps=FPS, total_frames=len(frames),
                         first_frame=np.zeros(SHAPE, np.uint8))
    job.frames = frames
    return job


def walker(k, n):
    """The subject walking toward the camera: grows from scale 0.4 to 1.0."""
    s = 0.4 + 0.6 * k / (n - 1)
    return person(300, 300 + 400 * s, scale=s)


def bystander(k):
    """Someone standing still at the side."""
    return person(650, 600, scale=0.8)


def detections_by_track(tracks):
    return sorted(sorted(t.detections.items()) for t in tracks)


class BuildTracksTest(unittest.TestCase):
    def test_two_people_moving_apart_stay_separate(self):
        # They start 60 px apart (well within each other's gate), and their
        # order in the detection list swaps every frame.
        n = 50
        fs = []
        for k in range(n):
            a, b = person(370 - 3 * k, 600, 0.6), person(430 + 3 * k, 600, 0.6)
            fs.append([a, b] if k % 2 == 0 else [b, a])
        tracks = people.build_tracks(frames_of(fs), SHAPE, FPS)
        self.assertEqual(len(tracks), 2)
        first = tracks[0]
        self.assertEqual(len(first.detections), n)
        self.assertEqual([first.detections[k] for k in range(4)], [0, 1, 0, 1])
        self.assertLess(first.last_center[0], 300)

    def test_short_dropout_is_bridged(self):
        fs = [[person(400, 600)] if not 20 <= k < 30 else [] for k in range(60)]   # 0.4 s unseen
        tracks = people.build_tracks(frames_of(fs), SHAPE, FPS)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].detections), 50)

    def test_long_dropout_starts_a_new_track(self):
        gap = int(people.MAX_GAP_S * FPS) + 5
        fs = [[person(400, 600)] if not 10 <= k < 10 + gap else [] for k in range(gap + 30)]
        tracks = people.build_tracks(frames_of(fs), SHAPE, FPS)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[1].first_frame(), 10 + gap)

    def test_jump_beyond_the_gate_starts_a_new_track(self):
        fs = [[person(100 if k < 10 else 700, 600)] for k in range(20)]
        self.assertEqual(len(people.build_tracks(frames_of(fs), SHAPE, FPS)), 2)

    def test_duplicate_detection_is_merged(self):
        # A partial second skeleton (legs only) inside the full one.
        legs = person(400, 600)
        for i in (0, 23, 24):
            legs[i] = None
        fs = [[person(400, 600), legs] for _ in range(30)]
        tracks = people.build_tracks(frames_of(fs), SHAPE, FPS)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(set(tracks[0].detections.values()), {0})

    def test_empty_frames_and_empty_poses(self):
        self.assertEqual(people.build_tracks([], SHAPE, FPS), [])
        fs = frames_of([[], [[None] * pc.NUM_LANDMARKS], []])
        self.assertEqual(people.build_tracks(fs, SHAPE, FPS), [])

    def test_center_and_height(self):
        pose = person(400, 600)
        self.assertEqual(people.center(pose, SHAPE), (400, 600))
        self.assertAlmostEqual(people.height(pose, SHAPE), 365.0, places=3)   # nose to toes
        no_ankles = [None] * pc.NUM_LANDMARKS
        no_ankles[0], no_ankles[11] = pc.Landmark(0.25, 0.25), pc.Landmark(0.75, 0.25)
        np.testing.assert_allclose(people.center(no_ankles, SHAPE), (400, 200))
        self.assertEqual(people.height(no_ankles, SHAPE), people._MIN_HEIGHT_PX)
        self.assertIsNone(people.center([None] * pc.NUM_LANDMARKS, SHAPE))

    def test_distinct_people(self):
        far_apart = [person(100, 600, 0.5), person(600, 600, 0.5)]
        self.assertEqual(people.distinct_people(far_apart, SHAPE), [0, 1])
        self.assertEqual(people.distinct_people([[None] * pc.NUM_LANDMARKS, far_apart[0]], SHAPE), [1])
        self.assertEqual(people.distinct_people([], SHAPE), [])

    # A one-landmark detection far from anyone is its own person, not a
    # duplicate (it would otherwise drop the walker from the frame).
    def test_single_point_detection_far_away_is_not_a_duplicate(self):
        lone = [None] * pc.NUM_LANDMARKS
        lone[pc.NOSE_IDX] = pc.Landmark(0.05, 0.05)       # top-left corner, far from anyone
        self.assertEqual(people.distinct_people([lone, person(600, 600)], SHAPE), [0, 1])

    def test_single_point_detection_inside_a_person_is_a_duplicate(self):
        walker = person(600, 600)
        inside = [None] * pc.NUM_LANDMARKS
        inside[pc.NOSE_IDX] = walker[23]                   # on the walker's hip
        self.assertEqual(people.distinct_people([walker, inside], SHAPE), [0])


class ChooseSubjectTest(unittest.TestCase):
    def test_the_person_who_grows_is_chosen(self):
        n = 75
        fs = frames_of([[bystander(k), walker(k, n)] for k in range(n)])
        tracks = people.build_tracks(fs, SHAPE, FPS)
        self.assertEqual(len(tracks), 2)
        subject = people.choose_subject(tracks, fs, SHAPE)
        self.assertEqual(set(subject.detections.values()), {1})
        self.assertGreater(subject.growth, 2.0)
        other = [t for t in tracks if t is not subject][0]
        self.assertAlmostEqual(other.growth, 1.0)

    def test_brief_false_detection_is_ignored(self):
        # Someone who "grows" a lot, but for only 0.4 s.
        n = 75
        fs = [[bystander(k)] for k in range(n)]
        for k in range(10):
            fs[k].append(person(100, 300 + 40 * k, scale=0.2 + 0.1 * k))
        frames = frames_of(fs)
        tracks = people.build_tracks(frames, SHAPE, FPS)
        subject = people.choose_subject(tracks, frames, SHAPE)
        self.assertEqual(len(subject.detections), n)

    def test_short_tracks_are_used_when_there_is_nothing_else(self):
        frames = frames_of([[person(400, 600)] for _ in range(5)])
        tracks = people.build_tracks(frames, SHAPE, FPS)
        self.assertIs(people.choose_subject(tracks, frames, SHAPE), tracks[0])

    def test_nobody(self):
        self.assertIsNone(people.choose_subject([], frames_of([[], []]), SHAPE))


class SetSubjectTest(unittest.TestCase):
    N = 75

    def setUp(self):
        self.job = job_of(frames_of([[bystander(k), walker(k, self.N)] for k in range(self.N)]))
        self.job.tracks = people.build_tracks(self.job.frames, SHAPE, FPS)
        self.walker_track = people.choose_subject(self.job.tracks, self.job.frames, SHAPE)
        self.bystander_track = [t for t in self.job.tracks if t is not self.walker_track][0]

    def test_fills_the_subject(self):
        job = self.job
        people.set_subject(job, self.walker_track)
        self.assertEqual(job.subject, self.walker_track.id)
        f = job.frames[10]
        self.assertIs(f.pose, f.people[1])
        self.assertEqual(f.body_px, pc.ankle_midpoint(f.people[1], SHAPE))
        self.assertEqual(f.ref_foot, (float(f.body_px[0]), float(f.body_px[1])))
        self.assertIsNotNone(f.ref_head)
        self.assertIsNotNone(f.ref_left_ankle)
        self.assertEqual(len(job.track), self.N)

    def test_unseen_frames_are_cleared(self):
        job = self.job
        track = people.Track(id=7, detections={k: 1 for k in range(5, self.N)})
        people.set_subject(job, track)
        self.assertIsNone(job.frames[0].pose)
        self.assertIsNone(job.frames[0].body_px)
        self.assertIsNone(job.frames[0].ref_foot)
        self.assertEqual(people.subject_start(job), tuple(int(round(v)) for v in job.frames[5].ref_foot))

    def test_no_head_means_no_reference_position(self):
        job = self.job
        for f in job.frames:
            f.people[1] = list(f.people[1])
            f.people[1][pc.NOSE_IDX] = None
        people.set_subject(job, self.walker_track)
        self.assertIsNotNone(job.frames[10].body_px)
        self.assertIsNone(job.frames[10].ref_foot)
        self.assertEqual(job.track, [])
        self.assertIsNone(people.subject_start(job))

    def test_none_track(self):
        job = self.job
        people.set_subject(job, None)
        self.assertIsNone(job.subject)
        self.assertTrue(all(f.pose is None for f in job.frames))
        self.assertEqual(job.pose_flags, [])

    def test_reference_positions_undo_the_camera_drift(self):
        job = self.job
        shift = np.array([[1, 0, 10], [0, 1, -20], [0, 0, 1]], dtype=np.float64)
        job.frames[10].H = shift
        people.set_subject(job, self.walker_track)
        f = job.frames[10]
        self.assertAlmostEqual(f.ref_foot[0], f.body_px[0] - 10, places=3)
        self.assertAlmostEqual(f.ref_foot[1], f.body_px[1] + 20, places=3)

    def thrown_toe(self):
        """Throw the walker's left big toe across the floor in frame 40."""
        pose = list(self.job.frames[40].people[1])
        pose[31] = pc.Landmark(pose[31].x + 0.4, pose[31].y)
        self.job.frames[40].people[1] = pose
        return pose

    def test_pose_flags_follow_the_subject(self):
        job = self.job
        self.thrown_toe()
        people.set_subject(job, self.walker_track)
        self.assertIn((40, 31), [(fl.frame, fl.landmark) for fl in job.pose_flags])
        self.assertIn(31, job.frames[40].pose_flags)
        people.set_subject(job, self.bystander_track)
        self.assertEqual(job.pose_flags, [])
        self.assertEqual(job.frames[40].pose_flags, {})

    def test_switching_subject_undoes_smoothing(self):
        job = self.job
        thrown = self.thrown_toe()
        original_x = thrown[31].x
        people.set_subject(job, self.walker_track)
        edits, _ = pose_smoothing.smooth_flagged(job)
        self.assertTrue(edits)
        self.assertNotEqual(job.frames[40].people[1][31].x, original_x)   # smoothed in place
        people.set_subject(job, self.bystander_track)
        self.assertEqual(job.pose_edits, [])
        self.assertEqual(job.frames[40].people[1][31].x, original_x)      # the walker's data is back
        self.assertEqual(job.frames[40].pose_smoothed, set())
        # Choosing the walker again flags the original point again.
        people.set_subject(job, self.walker_track)
        self.assertIn(31, job.frames[40].pose_flags)


class PeopleOnScreenTest(unittest.TestCase):
    def test_earliest_frame_with_the_most_people(self):
        n = 75
        fs = [[walker(k, n)] + ([bystander(k)] if k >= 10 else []) for k in range(n)]
        job = job_of(frames_of(fs))
        job.tracks = people.build_tracks(job.frames, SHAPE, FPS)
        idx, present = people.people_on_screen(job)
        self.assertEqual(idx, 10)
        self.assertEqual(len(present), 2)
        self.assertEqual({id(pose) for _, pose in present}, {id(p) for p in job.frames[10].people})

    def test_one_person_is_not_enough(self):
        job = job_of(frames_of([[walker(k, 50)] for k in range(50)]))
        job.tracks = people.build_tracks(job.frames, SHAPE, FPS)
        self.assertEqual(people.people_on_screen(job), (None, []))
        idx, present = people.people_on_screen(job, min_count=1)
        self.assertEqual((idx, len(present)), (0, 1))

    def test_brief_detections_do_not_count(self):
        fs = [[walker(k, 50)] + ([bystander(k)] if k < 5 else []) for k in range(50)]
        job = job_of(frames_of(fs))
        job.tracks = people.build_tracks(job.frames, SHAPE, FPS)
        self.assertEqual(people.people_on_screen(job), (None, []))

    def test_no_tracks(self):
        self.assertEqual(people.people_on_screen(job_of(frames_of([[], []]))), (None, []))


if __name__ == "__main__":
    unittest.main()
