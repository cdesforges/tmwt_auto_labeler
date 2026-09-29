"""
Following people through a video and choosing which one is the subject.

The pose model finds everyone in each frame (up to MAX_PEOPLE). Detections are
linked frame to frame into per-person tracks by nearest position, with the
allowed jump scaled by the person's apparent height (people look smaller far
from the camera) and short dropouts bridged.

The subject is then chosen in hindsight. In a 10 m walk test the subject walks
toward the camera, so they grow much larger in the picture over the video;
clinicians and bystanders standing around or walking beside don't grow the
same way. The track that grows the most is the subject. If that's wrong, the
user picks the right person at review (set_subject), and nothing has to be
re-analysed because every person's poses are kept.
"""

from dataclasses import dataclass, field

import numpy as np

from tmwt.pose import pose_common
from tmwt.detection import pose_check, pose_smoothing, tracking

# Most people detected per frame.
MAX_PEOPLE = 5
# A detection can join a track if it is within this many of the person's
# apparent heights of where the track was last seen (grows with the gap).
GATE_HEIGHTS = 0.6
# A track that goes unseen for longer than this ends.
MAX_GAP_S = 2.0
# Tracks shorter than this are ignored when choosing the subject or offering
# people to pick from (brief false detections).
MIN_TRACK_S = 1.0
# Heights at the start and end of a track are robust percentiles over this
# fraction of its samples.
_EDGE_FRACTION = 0.15
# Two detections whose landmark boxes overlap by more than this fraction of the
# smaller box are the same person detected twice — common when someone fills the
# frame right in front of the camera, where a partial second skeleton sits
# inside the full one. The later (lower-scoring) one is dropped.
DUPLICATE_OVERLAP = 0.7
# Floor for apparent heights, so tiny or partial skeletons don't shrink the gate
# to nothing.
_MIN_HEIGHT_PX = 30.0


@dataclass
class Track:
    """One person followed through the video."""
    id: int
    detections: dict = field(default_factory=dict)   # frame_idx -> index into FrameResult.people
    last_frame: int = -1
    last_center: tuple = None
    last_height: float = 0.0
    growth: float = 1.0                               # set by choose_subject

    def span_s(self, frames):
        idx = sorted(self.detections)
        return frames[idx[-1]].time_s - frames[idx[0]].time_s if idx else 0.0

    def first_frame(self):
        return min(self.detections)


def center(pose, frame_shape):
    """A person's position: the ankle midpoint, or the mean landmark if no ankles."""
    foot = pose_common.ankle_midpoint(pose, frame_shape)
    if foot is not None:
        return foot
    h, w = frame_shape[:2]
    pts = [(lm.x * w, lm.y * h) for lm in pose if lm is not None]
    if not pts:
        return None
    return tuple(np.mean(pts, axis=0))


def height(pose, frame_shape):
    """A person's apparent height in pixels (vertical extent of their landmarks)."""
    h = frame_shape[0]
    ys = [lm.y * h for lm in pose if lm is not None]
    return max(_MIN_HEIGHT_PX, max(ys) - min(ys)) if ys else _MIN_HEIGHT_PX


def _box(pose, frame_shape):
    """Bounding box (x0, y0, x1, y1) of a pose's landmarks, in pixels."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in pose if lm is not None]
    ys = [lm.y * h for lm in pose if lm is not None]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _overlap(a, b):
    """
    Intersection of two boxes as a fraction of the smaller one's area. A box
    with no area (a pose with one landmark, or all in a line) counts as fully
    overlapping only if it lies inside the other box.
    """
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if min(area_a, area_b) <= 0:
        small, big = (a, b) if area_a <= area_b else (b, a)
        inside = big[0] <= small[0] and small[2] <= big[2] and big[1] <= small[1] and small[3] <= big[3]
        return 1.0 if inside else 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min(area_a, area_b)


def distinct_people(poses, frame_shape):
    """
    Indices of the poses that are distinct people: a pose overlapping an
    earlier (higher-scoring) one by more than DUPLICATE_OVERLAP is a duplicate.
    """
    kept, boxes = [], []
    for k, pose in enumerate(poses):
        box = _box(pose, frame_shape)
        if box is None or any(_overlap(box, b) > DUPLICATE_OVERLAP for b in boxes):
            continue
        kept.append(k)
        boxes.append(box)
    return kept


def build_tracks(frames, frame_shape, fps):
    """
    Link every frame's detections (FrameResult.people) into per-person tracks.

    Duplicate detections of one person are dropped first (distinct_people).
    Then each frame, existing tracks and detections are paired greedily, closest
    first, within each track's gate; unpaired detections start new tracks.
    """
    tracks = []
    max_gap = max(1, int(round(MAX_GAP_S * fps)))
    for f in frames:
        dets = []
        for k in distinct_people(f.people, frame_shape):
            pose = f.people[k]
            c = center(pose, frame_shape)
            if c is not None:
                dets.append((k, c, height(pose, frame_shape)))
        live = [t for t in tracks if f.frame_idx - t.last_frame <= max_gap]

        pairs = []
        for t in live:
            gap = f.frame_idx - t.last_frame
            gate = GATE_HEIGHTS * t.last_height * min(1.0 + 0.25 * (gap - 1), 4.0)
            for k, c, hgt in dets:
                d = float(np.hypot(c[0] - t.last_center[0], c[1] - t.last_center[1]))
                if d <= gate:
                    pairs.append((d, t, k, c, hgt))
        used_tracks, used_dets = set(), set()
        for d, t, k, c, hgt in sorted(pairs, key=lambda p: p[0]):
            if t.id in used_tracks or k in used_dets:
                continue
            _extend(t, f.frame_idx, k, c, hgt)
            used_tracks.add(t.id)
            used_dets.add(k)
        for k, c, hgt in dets:
            if k not in used_dets:
                t = Track(id=len(tracks))
                _extend(t, f.frame_idx, k, c, hgt)
                tracks.append(t)
    return tracks


def _extend(track, frame_idx, det_index, c, hgt):
    track.detections[frame_idx] = det_index
    track.last_frame, track.last_center, track.last_height = frame_idx, c, hgt


def candidates(tracks, frames):
    """Tracks long enough to be a real person (see MIN_TRACK_S)."""
    return [t for t in tracks if t.span_s(frames) >= MIN_TRACK_S]


def choose_subject(tracks, frames, frame_shape):
    """
    The track most likely to be the walking subject: the one whose apparent
    height grows the most (walking toward the camera). Falls back to the
    longest track. Returns None if nobody was tracked.
    """
    pool = candidates(tracks, frames) or tracks
    if not pool:
        return None
    for t in pool:
        heights = [height(frames[i].people[k], frame_shape) for i, k in sorted(t.detections.items())]
        n = max(1, int(len(heights) * _EDGE_FRACTION))
        start = float(np.median(heights[:n]))
        peak = float(np.percentile(heights, 90))
        t.growth = peak / start if start > 0 else 1.0
    return max(pool, key=lambda t: (t.growth, len(t.detections)))


def set_subject(job, track):
    """
    Make `track` the subject: fill each frame's pose, body point and
    reference-frame positions from that person (None where they weren't seen),
    and check that person's pose for implausible points (pose_check.py).
    """
    if job.pose_edits:
        pose_smoothing.undo_all(job)   # smoothing belongs to the previous subject's pose
    job.subject = track.id if track is not None else None
    shape = job.info.first_frame.shape
    for f in job.frames:
        k = track.detections.get(f.frame_idx) if track is not None else None
        f.pose = f.people[k] if k is not None else None
        f.body_px = pose_common.ankle_midpoint(f.pose, shape) if f.pose is not None else None
        _locate_in_reference_frame(f, shape)
    # Plausibility of this person's pose (another subject has other flags).
    job.pose_flags = pose_check.check(job.frames, shape)
    flags = pose_check.by_frame(job.pose_flags)
    for k, f in enumerate(job.frames):
        f.pose_flags = flags.get(k, {})
        f.pose_smoothed = set()


def subject_start(job):
    """
    Where the subject stands at the start: their foot position in the first
    frame they're located, in reference-frame pixels (rounded). None if never.
    """
    for f in job.frames:
        if f.ref_foot is not None:
            return (int(round(f.ref_foot[0])), int(round(f.ref_foot[1])))
    return None


def _locate_in_reference_frame(result, frame_shape):
    """
    Fill result.ref_* with the subject's foot, head and ankles mapped back into
    the reference frame. Cleared unless both the ankle midpoint and the head
    (nose) were found — those two are what the vanishing-point fit needs.
    """
    result.ref_foot = result.ref_head = result.ref_left_ankle = result.ref_right_ankle = None
    if result.body_px is None:
        return
    h, w = frame_shape[:2]
    head = pose_common.landmark_px(result.pose[pose_common.NOSE_IDX], w, h)
    if head is None:
        return
    left = pose_common.landmark_px(result.pose[pose_common.LEFT_ANKLE_IDX], w, h)
    right = pose_common.landmark_px(result.pose[pose_common.RIGHT_ANKLE_IDX], w, h)
    foot, head, left_ref, right_ref = tracking.to_reference_frame(
        result.H, [result.body_px, head, left or result.body_px, right or result.body_px])
    result.ref_foot = foot
    result.ref_head = head
    result.ref_left_ankle = left_ref if left else None
    result.ref_right_ankle = right_ref if right else None


def people_on_screen(job, min_count=2):
    """
    A frame showing several candidate people, for the "pick the walker" screen:
    the earliest frame with the most candidates visible.

    Returns:
        (frame_idx, [(track, pose), ...]) or (None, []) if no frame shows at
        least min_count candidates.
    """
    pool = candidates(getattr(job, "tracks", []), job.frames)
    best_idx, best = None, []
    for f in job.frames:
        present = [(t, f.people[t.detections[f.frame_idx]]) for t in pool if f.frame_idx in t.detections]
        if len(present) > len(best):
            best_idx, best = f.frame_idx, present
    return (best_idx, best) if len(best) >= min_count else (None, [])
