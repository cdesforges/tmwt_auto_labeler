"""
Pose plausibility check: flags the subject's leg and foot points that can't be
right, so they can be re-estimated with a heavier model (processing.py), kept
out of the timing (timing.py), drawn in red at review and marked in the CSV.

Two checks, both per frame and scale-free (so they work near and far from the
camera):

  - foot_length: a foot point (heel, big toe, small toe) further from its
    ankle than FOOT_TO_LEG_MAX of the subject's leg length (hip to knee to
    ankle), taken as the median over the surrounding second. A real ankle-to-
    toe distance is about a fifth of the leg; three times that only catches
    points thrown somewhere else entirely (e.g. onto the floor beside the
    foot). The leg length is a median over time, not the current frame's,
    because a knee lifted toward the camera makes the shin look very short.
  - spike: a leg or foot point that jumps away for a single frame — far from
    where it was the frame before AND where it is the frame after, while
    those two agree. A real foot moves at most ~0.1 body heights per frame;
    the threshold is several times that.

The first and last EDGE_BUFFER_S of the video aren't checked: that's where
people step into or out of frame (and the camera gets picked up or covered),
so partial bodies there would raise alarms that don't affect the walk.

Nothing here changes any pose data: it only reports which points look wrong.
"""

from collections import namedtuple

import numpy as np


# Leg and foot landmarks checked, with the names used in reports and the CSV.
LANDMARK_NAMES = {
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_big_toe", 32: "right_big_toe",
    33: "left_small_toe", 34: "right_small_toe",
}
# (knee, ankle, foot points) per side.
_LEGS = ((25, 27, (29, 31, 33)), (26, 28, (30, 32, 34)))

# foot_length: a foot point further from the ankle than this many leg lengths
# (median over LEG_WINDOW_S around the frame).
FOOT_TO_LEG_MAX = 0.6
LEG_WINDOW_S = 1.0
# Seconds at the start and end of the video that aren't checked.
EDGE_BUFFER_S = 1.0
# spike: a jump bigger than this many body heights, to the frame before and the
# frame after, while those two frames are within half of it of each other.
SPIKE_HEIGHTS = 0.35

FOOT_LENGTH = "foot_length"
SPIKE = "spike"

# One flagged point: frame (index into the frames list), time, layout landmark
# index, and the check that flagged it.
Flag = namedtuple("Flag", "frame time_s landmark kind")


def check(frames, frame_shape):
    """
    Check the subject's pose in every frame (FrameResult.pose, None where the
    subject isn't seen), except the first and last EDGE_BUFFER_S.

    Returns:
        [Flag], in frame order.
    """
    h, w = frame_shape[:2]
    pts = [_points(f.pose, w, h) for f in frames]
    legs = _typical_leg_lengths(frames, pts)
    flags = []
    if not frames:
        return flags
    first, last = frames[0].time_s + EDGE_BUFFER_S, frames[-1].time_s - EDGE_BUFFER_S
    for k, f in enumerate(frames):
        if pts[k] is None or not first <= f.time_s <= last:
            continue
        for idx in _foot_length_flags(pts[k], legs[k]):
            flags.append(Flag(k, f.time_s, idx, FOOT_LENGTH))
        if 0 < k < len(frames) - 1:
            for idx in _spike_flags(pts[k - 1], pts[k], pts[k + 1]):
                if not any(fl.frame == k and fl.landmark == idx for fl in flags):
                    flags.append(Flag(k, f.time_s, idx, SPIKE))
    return flags


def by_frame(flags):
    """{frame index: {landmark index: kind}} for looking flags up while drawing or timing."""
    out = {}
    for fl in flags:
        out.setdefault(fl.frame, {})[fl.landmark] = fl.kind
    return out


def summary(flags, limit=8):
    """
    A JSON-friendly summary: the number of frames and points flagged, and the
    first `limit` flags as {"time_s", "landmark", "kind"}.
    """
    return {
        "flagged_frames": len({fl.frame for fl in flags}),
        "flagged_points": len(flags),
        "examples": [{"time_s": round(fl.time_s, 3), "landmark": LANDMARK_NAMES[fl.landmark],
                      "kind": fl.kind} for fl in flags[:limit]],
    }


def describe(flags):
    """Flags for one frame as CSV text, e.g. "left_big_toe:foot_length;right_knee:spike"."""
    return ";".join(f"{LANDMARK_NAMES[idx]}:{kind}" for idx, kind in sorted(flags.items()))


def names(indices):
    """Landmark indices as CSV text, e.g. "left_heel;left_small_toe"."""
    return ";".join(LANDMARK_NAMES[i] for i in sorted(indices))


def parse_names(text):
    """The inverse of names: a set of landmark indices."""
    index = {name: idx for idx, name in LANDMARK_NAMES.items()}
    return {index[n] for n in str(text).split(";") if n in index}


def parse(text):
    """The inverse of describe: {landmark index: kind} from CSV text."""
    index = {name: idx for idx, name in LANDMARK_NAMES.items()}
    out = {}
    for item in str(text).split(";"):
        name, _, kind = item.partition(":")
        if name in index:
            out[index[name]] = kind
    return out


def _points(pose, w, h):
    """{landmark index: np.array((x, y)) in pixels} for the checked landmarks, or None."""
    if pose is None:
        return None
    return {i: np.array((pose[i].x * w, pose[i].y * h)) for i in LANDMARK_NAMES
            if pose[i] is not None}


# (hip, knee, ankle) per side, for leg lengths.
_LEG_CHAINS = ((23, 25, 27), (24, 26, 28))


def _leg_length(p):
    """The longer leg in pixels (hip to knee plus knee to ankle), or NaN."""
    lengths = [np.linalg.norm(p[hip] - p[knee]) + np.linalg.norm(p[knee] - p[ankle])
               for hip, knee, ankle in _LEG_CHAINS if hip in p and knee in p and ankle in p]
    return max(lengths) if lengths else np.nan


def _typical_leg_lengths(frames, pts):
    """Each frame's median leg length over LEG_WINDOW_S around it (NaN if never seen)."""
    times = np.array([f.time_s for f in frames])
    raw = np.array([_leg_length(p) if p is not None else np.nan for p in pts])
    out = np.full(len(frames), np.nan)
    for k in range(len(frames)):
        window = raw[np.abs(times - times[k]) <= LEG_WINDOW_S / 2]
        window = window[np.isfinite(window)]
        if len(window):
            out[k] = np.median(window)
    return out


def _foot_length_flags(p, leg):
    if not np.isfinite(leg) or leg < 1.0:
        return
    for _, ankle, feet in _LEGS:
        if ankle not in p:
            continue
        for idx in feet:
            if idx in p and np.linalg.norm(p[idx] - p[ankle]) > FOOT_TO_LEG_MAX * leg:
                yield idx


def _height(p):
    """Rough body height in pixels from the checked points (hips to feet, doubled)."""
    ys = [pt[1] for pt in p.values()]
    return 2.0 * (max(ys) - min(ys)) if len(ys) > 1 else 0.0


def _spike_flags(prev, cur, nxt):
    if prev is None or nxt is None:
        return
    height = max(_height(prev), _height(nxt))
    if height <= 0:
        return
    limit = SPIKE_HEIGHTS * height
    for idx, pt in cur.items():
        if idx not in prev or idx not in nxt:
            continue
        if (np.linalg.norm(pt - prev[idx]) > limit and np.linalg.norm(pt - nxt[idx]) > limit
                and np.linalg.norm(prev[idx] - nxt[idx]) < limit / 2):
            yield idx
