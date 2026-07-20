"""
Perspective-correct metric positions along the walking line.

`t_along` is linear in IMAGE space, but perspective means equal image steps are
not equal metric steps. On a 10 m course viewed down its length, one unit of
t_along near the far end is worth ~8x more real distance than the same unit near
the camera. Converting properly requires the vanishing point of the walking
direction.

The ArUco marker cannot supply it. Extrapolating a 175 mm square out to 10 m is
numerically hopeless: 1 px of corner-detection noise produces tens of metres of
error at the far end, because the far end sits within ~80 px of the horizon line
where the homography denominator goes to zero.

Instead we recover the vanishing point from the subject's own walk. A person of
constant height walking a straight line traces two parallel 3D lines — one
through their feet, one through their head — and parallel 3D lines share a
vanishing point in the image. Fitting both lines across the whole track uses the
full 10 m as the baseline, which measured ~100x better conditioned than the
marker on real footage.

Because this needs the completed track, it is applied as a post-pass refinement
rather than live during playback.
"""

import numpy as np

# Minimum tracked frames before a vanishing-point fit is trustworthy.
MIN_TRACK_POINTS = 12

# Reject a fit whose foot/head lines are too close to parallel in the image —
# their intersection is then numerically meaningless.
MIN_LINE_SEPARATION = 1e-6


def fit_line(points):
    """
    Total-least-squares line through 2D points.

    Returns:
        (line, straightness) where line is homogeneous [a, b, c] with
        a*x + b*y + c = 0, and straightness is the residual ratio
        (0.0 = perfectly collinear). Returns (None, None) if degenerate.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return None, None
    mean = pts.mean(axis=0)
    centered = pts - mean
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    if s[0] < 1e-9:
        return None, None
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    c = -normal @ mean
    straightness = s[1] / s[0] if len(s) > 1 else 0.0
    return np.array([normal[0], normal[1], c]), straightness


def _intersect(line_a, line_b):
    """Intersection of two homogeneous lines, or None if parallel."""
    p = np.cross(line_a, line_b)
    if abs(p[2]) < MIN_LINE_SEPARATION:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]])


def estimate_vanishing_point(foot_points, head_points):
    """
    Recover the walking direction's vanishing point from the subject's track.

    Args:
        foot_points: list of (x, y) ankle-midpoint pixel positions.
        head_points: list of (x, y) head (nose) pixel positions.

    Returns:
        (V, info) where V is the (x, y) vanishing point or None, and info is a
        dict with diagnostic fields for logging.
    """
    info = {"n_points": min(len(foot_points), len(head_points))}
    if info["n_points"] < MIN_TRACK_POINTS:
        info["reason"] = f"only {info['n_points']} tracked frames"
        return None, info

    n = info["n_points"]
    foot_line, foot_straight = fit_line(foot_points[:n])
    head_line, head_straight = fit_line(head_points[:n])
    if foot_line is None or head_line is None:
        info["reason"] = "degenerate line fit"
        return None, info

    info["foot_straightness"] = foot_straight
    info["head_straightness"] = head_straight

    V = _intersect(foot_line, head_line)
    if V is None:
        info["reason"] = "foot and head lines are parallel in image"
        return None, info

    info["V"] = V
    return V, info


def metric_along(point, far_ep, near_ep, V, course_m):
    """
    Real distance from `far_ep` to `point`, measured along the far->near line.

    Uses the projective cross-ratio, which is exact under perspective:

        d(P) = L * (|P-A| * |B-V|) / (|B-A| * |P-V|)

    with A = far_ep (0 m), B = near_ep (course_m), V = vanishing point.

    Returns:
        Distance in metres, or None if the point is degenerate (at V).
    """
    P = np.asarray(point, dtype=np.float64)
    A = np.asarray(far_ep, dtype=np.float64)
    B = np.asarray(near_ep, dtype=np.float64)
    Vv = np.asarray(V, dtype=np.float64)

    pv = np.linalg.norm(P - Vv)
    ba = np.linalg.norm(B - A)
    if pv < 1e-9 or ba < 1e-9:
        return None
    value = course_m * (np.linalg.norm(P - A) * np.linalg.norm(B - Vv)) / (ba * pv)
    return float(value) if np.isfinite(value) else None


def refine_start_time(track, far_ep, near_ep, V, threshold_m, course_m):
    """
    Recompute the walk start using a TRUE metric motion threshold.

    Mirrors the live per-foot baseline logic, but compares displacement in real
    metres instead of image-space t_along units. Each foot keeps its own
    standstill baseline; the start fires when either foot has moved
    `threshold_m` past its own baseline, and the reported time is that foot's
    baseline timestamp.

    Args:
        track: list of dicts with keys "time_s", "left_ankle", "right_ankle";
               ankle values are (x, y) pixel tuples or None.
        far_ep, near_ep: rope endpoints in pixels (current-frame coords).
        V: vanishing point from estimate_vanishing_point.
        threshold_m: required forward displacement in metres.
        course_m: real length of the course in metres.

    Returns:
        (start_time, which_foot) or (None, None) if the threshold is never met.
    """
    baselines = {"left": None, "right": None}
    baseline_times = {"left": None, "right": None}

    for sample in track:
        time_s = sample.get("time_s")
        if time_s is None:
            continue
        for side, key in (("left", "left_ankle"), ("right", "right_ankle")):
            pt = sample.get(key)
            if pt is None:
                continue
            d = metric_along(pt, far_ep, near_ep, V, course_m)
            if d is None:
                continue
            base = baselines[side]
            if base is None or d < base:
                baselines[side] = d
                baseline_times[side] = time_s
            elif d - base >= threshold_m:
                return baseline_times[side], side

    return None, None
