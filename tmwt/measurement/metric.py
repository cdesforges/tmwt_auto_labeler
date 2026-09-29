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

Because this needs the completed track, it is computed after the whole video
has been analysed. The walk-start detection that uses these distances lives in
onset.py.
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


def t_along(point, far_ep, near_ep):
    """
    Image-space position of `point` along the far_ep -> near_ep line: its
    orthogonal projection as a fraction of the line (0 at far_ep, 1 at near_ep,
    negative behind far_ep). Linear in pixels, so NOT a true distance — see
    metric_along for that.

    Returns None if the endpoints coincide.
    """
    ax, ay = far_ep
    bx, by = near_ep
    vx, vy = float(bx - ax), float(by - ay)
    vv = vx * vx + vy * vy
    if vv <= 1e-6:
        return None
    return ((point[0] - ax) * vx + (point[1] - ay) * vy) / vv


def metric_along(point, far_ep, near_ep, V, course_m):
    """
    Signed real distance from `far_ep` to `point`, measured along the far->near line.

    The point is first projected orthogonally onto the A->B line, so sideways
    movement (weight shifts, the ankles' lateral offset from the line) does not
    read as forward distance. The projection's position is then converted with
    the projective cross-ratio, which is exact under perspective:

        d(P') = L * u * |B-V| / |P'-V|,   P' = A + u (B - A)

    with A = far_ep (0 m), B = near_ep (course_m), V = vanishing point and u the
    image-space fraction along A->B (t_along). Points behind far_ep come out
    negative.

    Returns:
        Distance in metres, or None if the point is degenerate (at V).
    """
    u = t_along(point, far_ep, near_ep)
    if u is None:
        return None
    A = np.asarray(far_ep, dtype=np.float64)
    B = np.asarray(near_ep, dtype=np.float64)
    Vv = np.asarray(V, dtype=np.float64)
    P_line = A + u * (B - A)
    pv = np.linalg.norm(P_line - Vv)
    if pv < 1e-9:
        return None
    value = course_m * u * np.linalg.norm(B - Vv) / pv
    return float(value) if np.isfinite(value) else None
