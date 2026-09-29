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
    P = np.asarray(point, dtype=np.float64)
    A = np.asarray(far_ep, dtype=np.float64)
    B = np.asarray(near_ep, dtype=np.float64)
    Vv = np.asarray(V, dtype=np.float64)

    ab = B - A
    ab2 = float(ab @ ab)
    if ab2 < 1e-9:
        return None
    u = float((P - A) @ ab) / ab2
    P_line = A + u * ab
    pv = np.linalg.norm(P_line - Vv)
    if pv < 1e-9:
        return None
    value = course_m * u * np.linalg.norm(B - Vv) / pv
    return float(value) if np.isfinite(value) else None


# --- Hindsight walk-start detection ------------------------------------------
#
# The start is found AFTER the whole walk has been tracked, in two steps:
#
#   1. Confirm: find the sustained forward advance of the ankle midpoint that
#      leads into the end of the walk, and trace it back to where it began.
#      Fidgets, heel raises and weight shifts never produce a sustained advance,
#      so they cannot be mistaken for the walk.
#   2. Onset: take the standstill just before that advance as each foot's
#      baseline (median) and noise level (MAD). The start is the first frame a
#      foot leaves its standstill band and stays clearly forward of it.

# Rolling-median window applied to the ankle-midpoint signal before confirming.
SMOOTH_S = 0.4
# Half-width of the centred window used to measure forward speed.
MOVE_HALF_WINDOW_S = 0.5
# Minimum forward speed that counts as walking. Kept low for slow clinical gait.
MIN_WALK_SPEED_MPS = 0.15
# Pauses shorter than this (e.g. long double-support in slow gait) don't split
# the walk into separate bouts.
MAX_PAUSE_S = 0.6
# The confirmed walk must cover at least this much ground.
CONFIRM_M = 0.5
# Standstill window, just before the confirmed walk, used for baseline + noise.
STILL_WINDOW_S = 1.0
# Minimum samples in the standstill window for a trustworthy baseline.
MIN_STILL_SAMPLES = 5
# Onset threshold = max(MIN_ONSET_M, NOISE_K * robust sigma of the standstill).
MIN_ONSET_M = 0.05
NOISE_K = 4.0
# A foot must stay past the threshold this long to count (rejects one-frame spikes).
PERSIST_S = 0.15
# How far before its forward swing a foot's first movement can be. The step
# begins with a heel lift, which moves the ankle keypoint up (reading as a small
# backward dip) roughly 0.1-0.3 s before the foot swings forward. Capping the
# look-back keeps slow whole-body leans before the step from being included.
MAX_LEAD_IN_S = 0.3


def _rolling_nanmedian(values, n):
    """Centred rolling median of length n (odd), ignoring NaNs."""
    if n <= 1:
        return values.copy()
    half = n // 2
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        window = window[np.isfinite(window)]
        if len(window):
            out[i] = np.median(window)
    return out


def find_walk_onset(times, left, right, mid, end_time=None):
    """
    Hindsight walk-start detection on 1-D distance-along-course signals.

    Args:
        times: sample timestamps in seconds (increasing).
        left, right: per-sample left / right ankle distance along the course (m);
            NaN where the ankle is missing.
        mid: per-sample ankle-midpoint distance along the course (m); NaN if missing.
            Only drives the walk confirmation, never the onset timing.
        end_time: time the walk ended, if known. The confirmed walk is the one
            leading into it; otherwise the furthest point reached is used.

    Returns:
        (start_time, info). start_time is None on failure; info always has a
        "reason" (on failure) or diagnostic fields (on success).
    """
    times = np.asarray(times, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mid = np.asarray(mid, dtype=np.float64)
    info = {}

    valid = np.isfinite(mid)
    if valid.sum() < MIN_STILL_SAMPLES * 2:
        info["reason"] = "too few tracked frames"
        return None, info

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    if dt <= 0:
        info["reason"] = "invalid timestamps"
        return None, info
    n_smooth = max(1, int(round(SMOOTH_S / dt)) | 1)
    smooth = _rolling_nanmedian(mid, n_smooth)
    ok = np.isfinite(smooth)
    t_ok, s_ok = times[ok], smooth[ok]

    # --- Step 1: confirm the walk and trace it back to where it began ---
    if end_time is not None:
        i_end = int(np.searchsorted(times, end_time, side="right")) - 1
    else:
        i_end = int(np.nanargmax(smooth))
    i_end = max(0, min(i_end, len(times) - 1))

    h = MOVE_HALF_WINDOW_S
    disp = np.interp(times + h, t_ok, s_ok) - np.interp(times - h, t_ok, s_ok)
    moving = disp >= MIN_WALK_SPEED_MPS * 2 * h

    moving_before_end = np.nonzero(moving[:i_end + 1])[0]
    if len(moving_before_end) == 0:
        info["reason"] = "no sustained forward movement found"
        return None, info
    run_start = int(moving_before_end[-1])
    for k in range(run_start - 1, -1, -1):
        if moving[k]:
            run_start = k
        elif times[run_start] - times[k] > MAX_PAUSE_S:
            break

    t_run = times[run_start]
    covered = float(np.interp(times[i_end], t_ok, s_ok) - np.interp(t_run, t_ok, s_ok))
    info["walk_confirmed_at"] = float(t_run)
    info["walk_covered_m"] = covered
    if covered < CONFIRM_M:
        info["reason"] = f"longest sustained advance only {covered:.2f} m"
        return None, info

    # --- Step 2: per-foot baseline + noise from the standstill before the walk ---
    # The speed window is centred, so movement can begin up to h before t_run.
    still_end = t_run - h
    still_mask = (times >= still_end - STILL_WINDOW_S) & (times <= still_end)
    if min(np.isfinite(left[still_mask]).sum(), np.isfinite(right[still_mask]).sum()) < MIN_STILL_SAMPLES:
        # Walk began soon after the video did: use everything before it.
        still_mask = times <= still_end

    onsets = {}
    for side, d in (("left", left), ("right", right)):
        still = d[still_mask & np.isfinite(d)]
        if len(still) < MIN_STILL_SAMPLES:
            continue
        base = float(np.median(still))
        sigma = 1.4826 * float(np.median(np.abs(still - base)))
        thr = max(MIN_ONSET_M, NOISE_K * sigma)
        band = thr / 2.0
        info[f"{side}_baseline_m"] = base
        info[f"{side}_sigma_m"] = sigma
        info[f"{side}_threshold_m"] = thr

        search_from = int(np.argmax(still_mask))
        idx = np.nonzero(np.isfinite(d))[0]
        idx = idx[(idx >= search_from) & (idx <= i_end)]
        for pos, j in enumerate(idx):
            if d[j] - base < thr:
                continue
            # Sustained: every sample in the next PERSIST_S stays past threshold.
            ahead = idx[pos:]
            ahead = ahead[times[ahead] <= times[j] + PERSIST_S]
            if len(ahead) < 2 or np.any(d[ahead] - base < thr):
                continue
            # Step back to the first sample that left the standstill band. The
            # band is two-sided: the step often begins with a small backward dip
            # of the ankle keypoint (heel lift / weight shift) before it moves forward.
            first_out = j
            for q in idx[:pos][::-1]:
                if abs(d[q] - base) <= band or times[j] - times[q] > MAX_LEAD_IN_S:
                    break
                first_out = q
            onsets[side] = float(times[first_out])
            break

    if not onsets:
        info["reason"] = "neither foot left its standstill band"
        return None, info

    side = min(onsets, key=onsets.get)
    info["foot"] = side
    info["onsets"] = onsets
    return onsets[side], info


def track_distances(track, dist_fn):
    """
    Turn a walk track into arrays for find_walk_onset.

    Args:
        track: list of dicts with "time_s", "foot", "left_ankle", "right_ankle"
            (pixel tuples or None).
        dist_fn: maps an (x, y) pixel point to metres along the course, or None.

    Returns:
        (times, left, right, mid) numpy arrays, NaN where missing. `mid` is the
        mean of the two ankle distances, falling back to the tracked "foot"
        point when only one ankle is available. The "foot" point is rounded to
        whole pixels, which at the far end of the course is ~0.3 m per pixel,
        so the unrounded ankles are preferred.
    """
    def dist(pt):
        if pt is None:
            return np.nan
        d = dist_fn(pt)
        return np.nan if d is None else d

    samples = [s for s in track if s.get("time_s") is not None]
    times = np.array([s["time_s"] for s in samples], dtype=np.float64)
    left = np.array([dist(s.get("left_ankle")) for s in samples])
    right = np.array([dist(s.get("right_ankle")) for s in samples])
    foot = np.array([dist(s.get("foot")) for s in samples])
    both = np.isfinite(left) & np.isfinite(right)
    mid = np.where(both, (left + right) / 2.0, foot)
    return times, left, right, mid
