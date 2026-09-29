"""
Hindsight walk-start detection.

The start is found AFTER the whole walk has been tracked, in two steps:

  1. Confirm the walk: find the sustained forward advance of the ankles that
     leads into the end of the walk, and trace it back to where it began.
     Fidgets, heel raises and weight shifts never produce a sustained advance,
     so they can't be mistaken for the walk.
  2. Find the onset: take the standstill just before that advance as each
     foot's baseline (median) and noise level (MAD). The start is the first
     frame a foot leaves its standstill band and then stays clearly forward.
     If the recording starts so soon before the walk that there's hardly any
     standstill, whatever there is gets used and the result is flagged
     (info["short_standstill"]): the noise estimate is then rough, so the
     start tends to come out a little late and should be checked.

Everything here works on 1-D "distance along the course" signals in metres
(see track_distances), so it is independent of how those distances were
measured — perspective-corrected (metric.metric_along) or not.
"""

import numpy as np

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
# A standstill shorter than this is flagged as short (see find_walk_onset).
SHORT_STILL_S = 0.5
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

# Scale factor turning a median absolute deviation into a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826


class OnsetFailed(Exception):
    """Internal: carries the reason detection gave up."""


# Failure reason when there's too little standstill before the walk to go on.
START_TOO_SOON = "the walk starts too soon after the recording begins"


def track_distances(samples, dist_fn):
    """
    Turn tracked frames into the distance signals find_walk_onset works on.

    Args:
        samples: objects with .time_s and reference-frame pixel points
            .ref_foot, .ref_left_ankle, .ref_right_ankle (each (x, y) or None).
        dist_fn: maps an (x, y) pixel point to metres along the course, or None.

    Returns:
        (times, left, right, mid) numpy arrays, NaN where missing. `mid` is the
        mean of the two ankle distances, falling back to the tracked foot point
        when only one ankle is available. The foot point is rounded to whole
        pixels, which at the far end of the course is ~0.3 m per pixel, so the
        unrounded ankles are preferred.
    """
    def dist(pt):
        d = dist_fn(pt) if pt is not None else None
        return np.nan if d is None else d

    times = np.array([s.time_s for s in samples], dtype=np.float64)
    left = np.array([dist(s.ref_left_ankle) for s in samples], dtype=np.float64)
    right = np.array([dist(s.ref_right_ankle) for s in samples], dtype=np.float64)
    foot = np.array([dist(s.ref_foot) for s in samples], dtype=np.float64)
    both = np.isfinite(left) & np.isfinite(right)
    mid = np.where(both, (left + right) / 2.0, foot)
    return times, left, right, mid


def find_walk_onset(times, left, right, mid, end_time=None):
    """
    Find the walk start in hindsight.

    Args:
        times: sample timestamps in seconds (increasing).
        left, right: per-sample left / right ankle distance along the course (m);
            NaN where the ankle is missing.
        mid: per-sample ankle-midpoint distance (m); NaN if missing. Only drives
            the walk confirmation, never the onset timing.
        end_time: time the walk ended, if known. The confirmed walk is the one
            leading into it; otherwise the furthest point reached is used.

    Returns:
        (start_time, info). start_time is None on failure, and info["reason"]
        says why. On success info has the confirmation time and distance
        ("walk_confirmed_at", "walk_covered_m"), per-foot "<side>_baseline_m",
        "<side>_sigma_m" and "<side>_threshold_m", every foot's "onsets", and
        the "foot" that moved first. "standstill_s" is how much standstill the
        baseline came from, and "short_standstill" is True if that was less
        than SHORT_STILL_S or had to run into the walk's first moments (the
        start is less certain).
    """
    times = np.asarray(times, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    mid = np.asarray(mid, dtype=np.float64)
    info = {}
    try:
        i_end, t_run = _confirm_walk(times, mid, end_time, info)
        still_mask, overlaps_walk = _standstill_mask(times, left, right, t_run)
        still_times = times[still_mask]
        info["standstill_s"] = float(still_times[-1] - still_times[0]) if len(still_times) else 0.0
        info["short_standstill"] = overlaps_walk or info["standstill_s"] < SHORT_STILL_S
        if min(np.isfinite(left[still_mask]).sum(), np.isfinite(right[still_mask]).sum()) < MIN_STILL_SAMPLES:
            raise OnsetFailed(START_TOO_SOON)
        onsets = {}
        for side, d in (("left", left), ("right", right)):
            onset = _foot_onset(times, d, still_mask, i_end, side, info)
            if onset is not None:
                onsets[side] = onset
        if not onsets:
            raise OnsetFailed("neither foot left its standstill band")
    except OnsetFailed as e:
        info["reason"] = str(e)
        return None, info

    side = min(onsets, key=onsets.get)
    info["foot"] = side
    info["onsets"] = onsets
    return onsets[side], info


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


def _confirm_walk(times, mid, end_time, info):
    """
    Step 1: find the sustained advance leading into the end of the walk.

    Smooths the midpoint signal, marks samples moving forward at walking speed
    (over a centred window), and walks back from the end through that moving
    run — bridging short pauses — to where it began.

    Returns:
        (i_end, t_run): index of the walk's end sample, and the time the
        sustained advance began. Records "walk_confirmed_at" / "walk_covered_m"
        in info.
    """
    if np.isfinite(mid).sum() < MIN_STILL_SAMPLES * 2:
        raise OnsetFailed("too few tracked frames")
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    if dt <= 0:
        raise OnsetFailed("invalid timestamps")

    n_smooth = max(1, int(round(SMOOTH_S / dt)) | 1)
    smooth = _rolling_nanmedian(mid, n_smooth)
    ok = np.isfinite(smooth)
    t_ok, s_ok = times[ok], smooth[ok]

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
        raise OnsetFailed("no sustained forward movement found")
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
        raise OnsetFailed(f"longest sustained advance only {covered:.2f} m")
    return i_end, t_run


def _standstill_mask(times, left, right, t_run):
    """
    Samples forming the standstill before the walk: STILL_WINDOW_S ending where
    the walk could first have begun (the speed window is centred, so movement
    can start up to MOVE_HALF_WINDOW_S before t_run). If the walk began too soon
    after the video did to fill that window, everything before that point is
    used; if even that is too little, everything before t_run (which may take
    in the very start of the movement, making the noise estimate larger and
    the start a little late, but is better than no start at all).

    Returns:
        (mask, overlaps_walk): overlaps_walk is True for that last resort.
    """
    def enough(mask):
        return min(np.isfinite(left[mask]).sum(), np.isfinite(right[mask]).sum()) >= MIN_STILL_SAMPLES

    still_end = t_run - MOVE_HALF_WINDOW_S
    mask = (times >= still_end - STILL_WINDOW_S) & (times <= still_end)
    if not enough(mask):
        mask = times <= still_end
    if not enough(mask):
        return times <= t_run, True
    return mask, False


def _foot_onset(times, d, still_mask, i_end, side, info):
    """
    Step 2 for one foot: when it first left its standstill.

    Finds the first sample, from the start of the standstill window up to the
    walk's end, that is past the threshold and stays past it for PERSIST_S,
    then steps back (at most MAX_LEAD_IN_S) through samples outside the
    standstill band to where the movement began. The band is two-sided because
    a step often starts with a small backward dip of the ankle keypoint.

    Returns:
        The onset time, or None if this foot never clearly left its standstill.
        Records "<side>_baseline_m", "<side>_sigma_m", "<side>_threshold_m".
    """
    still = d[still_mask & np.isfinite(d)]
    if len(still) < MIN_STILL_SAMPLES:
        return None
    base = float(np.median(still))
    sigma = _MAD_TO_SIGMA * float(np.median(np.abs(still - base)))
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
        ahead = idx[pos:]
        ahead = ahead[times[ahead] <= times[j] + PERSIST_S]
        if len(ahead) < 2 or np.any(d[ahead] - base < thr):
            continue
        first_out = j
        for q in idx[:pos][::-1]:
            if abs(d[q] - base) <= band or times[j] - times[q] > MAX_LEAD_IN_S:
                break
            first_out = q
        return float(times[first_out])
    return None
