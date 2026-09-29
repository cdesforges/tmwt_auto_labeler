"""
Walk timing: when the walk started and ended, decided after the whole video
has been analysed.

  - End: the first ankle to cross the near endpoint (the finish line), or with
    job.end_behavior == END_ANKLE_MIDPOINT, the midpoint of the two ankles.
  - Start: found in hindsight from the walk that leads into that end — the
    first movement of the first foot to leave its standstill (onset.py).
    If the user clicked the far endpoint as a start LINE and the subject stood
    behind it, the start is instead the moment they cross that line.

Line crossings use t_along (position along the rope in image space, 0 = far,
1 = near), smoothed with an EMA to suppress keypoint jitter.
"""

import metric
import onset
import pose_common
import tracking
from job import COURSE_M, END_ANKLE_MIDPOINT

# EMA weight on the newest t_along sample.
SMOOTH_ALPHA = 0.7
# t_along of the start line (far endpoint) and finish line (near endpoint).
FAR_T = 0.0
NEAR_T = 1.0


def smooth_t_along(values):
    """EMA-smooth a t_along series. None entries stay None and are skipped."""
    smoothed = []
    prev = None
    for t in values:
        if t is not None:
            prev = t if prev is None else SMOOTH_ALPHA * t + (1.0 - SMOOTH_ALPHA) * prev
        smoothed.append(prev if t is not None else None)
    return smoothed


def find_crossing(times, t_smooth, level, after=None):
    """
    Time at which a smoothed t_along series first crosses `level` from below,
    interpolated between samples. Samples that are None are skipped, and only
    crossings reached at or after `after` count.

    Returns:
        The crossing time in seconds, or None if it never crosses.
    """
    prev_t = prev_v = None
    for t, v in zip(times, t_smooth):
        if v is None:
            continue
        if (prev_v is not None and prev_v < level <= v
                and (after is None or t >= after)):
            frac = (level - prev_v) / (v - prev_v) if v != prev_v else 0.0
            return prev_t + frac * (t - prev_t)
        prev_t, prev_v = t, v
    return None


def apply_endpoints(job):
    """
    Project the job's rope endpoints into every frame and recompute each frame's
    far_ep / near_ep (that frame's pixels), t_along and t_smooth.
    """
    for f in job.frames:
        f.far_ep, f.near_ep = tracking.transform_points(f.H, [job.far_ep, job.near_ep])
        f.t_along = (metric.t_along(f.body_px, f.far_ep, f.near_ep)
                     if f.body_px is not None else None)
    for f, s in zip(job.frames, smooth_t_along([f.t_along for f in job.frames])):
        f.t_smooth = s


def update_timing(job):
    """Recompute the automatic walk timing from the job's current endpoints."""
    apply_endpoints(job)
    job.walk_start, job.walk_end, job.timing_detail = detect_walk_times(job)
    job.timing_source = "auto"


def detect_walk_times(job):
    """
    Decide the walk start and end (see the module docstring). Expects
    apply_endpoints to have run.

    Returns:
        (start, end, detail): times in seconds (either may be None), and a short
        description of how the start was found.
    """
    times = [f.time_s for f in job.frames]
    t_smooth = [f.t_smooth for f in job.frames]

    first_end = _find_end(job, times, t_smooth)
    start_move, info = _first_foot_movement(job, first_end)
    moved = (f"first {info['foot']} foot movement" if start_move is not None else "")

    if job.far_ep_is_standing_spot:
        start, detail = start_move, moved
    else:
        # Clicked start line: if the subject was already on or past it when they
        # started moving, the movement is the start; otherwise they must cross it.
        at_move = _value_at(times, t_smooth, start_move) if start_move is not None else None
        if at_move is not None and at_move >= FAR_T:
            start, detail = start_move, moved + " (on/past start line)"
        else:
            start = find_crossing(times, t_smooth, FAR_T, after=start_move)
            detail = "start-line crossing"
            if start is None and start_move is not None:
                start, detail = start_move, moved

    end = _find_end(job, times, t_smooth, after=start) if start is not None else None
    if start is not None:
        print(f"  Walk STARTED at {start:.3f}s ({detail})")
    if end is not None:
        print(f"  Walk FINISHED at {end:.3f}s ({job.end_behavior.replace('_', ' ')} at the finish line)")
    return start, end, detail


def _find_end(job, times, t_smooth, after=None):
    """
    When the walk ended: the first crossing of the finish line (at or after
    `after`) by the first ankle, or by the ankle midpoint (job.end_behavior).
    Falls back to the midpoint if neither ankle is ever seen crossing.
    """
    midpoint = find_crossing(times, t_smooth, NEAR_T, after=after)
    if job.end_behavior == END_ANKLE_MIDPOINT:
        return midpoint
    feet = [t for t in (_foot_crossing(job, times, idx, after)
                        for idx in (pose_common.LEFT_ANKLE_IDX, pose_common.RIGHT_ANKLE_IDX))
            if t is not None]
    return min(feet) if feet else midpoint


def _foot_crossing(job, times, ankle_idx, after):
    """First finish-line crossing (at or after `after`) of one ankle, smoothed like t_along."""
    h, w = job.info.first_frame.shape[:2]
    values = []
    for f in job.frames:
        lm = f.pose[ankle_idx] if f.pose is not None else None
        px = pose_common.landmark_px(lm, w, h)
        values.append(metric.t_along(px, f.far_ep, f.near_ep) if px is not None else None)
    return find_crossing(times, smooth_t_along(values), NEAR_T, after=after)


def _first_foot_movement(job, end_time):
    """Run the hindsight onset detection on the job's track. Returns (time, info)."""
    track = job.track
    dist_fn, units = _distance_function(job, track)
    start, info = onset.find_walk_onset(*onset.track_distances(track, dist_fn),
                                        end_time=end_time)
    if start is None:
        print(f"  No walk onset found: {info.get('reason', 'unknown')}")
        return None, info

    side = info["foot"]
    print(f"  Walk confirmed at {info['walk_confirmed_at']:.3f}s "
          f"({info['walk_covered_m']:.1f} m sustained advance, {units})")
    print(f"  First foot movement at {start:.3f}s ({side} foot; "
          f"threshold {info[side + '_threshold_m'] * 100:.0f} cm, "
          f"noise {info[side + '_sigma_m'] * 100:.1f} cm)")
    return start, info


def _distance_function(job, track):
    """
    (dist_fn, units): maps a reference-frame pixel point to metres along the
    course. Perspective-corrected when the vanishing point of the walk can be
    recovered; otherwise image-space, whose real scale varies with distance
    from the camera, so the onset thresholds are only approximate.
    """
    V, info = metric.estimate_vanishing_point([f.ref_foot for f in track],
                                              [f.ref_head for f in track])
    if V is not None:
        return (lambda p: metric.metric_along(p, job.far_ep, job.near_ep, V, COURSE_M),
                "perspective-corrected")

    print(f"  Vanishing-point fit failed ({info.get('reason', 'unknown')}); "
          f"using image-space distances for start detection.")

    def image_space(p):
        t = metric.t_along(p, job.far_ep, job.near_ep)
        return None if t is None else t * COURSE_M
    return image_space, "image-space"


def _value_at(times, values, time_s):
    """The last non-None value at or before `time_s`, or None."""
    result = None
    for t, v in zip(times, values):
        if t > time_s:
            break
        if v is not None:
            result = v
    return result
