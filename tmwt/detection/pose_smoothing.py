"""
Replacing flagged pose points (pose_check.py) by interpolation, at the
reviewer's request.

Each flagged point is rebuilt from the nearest frames before and after it
where the same point wasn't flagged (at most MAX_GAP_S away), interpolating
linearly in time:

  - Heels and toes are interpolated relative to their ankle (the foot's shape
    around the ankle), then placed on this frame's ankle, so the foot moves
    with the step instead of sliding in a straight line between the two good
    frames. If only one side has a good frame, its shape is held.
  - Other points (hips, knees, ankles) are interpolated in position; that
    needs a good frame on both sides.

Points that can't be rebuilt (no good frames near enough, or no usable ankle)
are left as they were, still flagged.

Every change is recorded as an Edit with the original and new landmark, so it
can be undone, saved with the review progress, and written out next to the
CSV (data_export.write_pose_corrections).
"""

from collections import namedtuple

from tmwt.pose import pose_common as pc

# Furthest a good frame used for interpolation may be from the flagged one.
MAX_GAP_S = 0.5
METHOD = ("linear interpolation from the nearest unflagged frames (within 0.5 s); "
          "heels and toes relative to their ankle")

# Foot points -> their ankle.
_ANKLE_OF = {29: 27, 31: 27, 33: 27, 30: 28, 32: 28, 34: 28}

# One replaced point: frame (index into job.frames), landmark index, the check
# that flagged it, and the landmark before and after.
Edit = namedtuple("Edit", "frame landmark kind original smoothed")


def smooth_flagged(job):
    """
    Replace the job's flagged points (those not smoothed already) and record
    them in job.pose_edits. Returns (edits made, number that couldn't be).
    """
    frames = job.frames
    done = {(e.frame, e.landmark) for e in job.pose_edits}
    todo = [(k, idx, kind) for k, f in enumerate(frames) for idx, kind in sorted(f.pose_flags.items())
            if (k, idx) not in done]
    # Work everything out from the original points first, so one replacement
    # never feeds another.
    replacements = []
    failed = 0
    for k, idx, kind in todo:
        new = _rebuild(frames, k, idx)
        if new is None:
            failed += 1
        else:
            replacements.append(Edit(k, idx, kind, frames[k].pose[idx], new))
    for edit in replacements:
        apply(job, edit)
    job.pose_edits.extend(replacements)
    return replacements, failed


def apply(job, edit):
    """Put an edit's smoothed point into the pose and mark it smoothed."""
    f = job.frames[edit.frame]
    f.pose[edit.landmark] = edit.smoothed
    f.pose_smoothed.add(edit.landmark)


def undo(job, edits):
    """Put back the originals of `edits` (made by smooth_flagged) and forget them."""
    for edit in reversed(edits):
        f = job.frames[edit.frame]
        f.pose[edit.landmark] = edit.original
        f.pose_smoothed.discard(edit.landmark)
    gone = {id(e) for e in edits}
    job.pose_edits = [e for e in job.pose_edits if id(e) not in gone]


def undo_all(job):
    """Put back every replaced point's original."""
    undo(job, list(job.pose_edits))


def _usable(f, idx):
    """The point, if this frame has it and it isn't flagged (or was already fixed)."""
    if f.pose is None or f.pose[idx] is None:
        return None
    if idx in f.pose_flags and idx not in f.pose_smoothed:
        return None
    return f.pose[idx]


def _neighbours(frames, k, idx, need):
    """
    The nearest usable frames before and after k (within MAX_GAP_S) at which
    every landmark in `need` is usable; None for a side without one.
    """
    t = frames[k].time_s
    before = after = None
    for j in range(k - 1, -1, -1):
        if t - frames[j].time_s > MAX_GAP_S:
            break
        if all(_usable(frames[j], i) is not None for i in need):
            before = j
            break
    for j in range(k + 1, len(frames)):
        if frames[j].time_s - t > MAX_GAP_S:
            break
        if all(_usable(frames[j], i) is not None for i in need):
            after = j
            break
    return before, after


def _lerp(a, b, w):
    return a + (b - a) * w


def _rebuild(frames, k, idx):
    """A replacement landmark for point idx in frame k, or None if it can't be rebuilt."""
    f = frames[k]
    original = f.pose[idx]
    vis = getattr(original, "visibility", 1.0)
    ankle = _ANKLE_OF.get(idx)
    if ankle is not None and _usable(f, ankle) is not None:
        before, after = _neighbours(frames, k, idx, (idx, ankle))
        offsets = []
        for j in (before, after):
            if j is not None:
                p, a = frames[j].pose[idx], frames[j].pose[ankle]
                offsets.append((j, p.x - a.x, p.y - a.y, (p.z or 0.0) - (a.z or 0.0)))
        if not offsets:
            return None
        if len(offsets) == 1:
            _, dx, dy, dz = offsets[0]
        else:
            (j0, *o0), (j1, *o1) = offsets
            w = (f.time_s - frames[j0].time_s) / (frames[j1].time_s - frames[j0].time_s)
            dx, dy, dz = (_lerp(a, b, w) for a, b in zip(o0, o1))
        a = f.pose[ankle]
        return pc.Landmark(a.x + dx, a.y + dy, (a.z or 0.0) + dz, vis)

    before, after = _neighbours(frames, k, idx, (idx,))
    if before is None or after is None:
        return None
    p0, p1 = frames[before].pose[idx], frames[after].pose[idx]
    w = (f.time_s - frames[before].time_s) / (frames[after].time_s - frames[before].time_s)
    return pc.Landmark(_lerp(p0.x, p1.x, w), _lerp(p0.y, p1.y, w),
                       _lerp(p0.z or 0.0, p1.z or 0.0, w), vis)
