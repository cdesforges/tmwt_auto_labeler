"""
Review progress: the state of an unfinished review, saved as it goes so it can
be continued later.

Saved to <videos folder>/tmwt_analysis/review_progress.json after every review
decision, whenever the user switches video, and when the review stops (quit,
exit without saving, or the window closed). It's deleted once the outputs have
been saved. On the next review of the folder the user is asked whether to
continue from it or start over.

For each video it keeps what the review can change: the decision, the
endpoints, the chosen person and the timing (including manual timing, which
can't be recomputed). Each entry also records the video's fingerprint and when
its analysis file was made, so it's ignored if the video was replaced or
re-processed since.
"""

import json
import os
from datetime import datetime

from tmwt.core import analysis_file
from tmwt.detection import people, pose_smoothing
from tmwt.pose import pose_common
from tmwt.measurement import timing
from tmwt.core.job import REVIEW_UNREVIEWED, STATUS_FAILED

FORMAT_VERSION = 1
FILE_NAME = "review_progress.json"


def progress_path(jobs):
    """The progress file for the folder the jobs' videos are in."""
    return os.path.join(os.path.dirname(analysis_file.analysis_path(jobs[0].path)), FILE_NAME)


def _point(p):
    return [int(p[0]), int(p[1])] if p is not None else None


def save(jobs, last_index):
    """Write the review state of every job; `last_index` is the video worked on last."""
    videos = {}
    for job in jobs:
        if job.status == STATUS_FAILED:
            continue
        videos[job.name] = {
            "fingerprint": job.analysis_meta.get("video_fingerprint"),
            "analysis_created": job.analysis_meta.get("created"),
            "review": job.review,
            "review_note": job.review_note,
            "subject": job.subject,
            "far_ep": _point(job.far_ep),
            "near_ep": _point(job.near_ep),
            "far_ep_is_standing_spot": job.far_ep_is_standing_spot,
            "endpoint_source": job.endpoint_source,
            "endpoint_behavior": job.endpoint_behavior,
            "walk_start": job.walk_start,
            "walk_end": job.walk_end,
            "timing_source": job.timing_source,
            "timing_detail": job.timing_detail,
            "timing_note": job.timing_note,
            "pose_confirmed": job.pose_confirmed,
            # Smoothed points: [frame, landmark, x, y, z] (the originals come
            # back from the analysis file).
            "pose_edits": [[e.frame, e.landmark, e.smoothed.x, e.smoothed.y, e.smoothed.z]
                           for e in job.pose_edits],
        }
    data = {"format_version": FORMAT_VERSION,
            "saved": datetime.now().isoformat(timespec="minutes"),
            "last": jobs[last_index].name if last_index is not None else None,
            "videos": videos}
    path = progress_path(jobs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)   # never leave a half-written file


def load(jobs):
    """
    The saved progress that still applies to these jobs, or None if there's none
    (no file, another format, or no video decided yet). Entries for videos that
    were replaced or re-processed since are dropped.
    """
    path = progress_path(jobs)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("format_version") != FORMAT_VERSION:
        return None
    current = {job.name: job for job in jobs if job.status != STATUS_FAILED}
    data["videos"] = {
        name: entry for name, entry in data.get("videos", {}).items()
        if name in current
        and entry.get("fingerprint") == current[name].analysis_meta.get("video_fingerprint")
        and entry.get("analysis_created") == current[name].analysis_meta.get("created")}
    if not any(e["review"] != REVIEW_UNREVIEWED for e in data["videos"].values()):
        return None
    return data


def counts(progress):
    """(approved, skipped, still to review) in saved progress."""
    reviews = [e["review"] for e in progress["videos"].values()]
    return (reviews.count("approved"), reviews.count("rejected"),
            reviews.count(REVIEW_UNREVIEWED))


def _restore_pose_edits(job, saved):
    """Re-apply smoothed points saved with the progress to the freshly loaded poses."""
    for k, idx, x, y, z in saved:
        f = job.frames[k] if 0 <= k < len(job.frames) else None
        if f is None or f.pose is None or f.pose[idx] is None:
            continue
        original = f.pose[idx]
        new = pose_common.Landmark(x, y, z, getattr(original, "visibility", 1.0))
        edit = pose_smoothing.Edit(k, idx, f.pose_flags.get(idx, ""), original, new)
        pose_smoothing.apply(job, edit)
        job.pose_edits.append(edit)


def restore(jobs, progress):
    """
    Apply saved progress to freshly loaded jobs. Returns the index of the video
    worked on last (or None).
    """
    last = None
    for i, job in enumerate(jobs):
        entry = progress["videos"].get(job.name)
        if entry is None:
            continue
        if job.name == progress.get("last"):
            last = i
        if entry["subject"] is not None and entry["subject"] != job.subject:
            track = next((t for t in job.tracks if t.id == entry["subject"]), None)
            if track is not None:
                people.set_subject(job, track)
                job.subject_start = people.subject_start(job)
        job.far_ep = tuple(entry["far_ep"]) if entry["far_ep"] else None
        job.near_ep = tuple(entry["near_ep"]) if entry["near_ep"] else None
        job.far_ep_is_standing_spot = entry["far_ep_is_standing_spot"]
        job.endpoint_source = entry["endpoint_source"]
        job.endpoint_behavior = entry["endpoint_behavior"]
        if job.far_ep is not None and job.near_ep is not None:
            timing.apply_endpoints(job)   # per-frame positions along the new line
        job.walk_start, job.walk_end = entry["walk_start"], entry["walk_end"]
        job.timing_source, job.timing_detail = entry["timing_source"], entry["timing_detail"]
        job.timing_note = entry.get("timing_note", "")
        job.pose_confirmed = entry.get("pose_confirmed", False)
        _restore_pose_edits(job, entry.get("pose_edits") or [])
        job.review, job.review_note = entry["review"], entry["review_note"]
    return last


def clear(jobs):
    """Delete the saved progress (after saving the outputs, or on Start over)."""
    path = progress_path(jobs)
    if os.path.exists(path):
        os.remove(path)
