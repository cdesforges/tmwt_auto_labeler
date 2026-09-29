"""
Flagged-points mode (review.py): checking the pose points the pose check
(detection/pose_check.py) found implausible, before the timing.

Opening a video with flagged points shows a full-screen notice
(alert_pose_flags), then review_flagged_points: the video paused on the first
flagged frame, flagged points in orange, with buttons to jump between flagged
frames, the usual frame step and play, Smooth points / Unsmooth
(detection/pose_smoothing.py; smoothed points turn yellow) and Confirm.
"""

from tmwt.core import video_io
from tmwt.detection import pose_check, pose_smoothing
from tmwt.measurement import timing
from tmwt.session.review_playback import pose_highlights
from tmwt.ui import annotate
from tmwt.ui.player import Player
from tmwt.ui.widgets import GREEN, GREY, KEY_ENTER, ORANGE, RED

# Flagged-points mode: jump between flagged frames ([ and ]).
_PREV_FLAG = ("Previous flagged frame ([)", "prev_flag", (ord("["),), "prev_flag")

_NEXT_FLAG = ("Next flagged frame (])", "next_flag", (ord("]"),), "next_flag")


def alert_pose_flags(job, ui):
    """Full-screen notice, when a video with flagged pose points is opened, before flagged-points mode."""
    flags = job.pose_flags
    examples = sorted({(round(fl.time_s, 2), pose_check.LANDMARK_NAMES[fl.landmark].replace("_", " "))
                       for fl in flags})[:3]
    check = job.analysis_meta.get("pose_check") or {}
    retried = len(check.get("runs", [])) > 1
    lines = [
        ("Pose detection anomalies", ORANGE),
        (f"{job.pose_flagged_frames} frame(s) have leg or foot points that look implausible,", GREY),
        ("e.g. " + ", ".join(f"{name} at {t:.2f}s" for t, name in examples) + ".", GREY),
        (f"Model strength: {job.model_strength}"
         + (" (re-analysed with the heavier model; these remain)." if retried else "."), GREY),
    ]
    if job.pose_edits:
        lines.append((f"{len(job.pose_edits)} of them were smoothed earlier (yellow).", GREY))
    lines += [
        ("Next: check them frame by frame (orange), smooth them if needed, then confirm.", GREY),
    ]
    ui.show_message(lines, [("Review flagged points", "review", KEY_ENTER)],
                    background=job.info.first_frame)


def review_flagged_points(job, ui):
    """
    Flagged-points mode: show the video paused on the first flagged frame,
    with the flagged points in orange (and orange on the timeline), to check
    them before the timing. Buttons: previous / next flagged frame, the usual
    frame step and play, Smooth points (replace the flagged points by
    interpolation; they turn yellow) or Unsmooth (put them back as detected),
    and Confirm, which records that the reviewer checked them and returns.
    Automatic timing is recomputed when the points change, since smoothed
    points count in it.
    """
    flagged = sorted(k for k, f in enumerate(job.frames) if f.pose_flags)
    player = Player([f.time_s for f in job.frames])
    player.k, player.paused = flagged[0], True
    source = video_io.FrameSource(job)
    ui.start_playback()
    alert = None
    try:
        while True:
            player.highlights = pose_highlights(job)
            frame = source.get(player.k)
            if frame is None:
                break
            image, _ = annotate.render_frame(frame, job.frames[player.k], job.walk_start,
                                             job.walk_end, model_strength=job.model_strength)
            smoothed = len(job.pose_edits)
            label = (f"FLAGGED POINTS  {job.pose_flagged_frames} frame(s)"
                     + (f"  {smoothed} smoothed (yellow)" if smoothed else "  (orange)"))
            toggle = (("Unsmooth", "unsmooth", ()) if job.pose_edits
                      else ("Smooth points", "smooth", ()))
            specs = ([_PREV_FLAG] + player.transport() + [_NEXT_FLAG, toggle]
                     + [("Confirm (Enter)", "confirm", KEY_ENTER)])
            value, _ = player.show(ui, image, specs, label,
                                   marks=[(job.walk_start, GREEN), (job.walk_end, RED)], alert=alert)
            alert = None
            if value == "confirm":
                job.pose_confirmed = True
                print(f"  Flagged points confirmed ({smoothed} smoothed).")
                break
            if value in ("prev_flag", "next_flag"):
                ahead = [k for k in flagged if k > player.k] if value == "next_flag" else \
                        [k for k in flagged if k < player.k][::-1]
                player.k = ahead[0] if ahead else player.k
                player.paused = True
            elif value == "smooth":
                edits, failed = pose_smoothing.smooth_flagged(job)
                print(f"  Smoothed {len(edits)} flagged point(s); {failed} couldn't be.")
                if failed:
                    alert = (f"{failed} point(s) had no good frames within "
                             f"{pose_smoothing.MAX_GAP_S:g} s and stay orange")
                _retime(job)
            elif value == "unsmooth":
                pose_smoothing.undo_all(job)
                print("  Smoothing undone.")
                _retime(job)
            if not player.advance():
                player.paused = True   # stay on the last frame until confirmed
    finally:
        source.close()


def _retime(job):
    """Recompute automatic timing after the points it's based on changed."""
    if job.timing_source == "auto" and job.far_ep is not None and job.near_ep is not None:
        timing.update_timing(job)
