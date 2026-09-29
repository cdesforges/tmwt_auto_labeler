"""
Frame annotation shared by the labeler (review playback and saved videos) and
view.py: the scene overlay (skeleton, body point, rope) and the info side panel
with the walk status and timer.
"""

import cv2
import numpy as np

from tmwt.pose import pose_common
from tmwt.core.job import COURSE_M
from tmwt.detection import pose_check
from tmwt.ui.widgets import truncate

# Width (px) of the info panel placed to the right of each frame.
PANEL_W = 300

FONT = cv2.FONT_HERSHEY_SIMPLEX

FAR_COLOR = (255, 0, 0)     # blue: far endpoint (start)
NEAR_COLOR = (0, 0, 255)    # red: near endpoint (finish)
ROPE_COLOR = (0, 255, 255)  # yellow: the course line
BODY_COLOR = (0, 0, 255)    # red: ankle midpoint

_WHITE = (255, 255, 255)
_LABEL = (150, 150, 150)
_INFO = (120, 120, 120)
_RULE = (80, 80, 80)
_GREEN = (0, 200, 0)
_BRIGHT_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_ORANGE = pose_common.FLAGGED_COLOR
_SMOOTHED = pose_common.SMOOTHED_COLOR

# Panel text shown before the walk starts.
WAITING_AUTO = ("Waiting for person", "to start walking...")


def draw_scene(img, pose, body_px, far_ep, near_ep, flags=(), smoothed=()):
    """
    Draw the skeleton, body point and rope on `img` (in place). Any may be
    None. Landmarks in `flags` (implausible points, see pose_check.py) are
    drawn in orange, and those in `smoothed` (replaced, pose_smoothing.py) in
    yellow.
    """
    if pose is not None:
        pose_common.draw_pose(img, pose, highlight=flags, smoothed=smoothed)
        if body_px is not None:
            cv2.circle(img, body_px, 5, BODY_COLOR, -1)
    if far_ep is not None and near_ep is not None:
        cv2.circle(img, far_ep, 7, FAR_COLOR, -1)
        cv2.circle(img, near_ep, 7, NEAR_COLOR, -1)
        cv2.line(img, far_ep, near_ep, ROPE_COLOR, 2)


def draw_info_panel(height, time_s, frame_idx, t_along, walk_start, walk_end,
                    title="TMWT Labeler", subtitle=None,
                    waiting_lines=WAITING_AUTO, controls="", model_strength=None, flagged=(),
                    smoothed=()):
    """
    Build the info side panel for one frame.

    The walk state is worked out from `time_s`: WAITING before walk_start,
    WALKING (with elapsed time) until walk_end, then FINISHED (with the walk
    time and speed). walk_start / walk_end may be None.

    Returns:
        A (height, PANEL_W, 3) image.
    """
    panel = np.zeros((height, PANEL_W, 3), dtype=np.uint8)
    x0 = 15
    y = 40

    def text(s, pos_y, scale, color, thickness=1, x=x0):
        # Shortened with "..." rather than drawn past the panel's edge.
        s = truncate(s, PANEL_W - x - 4, scale, thickness)
        cv2.putText(panel, s, (x, pos_y), FONT, scale, color, thickness)

    def rule(pos_y):
        cv2.line(panel, (x0, pos_y), (PANEL_W - x0, pos_y), _RULE, 1)

    text(title, y, 0.7, _WHITE, 2)
    if subtitle:
        y += 20
        text(subtitle, y, 0.4, _INFO)
        y += 5
    else:
        y += 15
    rule(y)
    y += 30

    started = walk_start is not None and time_s >= walk_start
    finished = started and walk_end is not None and time_s >= walk_end
    if finished:
        status, status_color = "FINISHED", _GREEN
    elif started:
        status, status_color = "WALKING", _YELLOW
    else:
        status, status_color = "WAITING", _LABEL
    text("Status:", y, 0.5, _LABEL)
    text(status, y, 0.7, status_color, 2, x=x0 + 80)
    y += 40

    if finished:
        duration = walk_end - walk_start
        text("Walk Time", y, 0.5, _LABEL)
        y += 35
        text(f"{duration:.3f}s", y, 1.2, _BRIGHT_GREEN, 2)
        y += 30
        if duration > 0:
            text(f"{COURSE_M / duration:.2f} m/s", y, 0.7, _GREEN, 2)
    elif started:
        text("Elapsed", y, 0.5, _LABEL)
        y += 35
        text(f"{time_s - walk_start:.2f}s", y, 1.2, _YELLOW, 2)
    else:
        text(waiting_lines[0], y, 0.5, _LABEL)
        y += 25
        text(waiting_lines[1], y, 0.5, _LABEL)

    y += 50
    rule(y)
    y += 25
    text(f"Frame: {frame_idx}", y, 0.45, _INFO)
    y += 22
    text(f"Time:  {time_s:.2f}s", y, 0.45, _INFO)
    y += 22
    if t_along is not None:
        text(f"t_along:  {t_along:+.3f}", y, 0.45, _INFO)
        y += 22
    if model_strength:
        text(f"Model strength:  {model_strength}", y, 0.45, _INFO)
        y += 22
    # The flag lists stop above the controls line at the bottom, ending with
    # "+N more" when they don't all fit.
    bottom = height - 30
    for names, heading, color in ((flagged, "Pose check (orange):", _ORANGE),
                                  (smoothed, "Smoothed (yellow):", _SMOOTHED)):
        if not names or y + 8 + 20 > bottom:
            continue
        y += 8
        text(heading, y, 0.45, color)
        for n, name in enumerate(names):
            left = len(names) - n
            if y + 40 > bottom and left > 1:     # room for one more line: say how many are left
                y += 20
                text(f"  +{left} more", y, 0.42, color)
                break
            y += 20
            text(f"  {name.replace('_', ' ')}", y, 0.42, color)
        y += 22

    text(controls, height - 15, 0.4, _RULE)
    return panel


def render_frame(frame_bgr, result, walk_start, walk_end,
                 waiting_lines=WAITING_AUTO, controls="", with_skeleton=False, model_strength=None):
    """
    Annotate one analysed frame for the labeler.

    Args:
        frame_bgr: the video frame (drawn on in place).
        result: its job.FrameResult (its pose_flags are drawn in orange).
        walk_start, walk_end: the walk timing to show (either may be None).
        with_skeleton: also build the de-identified version.
        model_strength: the pose model, listed in the panel.

    Returns:
        (annotated, skeleton): frame + info panel, and the same annotations on a
        black canvas + panel (None unless with_skeleton).
    """
    panel = draw_info_panel(frame_bgr.shape[0], result.time_s, result.frame_idx,
                            result.t_along, walk_start, walk_end,
                            waiting_lines=waiting_lines, controls=controls,
                            model_strength=model_strength,
                            flagged=[pose_check.LANDMARK_NAMES[i] for i in sorted(result.pose_flags)
                                     if i not in result.pose_smoothed],
                            smoothed=[pose_check.LANDMARK_NAMES[i] for i in sorted(result.pose_smoothed)])
    args = (result.pose, result.body_px, result.far_ep, result.near_ep, result.pose_flags,
            result.pose_smoothed)
    skeleton = None
    if with_skeleton:
        canvas = np.zeros_like(frame_bgr)
        draw_scene(canvas, *args)
        skeleton = np.hstack([canvas, panel])
    draw_scene(frame_bgr, *args)
    return np.hstack([frame_bgr, panel]), skeleton
