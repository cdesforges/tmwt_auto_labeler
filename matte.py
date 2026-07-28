"""
Detect and crop solid-color mattes (letterbox / pillarbox bars) around a frame.

Phone and editing software often pad video with solid black, grey, or white
bars so the active picture occupies only part of the frame. Those bars waste
pose-detector resolution and can confuse ArUco / ground tracking. This module
finds the active-picture rectangle from the first frame and provides a
VideoCapture wrapper that crops every frame to it transparently, so the rest of
the pipeline is unaffected and all coordinates stay in one consistent space.

A matte bar is, by definition, a band of near-uniform color along a frame edge.
We seed each side's expected color from its outermost line and scan inward while
lines stay both low-variance and close to that seed color, stopping at the first
line that carries real picture content.
"""

import cv2
import numpy as np

# A line (row/column) counts as matte only if its per-pixel spread is below this.
# Measured matte bars sit near 0-1; real content rows are 20+.
DEFAULT_STD_THRESH = 6.0

# ...and its mean color must stay within this L1 distance of the seed color,
# so a coincidentally-flat content band of a different shade isn't eaten.
DEFAULT_COLOR_TOL = 14.0

# Never trim more than this fraction from any single side — a guard against
# pathological frames where most of the picture happens to be low-variance.
DEFAULT_MAX_CROP_FRAC = 0.45


def _scan_edge(lines, std_thresh, color_tol, max_count):
    """
    Count consecutive matte lines from index 0 inward.

    Args:
        lines: array shaped (N, pixels_per_line, channels) ordered from the edge
               inward (so lines[0] is the outermost row/column).
        std_thresh, color_tol: matte thresholds.
        max_count: hard cap on how many lines may be consumed.

    Returns:
        Number of leading lines that are part of the matte (0 if the edge line
        already carries content).
    """
    if len(lines) == 0:
        return 0
    seed = lines[0].reshape(-1, lines.shape[-1]).mean(axis=0)
    count = 0
    for i in range(min(len(lines), max_count)):
        flat = lines[i].reshape(-1, lines.shape[-1]).astype(np.float32)
        if flat.std() >= std_thresh:
            break
        if np.abs(flat.mean(axis=0) - seed).sum() > color_tol:
            break
        count += 1
    return count


def detect_content_crop(frame_bgr,
                        std_thresh=DEFAULT_STD_THRESH,
                        color_tol=DEFAULT_COLOR_TOL,
                        max_crop_frac=DEFAULT_MAX_CROP_FRAC):
    """
    Find the active-picture rectangle inside a matted frame.

    Returns:
        (x, y, w, h) of the content region. Width and height are forced even
        (many codecs reject odd dimensions). Returns a full-frame rect if no
        matte is found.
    """
    h, w = frame_bgr.shape[:2]
    max_rows = int(h * max_crop_frac)
    max_cols = int(w * max_crop_frac)

    top = _scan_edge(frame_bgr, std_thresh, color_tol, max_rows)
    bottom = _scan_edge(frame_bgr[::-1], std_thresh, color_tol, max_rows)
    cols = frame_bgr.transpose(1, 0, 2)  # index by column
    left = _scan_edge(cols, std_thresh, color_tol, max_cols)
    right = _scan_edge(cols[::-1], std_thresh, color_tol, max_cols)

    x0, y0 = left, top
    x1, y1 = w - right, h - bottom
    cw, ch = x1 - x0, y1 - y0

    # Degenerate result — bail to full frame.
    if cw < 16 or ch < 16:
        return (0, 0, w - (w % 2), h - (h % 2))

    # Force even dimensions for codec friendliness.
    cw -= cw % 2
    ch -= ch % 2
    return (x0, y0, cw, ch)


def is_full_frame(crop, frame_shape):
    """True if `crop` covers (essentially) the whole frame — no wrap needed."""
    x, y, w, h = crop
    fh, fw = frame_shape[:2]
    return x == 0 and y == 0 and w >= fw - 1 and h >= fh - 1


class CroppingCapture:
    """
    VideoCapture wrapper that crops every frame to a fixed (x, y, w, h) rect.

    Delegates the small slice of the cv2.VideoCapture API the pipeline uses
    (read / get / set / isOpened / release), so it is a drop-in replacement.
    Every frame returned by read() is already cropped, which keeps all
    downstream pixel coordinates in the cropped frame's space.
    """

    def __init__(self, cap, crop):
        self._cap = cap
        self._x, self._y, self._w, self._h = crop

    def read(self):
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return ret, frame
        cropped = frame[self._y:self._y + self._h, self._x:self._x + self._w]
        return ret, cropped

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def isOpened(self):
        return self._cap.isOpened()

    def release(self):
        return self._cap.release()
