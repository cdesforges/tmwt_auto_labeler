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

import numpy as np

# How far a pixel may sit from the matte's seed color (L1 across B,G,R) and
# still count as matte. ~10 per channel.
DEFAULT_COLOR_TOL = 30.0

# Fraction of a line's pixels that must be within COLOR_TOL for it to count as
# matte. Deliberately below 1.0: video compression puts ringing artifacts in
# otherwise-flat matte bars near the picture edge, and a handful of outlier
# pixels must not disqualify the line. Measured on real footage, matte lines
# score >= 0.986 (even the noisy ones) while content lines score <= 0.04, so
# anything in that gap works.
DEFAULT_MIN_MATTE_FRAC = 0.90

# A line this uniform against its own median is treated as a fresh matte band,
# letting the scan cross a colour change (e.g. a grey border over a black bar).
# Real photographic content essentially never reaches this.
RESEED_UNIFORM_FRAC = 0.995

# Never trim more than this fraction from any single side — a guard against
# pathological frames where most of the picture happens to be uniform.
DEFAULT_MAX_CROP_FRAC = 0.45


def _matte_fraction(flat, seed, color_tol):
    """Fraction of a line's pixels within `color_tol` (L1) of `seed`."""
    return float((np.abs(flat - seed).sum(axis=1) < color_tol).mean())


def _scan_edge(lines, color_tol, min_frac, max_count):
    """
    Count consecutive matte lines from index 0 inward.

    Uses "what fraction of this line matches the matte colour" rather than the
    line's variance. Variance is unreliable here because compression ringing in
    a flat bar can spike it well past a content-like value, while the colour
    itself stays put.

    Args:
        lines: array shaped (N, pixels_per_line, channels) ordered from the edge
               inward (so lines[0] is the outermost row/column).
        color_tol, min_frac: matte thresholds.
        max_count: hard cap on how many lines may be consumed.

    Returns:
        Number of leading lines that are part of the matte (0 if the edge line
        already carries content).
    """
    if len(lines) == 0:
        return 0

    channels = lines.shape[-1]
    seed = np.median(lines[0].reshape(-1, channels).astype(np.float32), axis=0)

    count = 0
    for i in range(min(len(lines), max_count)):
        flat = lines[i].reshape(-1, channels).astype(np.float32)
        if _matte_fraction(flat, seed, color_tol) >= min_frac:
            count += 1
            continue

        # Colour changed. If this line is near-perfectly uniform it is a new
        # matte band, so adopt its colour and keep going; otherwise it is
        # picture content and the matte ends here.
        own_seed = np.median(flat, axis=0)
        if _matte_fraction(flat, own_seed, color_tol) >= RESEED_UNIFORM_FRAC:
            seed = own_seed
            count += 1
            continue
        break
    return count


def detect_content_crop(frame_bgr,
                        color_tol=DEFAULT_COLOR_TOL,
                        min_frac=DEFAULT_MIN_MATTE_FRAC,
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

    top = _scan_edge(frame_bgr, color_tol, min_frac, max_rows)
    bottom = _scan_edge(frame_bgr[::-1], color_tol, min_frac, max_rows)
    cols = frame_bgr.transpose(1, 0, 2)  # index by column
    left = _scan_edge(cols, color_tol, min_frac, max_cols)
    right = _scan_edge(cols[::-1], color_tol, min_frac, max_cols)

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
