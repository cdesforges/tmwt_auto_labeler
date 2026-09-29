"""
Opening videos consistently.

Every pass over a video (analysis, playback, saving) must see exactly the same
frames in the same pixel space. probe_video works that out once — the first
frame with picture content and the matte crop — and open_video reopens the
video with those settings applied.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import matte

# Frames darker than this mean pixel value are treated as leading black frames.
BLACK_FRAME_MEAN = 10
# Fallback frame rate when the container doesn't report one.
DEFAULT_FPS = 30.0


class VideoError(Exception):
    """A video can't be used; the message is a short reason for the report."""


@dataclass
class VideoInfo:
    """How to read one video: where its content starts and how to crop it."""
    first_frame_idx: int     # index of the first non-black frame
    crop: tuple              # (x, y, w, h) matte crop, or None for the full frame
    fps: float
    total_frames: int
    first_frame: np.ndarray  # the first non-black frame, already cropped


def open_video(path, crop=None, start_frame=0):
    """Open `path` with `crop` applied to every frame, positioned at `start_frame`."""
    cap = cv2.VideoCapture(path)
    if crop is not None:
        cap = matte.CroppingCapture(cap, crop)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return cap


def probe_video(path, matte_crop=True):
    """
    Find the first frame with content and (optionally) the matte crop.

    Raises:
        VideoError: if the video can't be opened or is entirely black.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("cannot open video")
    try:
        first_frame_idx, frame = _first_content_frame(cap)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    print(f"  First content frame: {first_frame_idx}")

    crop = None
    if matte_crop:
        detected = matte.detect_content_crop(frame)
        if matte.is_full_frame(detected, frame.shape):
            print("  No matte detected — using full frame.")
        else:
            crop = detected
            x, y, w, h = crop
            print(f"  Matte detected — cropping to {w}x{h} at ({x},{y}) "
                  f"from {frame.shape[1]}x{frame.shape[0]}.")
            frame = frame[y:y + h, x:x + w]

    fps = fps if fps and fps > 0 else DEFAULT_FPS
    print(f"  FPS: {fps:.2f}, total frames: {total_frames}")
    return VideoInfo(first_frame_idx, crop, fps, total_frames, frame)


def _first_content_frame(cap):
    """(index, frame) of the first frame that isn't (near-)black."""
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            raise VideoError("all frames are black")
        if frame.mean() >= BLACK_FRAME_MEAN:
            return idx, frame
        idx += 1


class FrameSource:
    """
    Random access to a job's video frames for playback, with frame k always the
    same frame the analysis saw as frame k.

    Seeking inside a compressed video can land on the wrong frame for some
    formats, so frames are only ever decoded in order, from the start, and kept
    (JPEG-compressed, at full resolution) as they are read. Going back is served
    from memory; going forward decodes on to the frame asked for.
    """

    def __init__(self, job, quality=90):
        self._cap = job.open_capture()
        self._jpeg = []          # compressed frames read so far, in order
        self._quality = quality
        self._last = None        # (index, image) of the latest frame read, to skip a decode

    def get(self, k):
        """Frame k (a BGR image), or None if the video ends before it."""
        while len(self._jpeg) <= k:
            ok, frame = self._cap.read()
            if not ok:
                return None
            self._jpeg.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])[1])
            self._last = (len(self._jpeg) - 1, frame)
        if self._last is not None and self._last[0] == k:
            return self._last[1].copy()
        return cv2.imdecode(self._jpeg[k], cv2.IMREAD_COLOR)

    def close(self):
        self._cap.release()
        self._jpeg = []
