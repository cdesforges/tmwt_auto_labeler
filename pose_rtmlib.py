"""
RTMLib pose backend.

Uses rtmlib's ONNX-Runtime Body pipeline — the same RTMPose weights MMPose uses,
through a much lighter inference stack, with Apple Silicon acceleration via
CoreML (device='mps'). It predicts COCO-17 keypoints, which are remapped into
the 33-landmark layout (see pose_common.py).

Install:
    pip install rtmlib onnxruntime

Notes:
    - `model_path` selects rtmlib's Body mode: "balanced" (default),
      "performance" (larger, more accurate), or "lightweight" (smaller, faster).
    - Device defaults to 'mps' on Apple Silicon, else 'cpu'. Override with the
      RTMLIB_DEVICE environment variable.
    - rtmlib has no streaming mode, so both landmarker factories are the same and
      detect_poses ignores its timestamp.
"""

import os
import platform

from pose_common import coco17_to_landmarks

try:
    from rtmlib import Body
except ImportError:
    Body = None

# Default Body mode.
DEFAULT_MODEL_PATH = "balanced"
_MODES = ("balanced", "performance", "lightweight")


def _default_device():
    """RTMLIB_DEVICE if set, else 'mps' on Apple Silicon, else 'cpu'."""
    override = os.environ.get("RTMLIB_DEVICE")
    if override:
        return override
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


class _Landmarker:
    """rtmlib.Body plus the pose cap, with the close() the other backends have."""

    def __init__(self, mode, num_poses):
        if Body is None:
            raise ImportError("rtmlib is not installed. Install with:\n"
                              "    pip install rtmlib onnxruntime")
        if mode not in _MODES:
            raise ValueError(f"Invalid rtmlib mode: {mode!r}. Use one of {_MODES}.")
        self.body = Body(mode=mode, backend="onnxruntime", device=_default_device())
        self.num_poses = num_poses

    def close(self):
        """Nothing to release; onnxruntime tears down at garbage collection."""


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    return _Landmarker(model_path, num_poses)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    return _Landmarker(model_path, num_poses)


def detect_poses_image(landmarker, frame_bgr):
    """Detect poses in one frame. Returns up to num_poses 33-landmark poses."""
    keypoints, scores = landmarker.body(frame_bgr)
    h, w = frame_bgr.shape[:2]
    # rtmlib already sorts people by descending detector score.
    n = min(len(keypoints), landmarker.num_poses)
    return [coco17_to_landmarks(keypoints[i], scores[i], w, h) for i in range(n)]


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    return detect_poses_image(landmarker, frame_bgr)
