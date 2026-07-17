"""
RTMLib backend for pose detection.

Uses rtmlib's ONNX-Runtime-based Body pipeline — the same RTMPose model weights
that MMPose uses, but through a much lighter inference stack. Supports Apple
Silicon acceleration via CoreML (device='mps') without any of MMPose's mmcv
op-availability caveats.

Predicts COCO-17 keypoints and remaps them into MediaPipe's 33-landmark layout
so downstream code (CSV export, view.py, drawing helpers) stays compatible
with the other backends.

Install:
    pip install rtmlib onnxruntime

Notes:
    - `model_path` selects rtmlib's Body mode: "balanced" (default),
      "performance" (larger, more accurate), or "lightweight" (smaller, faster).
    - Device defaults to 'mps' on Apple Silicon (CoreML EP), else 'cpu'.
      Override via the RTMLIB_DEVICE environment variable if needed.
    - RTMLib has no VIDEO vs IMAGE running-mode distinction; both landmarker
      factories return the same wrapper.
    - `timestamp_ms` in detect_poses is accepted for API parity but ignored.
    - Landmark slots that don't map from COCO-17 are returned as None so
      downstream code can skip them.
"""

import os
import platform

import cv2

try:
    from rtmlib import Body
    _RTMLIB_AVAILABLE = True
except ImportError:
    Body = None
    _RTMLIB_AVAILABLE = False


# Default mode. Valid values: "balanced", "performance", "lightweight".
DEFAULT_MODEL_PATH = "balanced"

NUM_LANDMARKS = 33

POSE_CONNECTIONS = [
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28
FACE_IDXS = set(range(0, 11))
NOSE_IDX = 0

_COCO_TO_MP = {
    0: 0,     # nose
    5: 11,    # left_shoulder
    6: 12,    # right_shoulder
    7: 13,    # left_elbow
    8: 14,    # right_elbow
    9: 15,    # left_wrist
    10: 16,   # right_wrist
    11: 23,   # left_hip
    12: 24,   # right_hip
    13: 25,   # left_knee
    14: 26,   # right_knee
    15: 27,   # left_ankle
    16: 28,   # right_ankle
}

_MIN_KEYPOINT_SCORE = 0.3


class _Landmark:
    """MediaPipe-shaped landmark with .x, .y, .z, .visibility (all normalized)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _default_device():
    """Pick a sensible ONNX Runtime execution provider for this machine."""
    override = os.environ.get("RTMLIB_DEVICE")
    if override:
        return override
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


class _BodyWrapper:
    """Thin wrapper so rtmlib.Body has the same close/infer surface as other backends."""

    def __init__(self, mode, num_poses):
        if not _RTMLIB_AVAILABLE:
            raise ImportError(
                "rtmlib is not installed. Install with:\n"
                "    pip install rtmlib onnxruntime"
            )
        if mode not in ("balanced", "performance", "lightweight"):
            raise ValueError(
                f"Invalid rtmlib mode: {mode!r}. "
                "Use 'balanced', 'performance', or 'lightweight'."
            )
        self._body = Body(
            mode=mode,
            backend="onnxruntime",
            device=_default_device(),
        )
        self.num_poses = num_poses

    def infer(self, frame_bgr):
        """Return (keypoints (N,17,2), scores (N,17)) from rtmlib."""
        return self._body(frame_bgr)

    def close(self):
        # onnxruntime handles teardown at GC time.
        pass


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """Streaming inferencer. `num_poses` caps the returned list."""
    return _BodyWrapper(model_path, num_poses)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """One-shot inferencer — rtmlib has no streaming mode distinction."""
    return _BodyWrapper(model_path, num_poses)


def _keypoints_to_poses(keypoints, scores, frame_shape, num_poses):
    """Convert rtmlib's (N,17,2)/(N,17) arrays into MediaPipe-shaped landmarks."""
    h, w = frame_shape[:2]
    # rtmlib already sorts detections by descending detector score.
    n = min(len(keypoints), num_poses)
    poses = []
    for i in range(n):
        person_kpts = keypoints[i]
        person_scores = scores[i]
        landmarks = [None] * NUM_LANDMARKS
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            score = float(person_scores[coco_idx])
            if score < _MIN_KEYPOINT_SCORE:
                continue
            x, y = person_kpts[coco_idx][:2]
            landmarks[mp_idx] = _Landmark(
                x=float(x) / w, y=float(y) / h, z=0.0, visibility=score
            )
        poses.append(landmarks)
    return poses


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    """Streaming detection. `timestamp_ms` accepted for API parity, ignored."""
    keypoints, scores = landmarker.infer(frame_bgr)
    return _keypoints_to_poses(keypoints, scores, frame_bgr.shape, landmarker.num_poses)


def detect_poses_image(landmarker, frame_bgr):
    """One-shot detection."""
    keypoints, scores = landmarker.infer(frame_bgr)
    return _keypoints_to_poses(keypoints, scores, frame_bgr.shape, landmarker.num_poses)


def draw_pose(frame_bgr, pose_landmarks, color=(0, 255, 0),
              point_radius=4, line_thickness=2):
    """Draw the skeleton, skipping None (unmapped or low-score) landmarks."""
    h, w = frame_bgr.shape[:2]

    for a, b in POSE_CONNECTIONS:
        la, lb = pose_landmarks[a], pose_landmarks[b]
        if la is None or lb is None:
            continue
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), color, line_thickness)

    for i, lm in enumerate(pose_landmarks):
        if lm is None:
            continue
        if i in FACE_IDXS and i != NOSE_IDX:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), point_radius, color, -1)


def get_ankle_midpoint(pose_landmarks, frame_shape):
    """
    Ankle midpoint in pixel coords. Falls back to a single ankle if only one
    is present, or None if both were dropped by the score threshold.
    """
    h, w = frame_shape[:2]
    left = pose_landmarks[LEFT_ANKLE_IDX]
    right = pose_landmarks[RIGHT_ANKLE_IDX]
    if left is not None and right is not None:
        x = int((left.x + right.x) * 0.5 * w)
        y = int((left.y + right.y) * 0.5 * h)
        return (x, y)
    if left is not None:
        return (int(left.x * w), int(left.y * h))
    if right is not None:
        return (int(right.x * w), int(right.y * h))
    return None
