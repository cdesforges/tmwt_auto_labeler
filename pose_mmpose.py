"""
MMPose backend for pose detection.

Uses MMPose's MMPoseInferencer to run top-down 2D pose estimation. Predicts
COCO-17 keypoints and remaps them into MediaPipe's 33-landmark layout so
downstream code (CSV export, view.py, drawing helpers) stays compatible
between backends.

Install:
    pip install openmim
    mim install mmpose
    mim install mmdet

Notes:
    - MMPose has no VIDEO vs IMAGE running-mode distinction; both landmarker
      factories return the same wrapper.
    - `timestamp_ms` in detect_poses is accepted for API parity but ignored.
    - Unmapped landmark slots (face detail, hands, feet) are returned as None
      so downstream code can skip them.
"""

import cv2

try:
    from mmpose.apis import MMPoseInferencer
    _MMPOSE_AVAILABLE = True
except ImportError:
    MMPoseInferencer = None
    _MMPOSE_AVAILABLE = False


# Default model — the 'human' alias uses RTMPose + RTMDet (COCO-17 output).
# Override via --model with any alias or config path recognized by MMPose.
DEFAULT_MODEL_PATH = "human"

NUM_LANDMARKS = 33

# Same skeleton constants as pose.py — shared consumers (view.py, data_export.py)
# don't need to know which backend produced the landmarks.
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

# COCO-17 index → MediaPipe-33 index. Missing MP indices stay None.
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

# Keep landmarks below this score as None (uncertain detections).
_MIN_KEYPOINT_SCORE = 0.3


class _Landmark:
    """MediaPipe-shaped landmark with .x, .y, .z, .visibility (all normalized)."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _InferencerWrapper:
    """
    Thin wrapper around MMPoseInferencer so it can be closed and reused
    like a MediaPipe landmarker.
    """

    def __init__(self, model_path, num_poses):
        if not _MMPOSE_AVAILABLE:
            raise ImportError(
                "MMPose is not installed. Install with:\n"
                "    pip install openmim\n"
                "    mim install mmpose mmdet"
            )
        # Force CPU: mmcv's compiled ops (nms, etc.) have no MPS implementation
        # on Apple Silicon. Trying to use MPS crashes with:
        #   "nms_impl: implementation for device mps:0 not found."
        self._inferencer = MMPoseInferencer(pose2d=model_path, device="cpu")
        self.num_poses = num_poses

    def infer(self, frame_bgr):
        """Run one inference pass; return the list of instance dicts for this frame."""
        gen = self._inferencer(frame_bgr, show=False)
        result = next(gen)
        predictions = result.get("predictions") or [[]]
        return predictions[0]

    def close(self):
        # MMPose has no explicit release step — GC handles it.
        pass


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """Streaming-mode inferencer. `num_poses` caps the returned list."""
    return _InferencerWrapper(model_path, num_poses)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """One-shot inferencer — MMPose doesn't distinguish modes."""
    return _InferencerWrapper(model_path, num_poses)


def _predictions_to_poses(predictions, frame_shape, num_poses):
    """Convert mmpose predictions into MediaPipe-shaped landmark lists."""
    h, w = frame_shape[:2]
    predictions = sorted(
        predictions, key=lambda p: p.get("bbox_score", 0.0), reverse=True
    )[:num_poses]

    poses = []
    for pred in predictions:
        keypoints = pred.get("keypoints", [])
        scores = pred.get("keypoint_scores", [1.0] * len(keypoints))
        landmarks = [None] * NUM_LANDMARKS
        for coco_idx, mp_idx in _COCO_TO_MP.items():
            if coco_idx >= len(keypoints):
                continue
            score = float(scores[coco_idx]) if coco_idx < len(scores) else 0.0
            if score < _MIN_KEYPOINT_SCORE:
                continue
            x, y = keypoints[coco_idx][:2]
            landmarks[mp_idx] = _Landmark(
                x=x / w, y=y / h, z=0.0, visibility=score
            )
        poses.append(landmarks)
    return poses


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    """Streaming detection. `timestamp_ms` is accepted for API parity and ignored."""
    predictions = landmarker.infer(frame_bgr)
    return _predictions_to_poses(predictions, frame_bgr.shape, landmarker.num_poses)


def detect_poses_image(landmarker, frame_bgr):
    """One-shot detection."""
    predictions = landmarker.infer(frame_bgr)
    return _predictions_to_poses(predictions, frame_bgr.shape, landmarker.num_poses)


def draw_pose(frame_bgr, pose_landmarks, color=(0, 255, 0),
              point_radius=4, line_thickness=2):
    """
    Draw the skeleton on the frame. Landmarks that were not filled by the
    COCO→MP mapping (or fell below the score threshold) are skipped.
    """
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
