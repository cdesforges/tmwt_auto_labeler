"""
MMPose pose backend.

Runs MMPose's MMPoseInferencer (top-down 2D pose estimation). It predicts
COCO-17 keypoints, which are remapped into the 33-landmark layout (see
pose_common.py).

Install:
    pip install openmim
    mim install mmpose
    mim install mmdet

Notes:
    - `model_path` is an MMPose alias or config path; the default 'human' alias
      uses RTMPose + RTMDet.
    - MMPose has no streaming mode, so both landmarker factories are the same and
      detect_poses ignores its timestamp.
    - Each model is loaded once per process and shared by every landmarker, so
      it isn't reloaded for every video (see _load_inferencer).
"""

from pose_common import coco17_to_landmarks

try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    MMPoseInferencer = None

DEFAULT_MODEL_PATH = "human"


# Loaded MMPoseInferencer models, by model path/alias; see _load_inferencer.
_INFERENCERS = {}


def _load_inferencer(model_path):
    """The MMPoseInferencer for `model_path`, loaded on first use and reused afterwards."""
    if MMPoseInferencer is None:
        raise ImportError("MMPose is not installed. Install with:\n"
                          "    pip install openmim\n"
                          "    mim install mmpose mmdet")
    if model_path not in _INFERENCERS:
        # Force CPU: mmcv's compiled ops (nms, etc.) have no MPS implementation
        # on Apple Silicon, and using MPS crashes with
        # "nms_impl: implementation for device mps:0 not found."
        _INFERENCERS[model_path] = MMPoseInferencer(pose2d=model_path, device="cpu")
    return _INFERENCERS[model_path]


class _Landmarker:
    """A shared MMPoseInferencer plus a pose cap, with the close() the other backends have."""

    def __init__(self, model_path, num_poses):
        self.inferencer = _load_inferencer(model_path)
        self.num_poses = num_poses

    def close(self):
        """Nothing to release: the model stays loaded for the next landmarker."""


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    return _Landmarker(model_path, num_poses)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    return _Landmarker(model_path, num_poses)


def detect_poses_image(landmarker, frame_bgr):
    """Detect poses in one frame. Returns up to num_poses 33-landmark poses."""
    result = next(landmarker.inferencer(frame_bgr, show=False))
    people = (result.get("predictions") or [[]])[0]
    people = sorted(people, key=lambda p: p.get("bbox_score", 0.0), reverse=True)
    h, w = frame_bgr.shape[:2]
    poses = []
    for person in people[:landmarker.num_poses]:
        keypoints = person.get("keypoints", [])
        scores = person.get("keypoint_scores", [1.0] * len(keypoints))
        poses.append(coco17_to_landmarks(keypoints, scores, w, h))
    return poses


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    return detect_poses_image(landmarker, frame_bgr)
