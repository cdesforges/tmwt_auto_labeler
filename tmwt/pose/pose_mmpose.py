"""
MMPose pose backend.

Runs MMPose's MMPoseInferencer (top-down 2D pose estimation). The default
`body26` model predicts Halpe-26 keypoints (body plus big toe, small toe and
heel on each foot), which are remapped into the shared landmark layout (see
pose_common.py). COCO-17 models (e.g. `human`) also work, without the feet.

Install:
    pip install openmim
    mim install mmpose
    mim install mmdet

Notes:
    - `model_path` is an MMPose alias or config path; the default 'body26' alias
      is RTMPose-m trained on Halpe-26, with RTMDet as the person detector.
    - MMPose has no streaming mode, so both landmarker factories are the same and
      detect_poses ignores its timestamp.
    - Each model is loaded once per process and shared by every landmarker, so
      it isn't reloaded for every video (see _load_inferencer).
"""

from tmwt.pose.pose_common import halpe26_to_landmarks

try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    MMPoseInferencer = None

DEFAULT_MODEL_PATH = "body26"


# Requested device (see set_device).
_requested_device = "auto"


def set_device(device):
    """Run on "auto" (CUDA if available, else CPU), "cpu" or "cuda". MPS isn't supported."""
    global _requested_device
    _requested_device = device


def _device():
    if _requested_device != "auto":
        return _requested_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def provenance():
    """Device and library versions, for the analysis file."""
    import mmpose
    return {"device": _device(), "versions": {"mmpose": mmpose.__version__}}


# Loaded MMPoseInferencer models, by model path/alias; see _load_inferencer.
_INFERENCERS = {}


def _load_inferencer(model_path):
    """The MMPoseInferencer for `model_path`, loaded on first use and reused afterwards."""
    if MMPoseInferencer is None:
        raise ImportError("MMPose is not installed. Install with:\n"
                          "    pip install openmim\n"
                          "    mim install mmpose mmdet")
    if model_path not in _INFERENCERS:
        # Never MPS: mmcv's compiled ops (nms, etc.) have no MPS implementation
        # on Apple Silicon, and using MPS crashes with
        # "nms_impl: implementation for device mps:0 not found."
        _INFERENCERS[model_path] = MMPoseInferencer(pose2d=model_path, device=_device())
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
    """Detect poses in one frame. Returns up to num_poses layout poses (pose_common)."""
    result = next(landmarker.inferencer(frame_bgr, show=False))
    people = (result.get("predictions") or [[]])[0]
    people = sorted(people, key=lambda p: p.get("bbox_score", 0.0), reverse=True)
    h, w = frame_bgr.shape[:2]
    poses = []
    for person in people[:landmarker.num_poses]:
        keypoints = person.get("keypoints", [])
        scores = person.get("keypoint_scores", [1.0] * len(keypoints))
        poses.append(halpe26_to_landmarks(keypoints, scores, w, h))
    return poses


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    return detect_poses_image(landmarker, frame_bgr)
