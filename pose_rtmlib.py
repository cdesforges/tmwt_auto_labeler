"""
RTMLib pose backend.

Uses rtmlib's ONNX-Runtime BodyWithFeet pipeline — RTMPose trained on
Halpe-26, the same weights MMPose's `body26` model uses, through a much lighter
inference stack, with Apple Silicon acceleration via CoreML (device='mps'). Its
26 keypoints (body plus big toe, small toe and heel on each foot) are remapped
into the shared landmark layout (see pose_common.py).

Install:
    pip install rtmlib onnxruntime

Notes:
    - `model_path` selects rtmlib's Body mode: "balanced" (default),
      "performance" (larger, more accurate), or "lightweight" (smaller, faster).
    - Device (set_device): "auto" picks 'cuda' when ONNX Runtime has CUDA,
      'mps' (CoreML) on Apple Silicon, else 'cpu'. The RTMLIB_DEVICE
      environment variable overrides it.
    - Models are downloaded on first use into $TORCH_HOME/hub, else
      $XDG_CACHE_HOME/rtmlib/hub, else ~/.cache/rtmlib/hub.
    - rtmlib has no streaming mode, so both landmarker factories are the same and
      detect_poses ignores its timestamp.
    - Loading a model takes several seconds (CoreML compiles it), and the models
      keep no state between frames, so each mode is loaded once per process and
      shared by every landmarker (see _load_body).
    - Frames with nobody in them return no poses. rtmlib's own Body() call
      instead runs the pose model over the whole frame when the detector finds
      no one, producing a phantom low-confidence skeleton; and on CoreML the
      detector itself can fail on such frames (see _detect_people).
"""

import os
import platform

from pose_common import halpe26_to_landmarks

try:
    import onnxruntime
    from rtmlib import BodyWithFeet as Body
    from onnxruntime.capi.onnxruntime_pybind11_state import Fail as OrtFail
except ImportError:
    Body = None
    OrtFail = None
else:
    # Silence ONNX Runtime's own console log (fatal only). Otherwise it prints an
    # error line for every empty frame that _detect_people deliberately handles,
    # plus CoreML partitioning warnings at load. Real failures still surface as
    # exceptions carrying the same message.
    onnxruntime.set_default_logger_severity(4)

# Default Body mode.
DEFAULT_MODEL_PATH = "balanced"
_MODES = ("balanced", "performance", "lightweight")


# Requested device (see set_device).
_requested_device = "auto"


def set_device(device):
    """Run on "auto", "cpu", "cuda" or "mps" (CoreML); see _device."""
    global _requested_device
    _requested_device = device


def _device():
    """The device to run on: RTMLIB_DEVICE, else the requested one, else detected."""
    override = os.environ.get("RTMLIB_DEVICE")
    if override:
        return override
    if _requested_device != "auto":
        return _requested_device
    if Body is not None and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return "cuda"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def provenance():
    """Device and library versions, for the analysis file."""
    import rtmlib
    return {"device": _device(),
            "versions": {"rtmlib": getattr(rtmlib, "__version__", "?"),
                         "onnxruntime": onnxruntime.__version__}}


# Loaded rtmlib.BodyWithFeet models, by (mode, device); see _load_body.
_BODIES = {}


def _load_body(mode):
    """The rtmlib.BodyWithFeet for `mode`, loaded on first use and reused afterwards."""
    if Body is None:
        raise ImportError("rtmlib is not installed. Install with:\n"
                          "    pip install rtmlib onnxruntime")
    if mode not in _MODES:
        raise ValueError(f"Invalid rtmlib mode: {mode!r}. Use one of {_MODES}.")
    key = (mode, _device())
    if key not in _BODIES:
        _BODIES[key] = Body(mode=mode, backend="onnxruntime", device=key[1])
    return _BODIES[key]


class _Landmarker:
    """A shared rtmlib.BodyWithFeet plus a pose cap, with the close() the other backends have."""

    def __init__(self, mode, num_poses):
        self.body = _load_body(mode)
        self.num_poses = num_poses

    def close(self):
        """Nothing to release: the model stays loaded for the next landmarker."""


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    return _Landmarker(model_path, num_poses)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    return _Landmarker(model_path, num_poses)


def _detect_people(body, frame_bgr):
    """
    Person bounding boxes in the frame (possibly none).

    CoreML can't run the detector's post-processing on an empty tensor, so on
    Apple Silicon a frame with nobody in it can raise instead of returning no
    boxes ("... has zero elements. This is not supported by the CoreML EP").
    That case is treated as no people; any other error is raised.
    """
    try:
        return body.det_model(frame_bgr)
    except OrtFail as e:
        if "zero elements" in str(e):
            return []
        raise


def detect_poses_image(landmarker, frame_bgr):
    """Detect poses in one frame. Returns up to num_poses layout poses (pose_common)."""
    body = landmarker.body
    if getattr(body, "one_stage", False):
        keypoints, scores = body.pose_model(frame_bgr)
    else:
        boxes = _detect_people(body, frame_bgr)
        if len(boxes) == 0:
            return []
        keypoints, scores = body.pose_model(frame_bgr, bboxes=boxes)
    h, w = frame_bgr.shape[:2]
    # rtmlib already sorts people by descending detector score.
    n = min(len(keypoints), landmarker.num_poses)
    return [halpe26_to_landmarks(keypoints[i], scores[i], w, h) for i in range(n)]


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    return detect_poses_image(landmarker, frame_bgr)
