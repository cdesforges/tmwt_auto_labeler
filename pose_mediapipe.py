"""
MediaPipe pose backend.

Wraps MediaPipe's PoseLandmarker, which natively returns the 33-landmark layout
described in pose_common.py.
"""

import cv2
import mediapipe as mp

# Default model path (relative to the project root).
DEFAULT_MODEL_PATH = "models/pose_landmarker_full.task"

# Confidence thresholds passed to the landmarker.
_MIN_CONFIDENCE = 0.5


def _create(model_path, running_mode, num_poses):
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=_MIN_CONFIDENCE,
        min_pose_presence_confidence=_MIN_CONFIDENCE,
        min_tracking_confidence=_MIN_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """VIDEO-mode landmarker for a stream of frames (timestamps must increase)."""
    return _create(model_path, mp.tasks.vision.RunningMode.VIDEO, num_poses)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """IMAGE-mode landmarker for one-off single-frame detection."""
    return _create(model_path, mp.tasks.vision.RunningMode.IMAGE, num_poses)


def _to_mp_image(frame_bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    """Detect poses in one frame of a stream. Returns a list of 33-landmark poses."""
    result = landmarker.detect_for_video(_to_mp_image(frame_bgr), int(timestamp_ms))
    return result.pose_landmarks or []


def detect_poses_image(landmarker, frame_bgr):
    """Detect poses in a single image. Returns a list of 33-landmark poses."""
    result = landmarker.detect(_to_mp_image(frame_bgr))
    return result.pose_landmarks or []
