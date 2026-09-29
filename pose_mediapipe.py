"""
MediaPipe pose backend.

Wraps MediaPipe's PoseLandmarker. Its 33 landmarks are the first 33 of the
layout described in pose_common.py (heels and "foot index" included); the two
small-toe slots it doesn't predict are padded with None.
"""

import cv2
import mediapipe as mp

from pose_common import NUM_LANDMARKS

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


def _to_layout(result):
    """MediaPipe's poses, padded to the shared layout's length."""
    return [list(pose) + [None] * (NUM_LANDMARKS - len(pose))
            for pose in (result.pose_landmarks or [])]


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    """Detect poses in one frame of a stream. Returns a list of layout poses."""
    return _to_layout(landmarker.detect_for_video(_to_mp_image(frame_bgr), int(timestamp_ms)))


def detect_poses_image(landmarker, frame_bgr):
    """Detect poses in a single image. Returns a list of layout poses."""
    return _to_layout(landmarker.detect(_to_mp_image(frame_bgr)))
