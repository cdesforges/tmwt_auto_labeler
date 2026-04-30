"""
Pose detection and skeleton drawing using MediaPipe.

Wraps MediaPipe's PoseLandmarker for per-frame pose estimation
and provides a function to draw the detected skeleton on a frame.
"""

import cv2
import mediapipe as mp

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = "models/pose_landmarker_full.task"

# Skeleton connections (pairs of landmark indices to draw lines between)
POSE_CONNECTIONS = [
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (30, 32),
]

# Landmark indices for ankle midpoint (used as body tracking point)
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28

# Face landmark indices (0-10) — we skip these when drawing body points
FACE_IDXS = set(range(0, 11))
NOSE_IDX = 0


def create_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """VIDEO-mode landmarker for streaming frames."""
    mp_tasks = mp.tasks
    options = mp_tasks.vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_tasks.vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.8,
        min_pose_presence_confidence=0.8,
        min_tracking_confidence=0.8,
    )
    return mp_tasks.vision.PoseLandmarker.create_from_options(options)


def create_image_landmarker(model_path=DEFAULT_MODEL_PATH, num_poses=1):
    """IMAGE-mode landmarker for one-off single-frame detection (e.g. manual selection)."""
    mp_tasks = mp.tasks
    options = mp_tasks.vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp_tasks.vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=0.8,
        min_pose_presence_confidence=0.8,
    )
    return mp_tasks.vision.PoseLandmarker.create_from_options(options)


def detect_poses(landmarker, frame_bgr, timestamp_ms):
    """
    Run pose detection on a single frame.

    Args:
        landmarker: A PoseLandmarker instance.
        frame_bgr: The frame in BGR format (from cv2.VideoCapture).
        timestamp_ms: Frame timestamp in milliseconds (must be monotonically increasing).

    Returns:
        List of pose landmark lists. Each pose is a list of landmarks with .x, .y, .z attributes
        (normalized 0-1). Returns empty list if no poses detected.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
    return result.pose_landmarks if result.pose_landmarks else []


def detect_poses_image(landmarker, frame_bgr):
    """Single-frame pose detection (IMAGE mode — no timestamps)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    return result.pose_landmarks if result.pose_landmarks else []


def draw_pose(frame_bgr, pose_landmarks, color=(0, 255, 0), point_radius=4, line_thickness=2):
    """
    Draw a single pose skeleton on the frame (modifies frame in-place).

    Draws skeleton lines, body landmarks (skipping face except nose),
    and a single nose dot for the head.

    Args:
        frame_bgr: The frame to draw on (modified in-place).
        pose_landmarks: List of landmarks for one pose.
        color: BGR color tuple for the skeleton.
        point_radius: Radius of landmark circles.
        line_thickness: Thickness of skeleton lines.
    """
    h, w = frame_bgr.shape[:2]

    # Draw skeleton lines
    for a, b in POSE_CONNECTIONS:
        la, lb = pose_landmarks[a], pose_landmarks[b]
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), color, line_thickness)

    # Draw body landmarks (skip face indices)
    for i, lm in enumerate(pose_landmarks):
        if i in FACE_IDXS:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), point_radius, color, -1)

    # Draw nose as single face point
    lm = pose_landmarks[NOSE_IDX]
    cv2.circle(frame_bgr, (int(lm.x * w), int(lm.y * h)), point_radius, color, -1)


def get_ankle_midpoint(pose_landmarks, frame_shape):
    """
    Compute the midpoint of left and right ankles in pixel coordinates.

    This serves as the body tracking point for measuring position along the rope.

    Args:
        pose_landmarks: List of landmarks for one pose.
        frame_shape: Shape of the frame (h, w, ...).

    Returns:
        (x, y) tuple in pixel coordinates.
    """
    h, w = frame_shape[:2]
    left = pose_landmarks[LEFT_ANKLE_IDX]
    right = pose_landmarks[RIGHT_ANKLE_IDX]
    x = int((left.x + right.x) * 0.5 * w)
    y = int((left.y + right.y) * 0.5 * h)
    return (x, y)


def get_pose_center(pose_landmarks, frame_shape):
    """
    Compute the center of a pose as the midpoint of the two hip landmarks.
    Used for matching a user click to the correct person.

    Args:
        pose_landmarks: List of landmarks for one pose.
        frame_shape: Shape of the frame (h, w, ...).

    Returns:
        (x, y) tuple in pixel coordinates.
    """
    h, w = frame_shape[:2]
    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]
    x = int((left_hip.x + right_hip.x) * 0.5 * w)
    y = int((left_hip.y + right_hip.y) * 0.5 * h)
    return (x, y)
