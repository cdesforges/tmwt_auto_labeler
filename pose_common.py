"""
Pose layout and helpers shared by every pose backend.

Every backend returns poses in MediaPipe's 33-landmark layout: a list of 33
landmarks, each with normalized .x / .y (0-1 across the frame) and .z. Backends
that predict fewer keypoints (the COCO-17 models used by rtmlib and mmpose)
leave the unmapped slots as None. Everything downstream — drawing, tracking,
CSV export and view.py — works on that one layout and never needs to know which
backend produced it.
"""

import cv2

# Number of landmarks in the MediaPipe pose layout.
NUM_LANDMARKS = 33

# Landmark indices used by the pipeline.
NOSE_IDX = 0
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28

# Face landmarks (0-10). Only the nose is drawn; the rest are skipped.
FACE_IDXS = set(range(0, 11))

# Skeleton lines (pairs of landmark indices). Connections whose landmarks are
# missing (None) are skipped when drawing, so this full MediaPipe set also works
# for COCO-17 backends that don't fill in hands and feet.
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

# COCO-17 keypoint index -> MediaPipe-33 landmark index. COCO keypoints not
# listed here (eyes, ears) are dropped.
COCO_TO_MP = {
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

# COCO keypoints scoring below this are treated as missing (None).
MIN_KEYPOINT_SCORE = 0.3


class Landmark:
    """A MediaPipe-shaped landmark: normalized .x, .y, plus .z and .visibility."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def coco17_to_landmarks(keypoints, scores, frame_w, frame_h):
    """
    Convert one person's COCO-17 keypoints into the MediaPipe-33 layout.

    Args:
        keypoints: sequence of 17 (x, y, ...) pixel positions.
        scores: sequence of 17 confidence scores.
        frame_w, frame_h: frame size, used to normalize the positions.

    Returns:
        List of 33 Landmark-or-None entries.
    """
    landmarks = [None] * NUM_LANDMARKS
    for coco_idx, mp_idx in COCO_TO_MP.items():
        if coco_idx >= len(keypoints) or coco_idx >= len(scores):
            continue
        score = float(scores[coco_idx])
        if score < MIN_KEYPOINT_SCORE:
            continue
        x, y = keypoints[coco_idx][:2]
        landmarks[mp_idx] = Landmark(float(x) / frame_w, float(y) / frame_h, 0.0, score)
    return landmarks


def landmark_px(landmark, frame_w, frame_h):
    """A landmark's (x, y) position in float pixels, or None if it is missing."""
    if landmark is None:
        return None
    return (landmark.x * frame_w, landmark.y * frame_h)


def ankle_midpoint(pose, frame_shape):
    """
    The body tracking point: midpoint of the two ankles, in integer pixels.

    Falls back to a single ankle if only one was detected; None if neither was.
    """
    h, w = frame_shape[:2]
    ankles = [lm for lm in (pose[LEFT_ANKLE_IDX], pose[RIGHT_ANKLE_IDX]) if lm is not None]
    if not ankles:
        return None
    x = sum(lm.x for lm in ankles) / len(ankles)
    y = sum(lm.y for lm in ankles) / len(ankles)
    return (int(x * w), int(y * h))


def draw_pose(img, pose, color=(0, 255, 0), point_radius=4, line_thickness=2):
    """
    Draw one pose's skeleton on `img` (in place): the connection lines, every
    body landmark, and the nose as the single face point. Missing landmarks
    are skipped.
    """
    h, w = img.shape[:2]

    def px(lm):
        return (int(lm.x * w), int(lm.y * h))

    for a, b in POSE_CONNECTIONS:
        if pose[a] is not None and pose[b] is not None:
            cv2.line(img, px(pose[a]), px(pose[b]), color, line_thickness)

    for i, lm in enumerate(pose):
        if lm is None or (i in FACE_IDXS and i != NOSE_IDX):
            continue
        cv2.circle(img, px(lm), point_radius, color, -1)
