"""
Pose layout and helpers shared by every pose backend.

Every backend returns poses in one 35-landmark layout: MediaPipe's 33 landmarks
plus a small toe for each foot (33 left, 34 right). Each landmark has
normalized .x / .y (0-1 across the frame) and .z. Slots a backend doesn't
predict are None — MediaPipe has no small toes; the Halpe-26 "body with feet"
models used by rtmlib and mmpose have no hand or face detail. Everything
downstream — drawing, tracking, timing, CSV export and view.py — works on this
one layout and never needs to know which backend produced it.

Foot points per side (left / right): ankle 27 / 28, heel 29 / 30, big toe
("foot index" in MediaPipe) 31 / 32, small toe 33 / 34.
"""

import cv2

# Number of landmarks in the layout: MediaPipe's 33 plus two small toes.
MEDIAPIPE_LANDMARKS = 33
NUM_LANDMARKS = 35

# Landmark indices used by the pipeline.
NOSE_IDX = 0
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28
LEFT_HEEL_IDX = 29
RIGHT_HEEL_IDX = 30
LEFT_BIG_TOE_IDX = 31
RIGHT_BIG_TOE_IDX = 32
LEFT_SMALL_TOE_IDX = 33
RIGHT_SMALL_TOE_IDX = 34
TOE_IDXS = (LEFT_BIG_TOE_IDX, RIGHT_BIG_TOE_IDX, LEFT_SMALL_TOE_IDX, RIGHT_SMALL_TOE_IDX)

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
    (27, 31), (28, 32),                 # ankle to big toe
    (31, 33), (32, 34),                 # big toe to small toe
]

# Halpe-26 keypoint index -> layout index. Halpe-26 is COCO-17 (0-16) plus
# head, neck, hip (17-19, not kept) and six foot points (20-25). The eyes and
# ears (1-4) aren't kept either.
HALPE26_TO_LAYOUT = {
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
    20: 31,   # left_big_toe
    21: 32,   # right_big_toe
    22: 33,   # left_small_toe
    23: 34,   # right_small_toe
    24: 29,   # left_heel
    25: 30,   # right_heel
}

# Keypoints scoring below this are treated as missing (None).
MIN_KEYPOINT_SCORE = 0.3


class Landmark:
    """A MediaPipe-shaped landmark: normalized .x, .y, plus .z and .visibility."""
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def halpe26_to_landmarks(keypoints, scores, frame_w, frame_h):
    """
    Convert one person's Halpe-26 keypoints into the layout.

    Args:
        keypoints: sequence of 26 (x, y, ...) pixel positions.
        scores: sequence of 26 confidence scores.
        frame_w, frame_h: frame size, used to normalize the positions.

    Returns:
        List of NUM_LANDMARKS Landmark-or-None entries.
    """
    landmarks = [None] * NUM_LANDMARKS
    for src_idx, dst_idx in HALPE26_TO_LAYOUT.items():
        if src_idx >= len(keypoints) or src_idx >= len(scores):
            continue
        score = float(scores[src_idx])
        if score < MIN_KEYPOINT_SCORE:
            continue
        x, y = keypoints[src_idx][:2]
        landmarks[dst_idx] = Landmark(float(x) / frame_w, float(y) / frame_h, 0.0, score)
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
