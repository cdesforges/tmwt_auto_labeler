"""
Ground-plane tracking via optical flow and homography.

Hand-held or bumped cameras move during a recording. GroundTracker follows
features on the floor with Lucas-Kanade optical flow and fits a homography from
the first (reference) frame to each later frame. The rope endpoints are chosen
in the reference frame and mapped into every frame with transform_points; the
subject's positions are mapped back with to_reference_frame so the whole walk
is measured in one consistent frame.
"""

import cv2
import numpy as np

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def detect_ground_features(frame_bgr, num_points=300):
    """
    Find good features to track in the bottom portion of the frame.

    These ground-plane features are used to estimate the homography
    between the first frame and subsequent frames, so we can track
    where the rope endpoints move over time.

    Args:
        frame_bgr: The frame in BGR format.
        num_points: Max number of feature points to detect.

    Returns:
        Array of shape (N, 1, 2) with feature point coordinates,
        or None if detection fails.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Only look in the bottom 60% of the frame (ground region)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[int(h * 0.4):, :] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=num_points,
        qualityLevel=0.01,
        minDistance=8,
        mask=mask,
    )
    return pts


class GroundTracker:
    """
    Tracks ground-plane features across frames using optical flow.

    Each update() returns the homography from the first frame's feature
    positions to the current frame's.
    """

    def __init__(self, first_frame_bgr):
        """
        Initialize with the first frame. Detects ground features automatically.

        Args:
            first_frame_bgr: The first video frame in BGR format.

        Raises:
            RuntimeError: If not enough ground features are found.
        """
        self.p0 = detect_ground_features(first_frame_bgr)
        if self.p0 is None or len(self.p0) < 10:
            raise RuntimeError("Not enough ground features to track.")

        self.old_gray = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2GRAY)
        self.p_prev = self.p0.copy()

    def update(self, frame_bgr):
        """
        Process a new frame and compute the homography from frame 0.

        Args:
            frame_bgr: The current frame in BGR format.

        Returns:
            3x3 homography matrix (np.ndarray), or None if tracking fails.
        """
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Track features from previous frame to current frame
        p_next, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, frame_gray, self.p_prev, None, **LK_PARAMS
        )

        H = None
        if p_next is not None:
            good_new = p_next[st == 1]
            good_old = self.p0[st == 1]
            if len(good_new) >= 4:
                H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
            self.p_prev = p_next.copy()

        self.old_gray = frame_gray.copy()
        return H


def transform_points(H, points):
    """
    Map reference-frame (first-frame) pixel points into the current frame.

    Args:
        H: 3x3 reference -> current homography from GroundTracker.update, or
           None if tracking failed (points are then returned unchanged).
        points: list of (x, y) points.

    Returns:
        List of integer (x, y) tuples, ready for drawing.
    """
    if H is None:
        return [tuple(p) for p in points]
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return [(int(p[0][0]), int(p[0][1])) for p in transformed]


def to_reference_frame(H, points):
    """
    Map current-frame pixel points back into reference-frame coordinates.

    Keeping the walk track in one consistent frame means camera drift doesn't
    corrupt the vanishing-point fit or the distance measurements.

    Args:
        H: 3x3 reference -> current homography, or None (points are then
           returned unchanged).
        points: list of (x, y) points.

    Returns:
        List of float (x, y) tuples.
    """
    if H is None:
        return list(points)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return list(points)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_inv)
    return [(float(p[0][0]), float(p[0][1])) for p in out]
