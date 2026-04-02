"""
Ground-plane tracking via optical flow and homography.

Uses Lucas-Kanade optical flow on ground features to estimate
a homography matrix each frame. This lets us track where the
rope endpoints have moved as the camera shifts.
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

    Computes a homography from the initial frame's feature positions
    to the current frame, then uses it to transform arbitrary points
    (like rope endpoints) from initial-frame coords to current-frame coords.
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

    @staticmethod
    def transform_points(H, points):
        """
        Apply a homography to a list of (x, y) points.

        Args:
            H: 3x3 homography matrix.
            points: List of (x, y) tuples to transform.

        Returns:
            List of (x, y) tuples in the new coordinate frame.
            Returns the original points unchanged if H is None.
        """
        if H is None:
            return list(points)

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, H)
        return [(int(p[0][0]), int(p[0][1])) for p in transformed]
