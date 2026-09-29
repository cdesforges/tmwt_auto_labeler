"""
Automatic rope-endpoint detection from a video's first frame.

The walking course is the line between two endpoints:
  - far endpoint (0 m, start of the walk): where the subject stands, taken as
    their ankle midpoint in the first frame.
  - near endpoint (course end, finish line): a corner of the single ArUco marker
    placed there.

When either can't be found automatically, the user clicks them during review
(LabelerUI.pick_endpoints).
"""

import cv2

import pose_common

# ArUco marker dictionary printed on the finish-line marker.
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, cv2.aruco.DetectorParameters())

# Marker corner used as the near endpoint. OpenCV orders corners top-left,
# top-right, bottom-right, bottom-left.
_NEAR_CORNER = 2

# How many people to look for in the first frame. More than one means the
# subject is ambiguous and the user has to click the endpoints.
MAX_PEOPLE = 5


def detect_near_endpoint(frame_bgr):
    """Corner of the single ArUco marker in the frame, or None (none, or several)."""
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(frame_bgr)
    if ids is None or len(ids.flatten()) != 1:
        return None
    x, y = corners[0][0][_NEAR_CORNER]
    return (int(x), int(y))


def auto_detect_endpoints(frame_bgr, landmarker, backend):
    """
    Detect both rope endpoints without any user input.

    Args:
        frame_bgr: the first real frame of the video.
        landmarker: an IMAGE-mode landmarker from `backend`, created with
            num_poses=MAX_PEOPLE so extra people are noticed.
        backend: pose backend module (see pose_backend.get_backend).

    Returns:
        (far_ep, near_ep, problem). Both endpoints are (x, y) tuples when
        detection worked and problem is None; otherwise the endpoints are None
        and problem is a short human-readable reason.
    """
    near_ep = detect_near_endpoint(frame_bgr)
    if near_ep is None:
        return None, None, "no ArUco marker"

    poses = backend.detect_poses_image(landmarker, frame_bgr)
    feet = [c for c in (pose_common.ankle_midpoint(p, frame_bgr.shape) for p in poses)
            if c is not None]
    if not feet:
        return None, None, "no person at start"
    if len(feet) > 1:
        return None, None, f"{len(feet)} people at start"
    return feet[0], near_ep, None
