"""
Finding the finish line: the ArUco marker placed at the near endpoint.

The walking course is the line between two endpoints:
  - far endpoint (0 m, start of the walk): where the subject stands at the
    start (people.subject_start).
  - near endpoint (course end, finish line): a corner of the single ArUco marker
    placed there, found here in the video's first frame.

Without a marker the user clicks the finish point during review, with the start
point pre-placed (LabelerUI.pick_endpoints).
"""

import cv2

# ArUco marker dictionary printed on the finish-line marker.
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, cv2.aruco.DetectorParameters())

# Marker corner used as the near endpoint. OpenCV orders corners top-left,
# top-right, bottom-right, bottom-left.
_NEAR_CORNER = 2


def detect_near_endpoint(frame_bgr):
    """Corner of the single ArUco marker in the frame, or None (none, or several)."""
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(frame_bgr)
    if ids is None or len(ids.flatten()) != 1:
        return None
    x, y = corners[0][0][_NEAR_CORNER]
    return (int(x), int(y))
