"""
Manual selection UI for rope endpoints and person tracking.

Displays the first frame of a video and lets the user:
  1. Click to set the far rope endpoint
  2. Click to set the near rope endpoint
  3. Click on the person they want to track (if multiple detected)
"""

import cv2
import numpy as np

# ---- State for mouse callbacks ----

_click_point = None  # Stores the most recent click


def _on_mouse_click(event, x, y, flags, param):
    """Mouse callback that records left-click coordinates."""
    global _click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point = (x, y)


def _wait_for_click_or_key(window_name, cancel_key=None):
    """
    Block until the user clicks or presses a specific key.

    Args:
        window_name: Name of the OpenCV window to watch.
        cancel_key: Optional key character that returns a special sentinel.

    Returns:
        (x, y) pixel coordinates of the click,
        "cancel" if cancel_key was pressed,
        or None if 'q' was pressed.
    """
    global _click_point
    _click_point = None
    while _click_point is None:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            return None
        if cancel_key and key == ord(cancel_key):
            return "cancel"
    return _click_point


def select_rope_endpoints(frame_bgr):
    """
    Show the first frame and ask the user to click the two rope endpoints.
    After both are selected, the user can press 'u' to undo and reselect,
    or any other key to confirm.

    The user clicks:
      1. The FAR endpoint (start of the walk, further from camera)
      2. The NEAR endpoint (end of the walk, closer to camera)

    Args:
        frame_bgr: The first frame of the video.

    Returns:
        (far_endpoint, near_endpoint) as (x, y) tuples,
        or (None, None) if the user quits.
    """
    window = "Select Rope Endpoints"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse_click)

    while True:
        # --- Click 1: Far endpoint ---
        display = frame_bgr.copy()
        cv2.putText(
            display,
            "Click the FAR rope endpoint (start of walk) | q=quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(window, display)

        far_ep = _wait_for_click_or_key(window)
        if far_ep is None:
            cv2.destroyWindow(window)
            return None, None

        # --- Click 2: Near endpoint ---
        display = frame_bgr.copy()
        cv2.circle(display, far_ep, 7, (255, 0, 0), -1)  # Blue = far
        cv2.putText(
            display,
            "Click the NEAR rope endpoint (end of walk) | q=quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(window, display)

        near_ep = _wait_for_click_or_key(window)
        if near_ep is None:
            cv2.destroyWindow(window)
            return None, None

        # --- Show result with undo option ---
        display = frame_bgr.copy()
        cv2.circle(display, far_ep, 7, (255, 0, 0), -1)
        cv2.circle(display, near_ep, 7, (0, 0, 255), -1)
        cv2.line(display, far_ep, near_ep, (0, 255, 255), 2)
        cv2.putText(
            display,
            "Press 'u' to undo and reselect, any other key to confirm.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(window, display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("u"):
            # Undo — loop back to the start
            continue
        else:
            # Confirm selection
            cv2.destroyWindow(window)
            return far_ep, near_ep


def select_person(frame_bgr, landmarker, backend):
    """
    Detect all poses in the first frame and let the user click on the
    person they want to track.

    Args:
        frame_bgr: The first frame of the video.
        landmarker: A pose landmarker instance from `backend`.
        backend: Pose backend module (see pose_backend.get_backend).

    Returns:
        (pose_index, initial_center) — the index and ankle-center (x, y) of the
        selected pose, or (None, None) if no usable poses or user quits.
    """
    all_poses = backend.detect_poses_image(landmarker, frame_bgr)

    # Keep only poses with a resolvable ankle midpoint — some backends may
    # return a pose whose ankle landmarks were dropped below the confidence
    # threshold, and downstream code needs a real (x, y) far_ep.
    poses_with_centers = []
    for pose_lm in all_poses:
        center = backend.get_ankle_midpoint(pose_lm, frame_bgr.shape)
        if center is not None:
            poses_with_centers.append((pose_lm, center))

    if not poses_with_centers:
        print("  No poses detected in first frame.")
        return None, None

    if len(poses_with_centers) == 1:
        print("  Single person detected — auto-selected.")
        return 0, poses_with_centers[0][1]

    # Multiple people: draw them all with different colors and ask user to click
    colors = [
        (0, 255, 0),    # green
        (0, 0, 255),    # red
        (255, 0, 0),    # blue
        (255, 255, 0),  # cyan
        (0, 255, 255),  # yellow
    ]

    window = "Select Person to Track"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse_click)

    display = frame_bgr.copy()
    centers = []
    for i, (pose_lm, center) in enumerate(poses_with_centers):
        color = colors[i % len(colors)]
        backend.draw_pose(display, pose_lm, color=color)
        centers.append(center)
        cv2.putText(
            display, str(i + 1), (center[0] - 10, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

    cv2.putText(
        display,
        f"{len(poses_with_centers)} people detected. Click the person to track.",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    cv2.imshow(window, display)

    click = _wait_for_click_or_key(window)
    cv2.destroyWindow(window)

    if click is None:
        return None, None

    click_arr = np.array(click, dtype=np.float32)
    dists = [np.linalg.norm(np.array(c, dtype=np.float32) - click_arr) for c in centers]
    selected = int(np.argmin(dists))
    print(f"  Selected person {selected + 1} of {len(poses_with_centers)}.")
    return selected, centers[selected]



def _detect_aruco_near_endpoint(detector, frame_bgr):
    """Detect exactly one ArUco marker and return its top-right corner, or None."""
    corners, ids, _ = detector.detectMarkers(frame_bgr)
    print(f"ids = {ids}")
    if ids is not None and len(ids.flatten()) == 1:
        return tuple(corners[0][0][2].astype(int))
    return None


def auto_detect_endpoints(frame_bgr, detector, landmarker, backend):
    """
    Detect both rope endpoints from the first frame without any user input.

    - Far endpoint: the ankle midpoint of the only person in frame (they stand
      at the start of the walk).
    - Near endpoint: the corner of the single ArUco marker at the finish line.

    Returns:
        (far_ep, near_ep, problem). Both endpoints are (x, y) tuples when
        detection worked and problem is None; otherwise the endpoints are None
        and problem is a short human-readable reason.
    """
    near_ep = _detect_aruco_near_endpoint(detector, frame_bgr)
    if near_ep is None:
        return None, None, "no ArUco marker"

    all_poses = backend.detect_poses_image(landmarker, frame_bgr)
    centers = [c for c in (backend.get_ankle_midpoint(p, frame_bgr.shape) for p in all_poses)
               if c is not None]
    if not centers:
        return None, None, "no person at start"
    if len(centers) > 1:
        return None, None, f"{len(centers)} people at start"
    return centers[0], (int(near_ep[0]), int(near_ep[1])), None
