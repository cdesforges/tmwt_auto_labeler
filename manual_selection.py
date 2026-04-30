"""
Manual selection UI for rope endpoints and person tracking.

Displays the first frame of a video and lets the user:
  1. Click to set the far rope endpoint
  2. Click to set the near rope endpoint
  3. Click on the person they want to track (if multiple detected)
"""

import cv2
import numpy as np
from pose import detect_poses, detect_poses_image, draw_pose, get_pose_center, get_ankle_midpoint

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


def select_person(frame_bgr, landmarker):
    """
    Detect all poses in the first frame and let the user click on the
    person they want to track.

    If only one person is detected, that person is selected automatically.

    Args:
        frame_bgr: The first frame of the video.
        landmarker: A MediaPipe PoseLandmarker instance.

    Returns:
        (pose_index, initial_center) — the index and ankle-center (x, y) of the
        selected pose, or (None, None) if no poses detected or user quits.
    """
    all_poses = detect_poses_image(landmarker, frame_bgr)

    if not all_poses:
        print("  No poses detected in first frame.")
        return None, None

    if len(all_poses) == 1:
        print("  Single person detected — auto-selected.")
        center = get_ankle_midpoint(all_poses[0], frame_bgr.shape)
        return 0, center

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
    for i, pose_lm in enumerate(all_poses):
        color = colors[i % len(colors)]
        draw_pose(display, pose_lm, color=color)
        center = get_ankle_midpoint(pose_lm, frame_bgr.shape)
        centers.append(center)
        # Label each person with a number
        cv2.putText(
            display, str(i + 1), (center[0] - 10, center[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
        )

    cv2.putText(
        display,
        f"{len(all_poses)} people detected. Click the person to track.",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
    )
    cv2.imshow(window, display)

    click = _wait_for_click_or_key(window)
    cv2.destroyWindow(window)

    if click is None:
        return None, None

    # Find the person whose center is closest to the click
    click_arr = np.array(click, dtype=np.float32)
    dists = [np.linalg.norm(np.array(c, dtype=np.float32) - click_arr) for c in centers]
    selected = int(np.argmin(dists))
    print(f"  Selected person {selected + 1} of {len(all_poses)}.")
    return selected, centers[selected]



def detect_endpoints(cap, detector, landmarker):
    """
    Returns (far_ep, near_ep, manual_start_mode):
      far_ep: (x, y) far rope endpoint, or None if pose was not detected.
      near_ep: (x, y) near rope endpoint from the ArUco marker.
      manual_start_mode: True if pose was not detected and the user must
        drive timing with spacebar in the main loop.
      Returns (None, None, False) on failure.
    """
    if not cap.isOpened():
        print(f"  ERROR: Detect endpoints was passed an unopened cap object!")
        return None, None, False

    ret, frame_bgr = cap.read()
    if not ret:
        print("  ERROR: Could not read first frame.")
        return None, None, False

    _, far_ep = select_person(frame_bgr, landmarker)
    manual_start_mode = far_ep is None

    corners, ids, rejected = detector.detectMarkers(frame_bgr)
    print(f"ids = {ids}")
    near_ep = None
    if ids is not None and len(ids.flatten()) == 1:
        near_ep = tuple(corners[0][0][2].astype(int))
    if near_ep is None:
        print("  ERROR: Need exactly one ArUco marker in the first frame.")
        return None, None, False

    window = "Verify Endpoints"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse_click)

    # In manual-start mode, ask the user to click where the person starts walking
    # so the rope direction (far→near) is still defined for end-detection.
    if manual_start_mode:
        display = frame_bgr.copy()
        cv2.circle(display, near_ep, 7, (0, 0, 255), -1)
        cv2.putText(
            display, "Person not detected, manual timer start needed!",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2,
        )
        cv2.putText(
            display, "Use the spacebar once the person has started walking.",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        cv2.putText(
            display, "First, click where the person starts walking.",
            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.imshow(window, display)
        click = _wait_for_click_or_key(window)
        if click is None:
            cv2.destroyWindow(window)
            return None, None, False
        far_ep = click

    while True:
        display = frame_bgr.copy()
        cv2.circle(display, far_ep, 7, (255, 0, 0), -1)
        cv2.circle(display, near_ep, 7, (0, 0, 255), -1)
        cv2.line(display, far_ep, near_ep, (0, 255, 255), 2)
        prompt = (
            "Press any key to BEGIN (spacebar will start the timer), or 'q' to quit"
            if manual_start_mode
            else "Press any key to confirm, or 'q' to quit"
        )
        cv2.putText(
            display, prompt,
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(window, display)

        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window)
        if key == ord("q"):
            return None, None, False
        return far_ep, near_ep, manual_start_mode
