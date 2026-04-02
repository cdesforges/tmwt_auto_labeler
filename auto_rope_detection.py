"""
Automatic rope endpoint detection.

Uses image processing (CLAHE, top-hat, connected components) to find
the near rope endpoint, then walks upward along the rope to find the
far endpoint. This is the original auto-detection logic preserved
as a standalone module.
"""

import cv2
import numpy as np


def follow_rope_upwards(
    roi_eq,
    y0,
    start_full,
    band_half_width=5,
    max_steps=800,
    max_vertical_step=2,
    min_rel_intensity=0.9,
    max_step_dist=4.0,
):
    """
    Starting from a known rope point, walk upward pixel-by-pixel
    following the brightest connected path.

    The walk uses direction prediction from the last two path points
    and searches in a narrow band around the predicted position.

    Args:
        roi_eq: CLAHE-equalized grayscale image of the ground ROI.
        y0: Y-offset of the ROI within the full frame.
        start_full: (x, y) starting point in full-frame coordinates.
        band_half_width: How far left/right to search at each step.
        max_steps: Maximum number of upward steps.
        max_vertical_step: Max rows to jump per step.
        min_rel_intensity: Minimum relative intensity to continue.
        max_step_dist: Max Euclidean distance per step.

    Returns:
        (far_endpoint, path) where far_endpoint is the last point reached
        and path is a list of (x, y) tuples in full-frame coords.
    """
    h_roi, w = roi_eq.shape
    x_full, y_full = start_full

    curr_x = int(np.clip(x_full, 0, w - 1))
    curr_y = int(np.clip(y_full - y0, 0, h_roi - 1))

    start_int = float(roi_eq[curr_y, curr_x])
    min_intensity = start_int * min_rel_intensity
    max_step_dist2 = max_step_dist * max_step_dist

    path = [(int(x_full), int(y_full))]
    prev_int = start_int

    for _ in range(max_steps):
        if curr_y <= 0:
            break

        # Estimate line direction from last two points
        if len(path) >= 2:
            x2_full, y2_full = path[-1]
            x1_full, y1_full = path[-2]
            x2 = int(np.clip(x2_full, 0, w - 1))
            y2 = int(np.clip(y2_full - y0, 0, h_roi - 1))
            x1 = int(np.clip(x1_full, 0, w - 1))
            y1 = int(np.clip(y1_full - y0, 0, h_roi - 1))
            vx = x2 - x1
            vy = y2 - y1
        else:
            vx, vy = 0.0, -1.0  # First step: straight up

        # Force upward direction
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            vx, vy = 0.0, -1.0
        if vy > 0:
            vx, vy = -vx, -vy

        found = False
        best_x = None
        best_y = None

        # Scan rows above current point
        for dy in range(1, max_vertical_step + 1):
            y = curr_y - dy
            if y < 0:
                break

            # Predict where the rope should be at this row
            if abs(vy) > 1e-3:
                t = (y - curr_y) / float(vy)
                x_center_f = curr_x + t * vx
            else:
                x_center_f = float(curr_x)
            x_center = int(round(x_center_f))

            # Search left/right around predicted x (center-out)
            for dx in range(0, band_half_width + 1):
                for sign in (+1, -1) if dx > 0 else (+1,):
                    x = x_center + sign * dx
                    if x < 0 or x >= w:
                        continue

                    val = float(roi_eq[y, x])
                    if val < min_intensity:
                        continue
                    if val < min_rel_intensity * prev_int:
                        continue

                    ddx = x - curr_x
                    ddy = y - curr_y
                    if ddx * ddx + ddy * ddy > max_step_dist2:
                        continue

                    best_x = x
                    best_y = y
                    found = True
                    break
                if found:
                    break
            if found:
                break

        if not found:
            break

        curr_x, curr_y = best_x, best_y
        prev_int = float(roi_eq[curr_y, curr_x])
        path.append((int(curr_x), int(curr_y + y0)))

    return path[-1], path


def detect_rope_endpoints_auto(first_frame, show_debug=True):
    """
    Automatically detect the near and far rope endpoints.

    Strategy:
      1. Build a bright-thin-structure mask in the ground ROI.
      2. Use connected components to find the bottom-most elongated segment
         (this is the near end of the rope).
      3. Walk upward from the near endpoint to find the far endpoint.

    Args:
        first_frame: First video frame in BGR format.
        show_debug: If True, display debug windows showing detection.

    Returns:
        (far_endpoint, near_endpoint) as (x, y) tuples,
        or (None, None) if detection fails.
    """
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Ground ROI (bottom 70% of frame)
    y0 = int(h * 0.3)
    roi = gray[y0:, :]

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)

    # Top-hat to emphasize thin bright structures (like a rope/string)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(roi_eq, cv2.MORPH_TOPHAT, kernel)

    # Otsu threshold to get candidate bright pixels
    _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean the mask with morphological open/close
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        print("No connected components found in rope mask.")
        if show_debug:
            cv2.imshow("Rope tophat", tophat)
            cv2.imshow("Rope mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None, None

    debug_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Find the near-end segment: bottom-most elongated component
    best_label = None
    best_score = -1.0

    print("Connected components (for near-end search):")
    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]
        if area < 50:
            continue

        aspect = max(float(w_box), float(h_box)) / (min(float(w_box), float(h_box)) + 1e-3)
        bottom_y_full = y + h_box + y0

        # Score: prefer components that are low and elongated
        score = bottom_y_full * np.sqrt(area) * (1.0 + 0.5 * aspect)

        print(
            f"  label={label:3d}, x={x:4d}, y={y:4d}, w={w_box:4d}, h={h_box:4d}, "
            f"area={area:5d}, aspect={aspect:5.2f}, bottom_y_full={bottom_y_full:5.1f}, "
            f"score={score:10.1f}"
        )
        cv2.rectangle(debug_color, (x, y), (x + w_box, y + h_box), (255, 0, 0), 1)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        print("No usable near segment found.")
        if show_debug:
            cv2.imshow("Rope tophat", tophat)
            cv2.imshow("Rope mask", mask)
            cv2.imshow("Rope components (ROI)", debug_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None, None

    # Near endpoint = bottom-most pixel of the best component
    ys_lbl, xs_lbl = np.where(labels == best_label)
    ys_full_lbl = ys_lbl + y0
    near_idx = np.argmax(ys_full_lbl)
    near_endpoint = (int(xs_lbl[near_idx]), int(ys_full_lbl[near_idx]))

    print(f"\nSelected label {best_label} as near-end segment (score={best_score:.1f})")
    print(f"  Near endpoint: {near_endpoint}")

    # Walk upward from near endpoint to find the far endpoint
    far_endpoint, path = follow_rope_upwards(roi_eq, y0, near_endpoint)
    print(f"  Far endpoint (from walking): {far_endpoint}")

    if show_debug:
        debug_full = first_frame.copy()
        mask_full = np.zeros_like(gray)
        mask_full[y0:, :] = mask
        mask_full_color = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(debug_full, 0.7, mask_full_color, 0.3, 0)

        for (px, py) in path:
            cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)

        cv2.circle(overlay, far_endpoint, 8, (255, 0, 0), -1)
        cv2.circle(overlay, near_endpoint, 8, (0, 0, 255), -1)

        cv2.imshow("Rope tophat", tophat)
        cv2.imshow("Rope mask", mask)
        cv2.imshow("Rope components (ROI)", debug_color)
        cv2.imshow("Rope detection (path)", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return far_endpoint, near_endpoint
