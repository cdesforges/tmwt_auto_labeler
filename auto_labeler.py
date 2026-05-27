import cv2
import mediapipe as mp
import numpy as np

VIDEO_PATH = "media/no_aruco/SV_10MWRT_string.MOV"
MODEL_PATH = "models/pose_landmarker_full.task"

# ------------------ Pose skeleton + params ------------------

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

LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28

# Thresholds in t-space | t=0 at far endpoint, t=1 at near endpoint
FAR_T_THRESHOLD = 0.0
NEAR_T_THRESHOLD = 1.0

# Exponential smoothing factor for t_along
SMOOTH_ALPHA = 0.7  # 0.7 current frame, 0.3 previous

# ------------------ Helper functions ------------------

def draw_pose(frame_bgr, pose_landmarks, *, point_radius=4, line_thickness=2):
    h, w = frame_bgr.shape[:2]

    # draw skeleton lines
    for a, b in POSE_CONNECTIONS:
        la = pose_landmarks[a]
        lb = pose_landmarks[b]
        xa, ya = int(la.x * w), int(la.y * h)
        xb, yb = int(lb.x * w), int(lb.y * h)
        cv2.line(frame_bgr, (xa, ya), (xb, yb), (0, 255, 0), line_thickness)

    FACE_IDXS = set(range(0, 11))  # mediapipe face landmarks

    # draw body landmarks
    for i, lm in enumerate(pose_landmarks):
        if i in FACE_IDXS:
            continue

        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), point_radius, (0, 255, 0), -1)

    # draw single face point
    NOSE_IDX = 0
    lm = pose_landmarks[NOSE_IDX]
    x = int(lm.x * w)
    y = int(lm.y * h)
    cv2.circle(frame_bgr, (x, y), point_radius, (0, 255, 0), -1)


def compute_body_point_px(pose_landmarks, frame_shape):
    """Use midpoint of left/right ankles in pixel coords."""
    h, w = frame_shape[:2]
    left = pose_landmarks[LEFT_ANKLE_IDX]
    right = pose_landmarks[RIGHT_ANKLE_IDX]
    x = int((left.x + right.x) * 0.5 * w)
    y = int((left.y + right.y) * 0.5 * h)
    return (x, y)


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
    h_roi, w = roi_eq.shape
    x_full, y_full = start_full

    curr_x = int(np.clip(x_full, 0, w - 1))             # clip to make sure we stay within image bounds
    curr_y = int(np.clip(y_full - y0, 0, h_roi - 1))    # curr_x and curr_y are back in full image coordinates

    start_int = float(roi_eq[curr_y, curr_x])
    min_intensity = start_int * min_rel_intensity
    max_step_dist2 = max_step_dist * max_step_dist

    path = [(int(x_full), int(y_full))]                        # list of tuples containing path coordinates in order we will build
    prev_int = start_int

    for _ in range(max_steps):
        if curr_y <= 0:
            break

        # Estimate line direction from last two points
        if len(path) >= 2:
            x2_full, y2_full = path[-1]
            x1_full, y1_full = path[-2]

            # Convert both to ROI coords
            x2 = int(np.clip(x2_full, 0, w - 1))
            y2 = int(np.clip(y2_full - y0, 0, h_roi - 1))
            x1 = int(np.clip(x1_full, 0, w - 1))
            y1 = int(np.clip(y1_full - y0, 0, h_roi - 1))

            vx = x2 - x1
            vy = y2 - y1
        else:
            vx, vy = 0.0, -1.0   # first step: straight up

        # If direction is degenerate or going down, force straight up
        if abs(vx) < 1e-3 and abs(vy) < 1e-3:
            vx, vy = 0.0, -1.0
        if vy > 0:   # make sure we move upwards in image coordinates
            vx, vy = -vx, -vy

        found = False
        best_x = None
        best_y = None

        # Scan rows above current point
        for dy in range(1, max_vertical_step + 1):
            y = curr_y - dy
            if y < 0:
                break

            # Predict x where the line (curr_x, curr_y) with direction (vx,vy)
            # would intersect this row y.
            if abs(vy) > 1e-3:
                t = (y - curr_y) / float(vy)
                x_center_f = curr_x + t * vx
            else:
                x_center_f = float(curr_x)

            x_center = int(round(x_center_f))

            # Now look left/right around this predicted x, center-out
            for dx in range(0, band_half_width + 1):
                for sign in (+1, -1) if dx > 0 else (+1,):
                    x = x_center + sign * dx
                    if x < 0 or x >= w:
                        continue

                    val = float(roi_eq[y, x])

                    # Intensity constraints
                    if val < min_intensity:
                        continue
                    if val < min_rel_intensity * prev_int:
                        continue

                    # Distance constraint
                    ddx = x - curr_x
                    ddy = y - curr_y
                    dist2 = ddx * ddx + ddy * ddy
                    if dist2 > max_step_dist2:
                        continue

                    # First acceptable pixel -> next rope point
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

        new_y_full = curr_y + y0
        path.append((int(curr_x), int(new_y_full)))

    far_full = path[-1]
    return far_full, path


def detect_rope_endpoints_auto(first_frame, show_debug=True):
    """
    1) Build a bright-thin-structure mask in the ground ROI (region of interest).
    2) Use connected components to find a bottom-most sizable, elongated segment
       (near rope segment).
    3) Define near endpoint as the bottom-most pixel in that segment.
    4) Starting from near endpoint, walk upward along the rope in roi_eq using
       follow_rope_upwards to find the far endpoint.
    """
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Ground ROI (bottom 70%)
    y0 = int(h * 0.3)
    roi = gray[y0:, :]

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)

    # Top-hat to emphasize thin bright structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(roi_eq, cv2.MORPH_TOPHAT, kernel)

    # Threshold to get candidate bright pixels
    _, mask = cv2.threshold(
        tophat, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Clean mask a bit
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    # Connected components in ROI
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

    # Find near segment: bottom-most elongated component
    best_label = None
    best_score = -1.0
    band_half_width_for_walk = 12  # default; will update from bbox

    print("Connected components (for near-end search):")
    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]

        if area < 50:
            continue  # tiny specks

        width = float(w_box)
        height = float(h_box)
        aspect = max(width, height) / (min(width, height) + 1e-3)

        bottom_y_roi = y + h_box
        bottom_y_full = bottom_y_roi + y0

        # Prefer components that are low in the image and somewhat elongated
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
            # Use this bbox to set initial horizontal band for walking
            band_half_width_for_walk = max(int(w_box / 2) + 4, 10)

    if best_label is None:
        print("No usable near segment found.")
        if show_debug:
            cv2.imshow("Rope tophat", tophat)
            cv2.imshow("Rope mask", mask)
            cv2.imshow("Rope components (ROI)", debug_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return None, None

    # Near endpoint = bottom-most pixel of best_label
    ys_lbl, xs_lbl = np.where(labels == best_label)
    ys_full_lbl = ys_lbl + y0
    xs_full_lbl = xs_lbl
    near_idx = np.argmax(ys_full_lbl)
    near_endpoint = (int(xs_full_lbl[near_idx]), int(ys_full_lbl[near_idx]))

    x, y, w_box, h_box, area = stats[best_label]
    x_center = x + w_box / 2.0

    print(f"\nSelected label {best_label} as near-end segment (score={best_score:.1f})")
    print(f"  Near endpoint: {near_endpoint}")
    print(f"  Initial band half-width for walking: {band_half_width_for_walk}")

    # Walk upward from near_endpoint along rope
    far_endpoint, path = follow_rope_upwards(
        roi_eq,
        y0,
        near_endpoint
    )

    print("  Far endpoint (from walking):", far_endpoint)

    if show_debug:
        gray_full = gray
        debug_full = first_frame.copy()

        # Visualize mask over full frame
        mask_full = np.zeros_like(gray_full)
        mask_full[y0:, :] = mask
        mask_full_color = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(debug_full, 0.7, mask_full_color, 0.3, 0)

        # Draw path
        for (px, py) in path:
            cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)

        # Endpoints
        cv2.circle(overlay, far_endpoint, 8, (255, 0, 0), -1)
        cv2.circle(overlay, near_endpoint, 8, (0, 0, 255), -1)

        cv2.imshow("Rope tophat", tophat)
        cv2.imshow("Rope mask", mask)
        cv2.imshow("Rope components (ROI)", debug_color)
        cv2.imshow("Rope detection (path)", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return far_endpoint, near_endpoint


def detect_ground_points(frame_bgr, num_points=300):
    """Find good ground features (bottom part of image) for homography tracking."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[int(h * 0.4):, :] = 255  # bottom 60 %

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=num_points,
        qualityLevel=0.01,
        minDistance=8,
        mask=mask
    )
    return pts

# ------------------ MediaPipe setup ------------------

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = mp_tasks.vision.RunningMode
PoseLandmarker = mp_tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp_tasks.vision.PoseLandmarkerOptions
mp_image_module = mp.Image

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ------------------ Video + initialization ------------------

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video!")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(400 / fps) if fps and fps > 0 else 30
print("FPS:", fps)

# First frame: auto rope detection + ground features
ret, first_frame = cap.read()
if not ret:
    print("Failed to read first frame.")
    cap.release()
    raise SystemExit(1)

far_ep0, near_ep0 = detect_rope_endpoints_auto(first_frame, show_debug=True)
if far_ep0 is None or near_ep0 is None:
    print("Rope detection failed, exiting.")
    cap.release()
    raise SystemExit(1)

far_ep0_np = np.array([[far_ep0[0], far_ep0[1]]], dtype=np.float32)
near_ep0_np = np.array([[near_ep0[0], near_ep0[1]]], dtype=np.float32)

p0_ground = detect_ground_points(first_frame)
if p0_ground is None or len(p0_ground) < 10:
    print("Not enough ground features to track, exiting.")
    cap.release()
    raise SystemExit(1)

print(f"Initial ground features: {len(p0_ground)}")

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p_prev = p0_ground.copy()
p0 = p0_ground.copy()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# ------------------ Main loop ------------------

with PoseLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0

    t_far = None
    t_near = None
    far_hit = False
    near_hit = False

    max_t_seen = -1e9

    prev_t_smooth = None
    prev_time_s = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ts_ms or ts_ms < 0:
            ts_ms = (frame_idx / fps) * 1000.0 if fps and fps > 0 else frame_idx * delay
        frame_idx += 1
        time_s = ts_ms / 1000.0

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # homography for the ground plane
        p_next, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p_prev, None, **lk_params)
        if p_next is not None:
            good_new = p_next[st == 1]
            good_old = p0[st == 1]
            if len(good_new) >= 4:
                H, mask_H = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
            else:
                H = None
        else:
            H = None

        old_gray = frame_gray.copy()
        if p_next is not None:
            p_prev = p_next.copy()

        # Track endpoints via homography
        far_ep_curr = far_ep0
        near_ep_curr = near_ep0
        if H is not None:
            pts0 = np.vstack([far_ep0_np, near_ep0_np])          # (2,2)
            pts0_h = cv2.convertPointsToHomogeneous(pts0).reshape(-1, 3).T  # (3,2)
            pts_curr_h = H @ pts0_h
            pts_curr = (pts_curr_h[:2, :] / pts_curr_h[2, :]).T  # (2,2)
            far_ep_curr = (int(pts_curr[0, 0]), int(pts_curr[0, 1]))
            near_ep_curr = (int(pts_curr[1, 0]), int(pts_curr[1, 1]))

        # pose estimation
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp_image_module(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        result = landmarker.detect_for_video(mp_image, int(ts_ms))

        if result.pose_landmarks:
            pose_lm = result.pose_landmarks[0]
            draw_pose(frame_bgr, pose_lm)

            body_px = compute_body_point_px(pose_lm, frame_bgr.shape)
            cv2.circle(frame_bgr, body_px, 5, (0, 0, 255), -1)

            # draw rope
            cv2.circle(frame_bgr, far_ep_curr, 7, (255, 0, 0), -1)
            cv2.circle(frame_bgr, near_ep_curr, 7, (0, 0, 255), -1)
            cv2.line(frame_bgr, far_ep_curr, near_ep_curr, (0, 255, 255), 2)

            # 1D depth along rope: t_along
            Ax, Ay = far_ep_curr
            Bx, By = near_ep_curr
            Px, Py = body_px

            v = np.array([Bx - Ax, By - Ay], dtype=np.float32)
            w_vec = np.array([Px - Ax, Py - Ay], dtype=np.float32)

            vv = float(v.dot(v))
            if vv > 1e-6:
                t_raw = float(v.dot(w_vec) / vv)  # 0 at far, 1 at near
                if t_raw > max_t_seen:
                    max_t_seen = t_raw

                # Exponential smoothing
                if prev_t_smooth is None:
                    t_smooth = t_raw
                else:
                    t_smooth = SMOOTH_ALPHA * t_raw + (1.0 - SMOOTH_ALPHA) * prev_t_smooth

                # If we have a previous smoothed t, we can check for threshold crossings
                if prev_t_smooth is not None:
                    # Far plane crossing
                    if (not far_hit) and (prev_t_smooth < FAR_T_THRESHOLD) and (t_smooth >= FAR_T_THRESHOLD):
                        if t_smooth != prev_t_smooth:
                            frac = (FAR_T_THRESHOLD - prev_t_smooth) / (t_smooth - prev_t_smooth)
                        else:
                            frac = 0.0
                        t_far = prev_time_s + frac * (time_s - prev_time_s)
                        far_hit = True
                        print(f"Hit FAR plane at {t_far:.3f} s (t_smooth={t_smooth:.3f})")

                    # Near plane crossing
                    if far_hit and (not near_hit) and (prev_t_smooth < NEAR_T_THRESHOLD) and (t_smooth >= NEAR_T_THRESHOLD):
                        if t_smooth != prev_t_smooth:
                            frac = (NEAR_T_THRESHOLD - prev_t_smooth) / (t_smooth - prev_t_smooth)
                        else:
                            frac = 0.0
                        t_near = prev_time_s + frac * (time_s - prev_time_s)
                        near_hit = True
                        print(f"Hit NEAR plane at {t_near:.3f} s (t_smooth={t_smooth:.3f})")

                # Initialize far plane if we start already inside
                if prev_t_smooth is None and (not far_hit) and (t_smooth >= FAR_T_THRESHOLD):
                    t_far = time_s
                    far_hit = True
                    print(f"Hit FAR plane at {t_far:.3f} s (initial)")

                prev_t_smooth = t_smooth
                prev_time_s = time_s

        cv2.putText(
            frame_bgr,
            "Press 'q' to quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Video", frame_bgr)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if t_far is not None and t_near is not None:
        dt = t_near - t_far
        print(f"Time between far and near planes: {dt:.3f} seconds")
        print(f"Average speed over 10 m: {10.0 / dt:.3f} m/s")
    else:
        print("Did not detect both crossings.")

    print(f"Max t_along observed (raw): {max_t_seen:.3f}")