"""
Phase 1 of a run: analyse one video without any user input.

For each frame this records the pose, the ankle midpoint and the camera-drift
homography (job.FrameResult). None of that depends on the rope endpoints, so
if the endpoints are changed later only the timing is recomputed
(timing.update_timing) — pose estimation never runs twice.
"""

import cv2

import endpoints
import pose_common
import timing
import tracking
import video_io
from job import (FrameResult, STATUS_FAILED, STATUS_INCOMPLETE,
                 STATUS_NEEDS_INPUT, STATUS_OK)
from tracking import GroundTracker


def analyze_job(job, model_path, backend, matte_crop=True, on_progress=None):
    """
    Analyse one video: first frame and matte crop, rope endpoints, pose and
    tracking over every frame, and automatic walk timing.

    Sets job.status to one of STATUS_OK / STATUS_INCOMPLETE / STATUS_NEEDS_INPUT
    / STATUS_FAILED (with job.error).

    Args:
        job: the VideoJob to fill in.
        model_path: pose model path/alias (backend-specific).
        backend: pose backend module (see pose_backend.get_backend).
        matte_crop: auto-detect and crop letterbox / pillarbox bars.
        on_progress: optional callback(fraction, frame_bgr, pose), called per frame.
    """
    try:
        job.info = video_io.probe_video(job.path, matte_crop)
    except video_io.VideoError as e:
        job.status, job.error = STATUS_FAILED, str(e)
        return

    image_landmarker = backend.create_image_landmarker(model_path, num_poses=endpoints.MAX_PEOPLE)
    try:
        far_ep, near_ep, problem = endpoints.auto_detect_endpoints(
            job.info.first_frame, image_landmarker, backend)
    finally:
        image_landmarker.close()

    try:
        tracker = GroundTracker(job.info.first_frame)
    except RuntimeError as e:
        job.status, job.error = STATUS_FAILED, str(e)
        return
    landmarker = backend.create_landmarker(model_path)
    try:
        job.frames = track_frames(job, tracker, landmarker, backend, on_progress)
    finally:
        landmarker.close()

    if not any(f.pose is not None for f in job.frames):
        job.status, job.error = STATUS_FAILED, "no body detected in any frame"
    elif problem:
        job.status, job.endpoint_problem = STATUS_NEEDS_INPUT, problem
        print(f"  Rope endpoints need clicking at review: {problem}")
    else:
        job.far_ep, job.near_ep = far_ep, near_ep
        job.endpoint_source = "auto"
        job.far_ep_is_standing_spot = True
        timing.update_timing(job)
        job.status = STATUS_OK if job.duration is not None else STATUS_INCOMPLETE


def track_frames(job, tracker, landmarker, backend, on_progress=None):
    """
    Run pose estimation and ground tracking over every frame of the job's video.

    Returns:
        List of FrameResult, one per frame read.
    """
    info = job.info
    cap = job.open_capture()
    frames = []
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx = len(frames)
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not ts_ms or ts_ms < 0:
                ts_ms = frame_idx / info.fps * 1000.0

            H = tracker.update(frame_bgr)
            poses = backend.detect_poses(landmarker, frame_bgr, ts_ms)
            pose = poses[0] if poses else None
            result = FrameResult(
                frame_idx=frame_idx,
                time_s=ts_ms / 1000.0,
                H=H,
                pose=pose,
                body_px=(pose_common.ankle_midpoint(pose, frame_bgr.shape)
                         if pose is not None else None),
            )
            _locate_in_reference_frame(result, frame_bgr.shape)
            frames.append(result)

            if on_progress is not None:
                total = info.total_frames
                on_progress(len(frames) / total if total > 0 else 0.0, frame_bgr, pose)
    finally:
        cap.release()
    return frames


def _locate_in_reference_frame(result, frame_shape):
    """
    Fill result.ref_* with the subject's foot, head and ankles mapped back into
    the reference frame. Skipped unless both the ankle midpoint and the head
    (nose) were found — those two are what the vanishing-point fit needs.
    """
    if result.body_px is None:
        return
    h, w = frame_shape[:2]
    head = pose_common.landmark_px(result.pose[pose_common.NOSE_IDX], w, h)
    if head is None:
        return
    left = pose_common.landmark_px(result.pose[pose_common.LEFT_ANKLE_IDX], w, h)
    right = pose_common.landmark_px(result.pose[pose_common.RIGHT_ANKLE_IDX], w, h)
    foot, head, left_ref, right_ref = tracking.to_reference_frame(
        result.H, [result.body_px, head, left or result.body_px, right or result.body_px])
    result.ref_foot = foot
    result.ref_head = head
    result.ref_left_ankle = left_ref if left else None
    result.ref_right_ankle = right_ref if right else None
