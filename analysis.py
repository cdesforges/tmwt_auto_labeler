"""
Phase 1 of a run: analyse one video without any user input, in two parts.

process_video is the slow part: every person's pose and the camera-drift
homography in every frame (job.FrameResult), plus the ArUco finish marker. It
can run on a cluster (process_videos.py); its results are saved to an analysis
file (analysis_file.py).

interpret is the fast part, run wherever the review happens: people are linked
into tracks, the walking subject is chosen (people.py), the endpoints are set
and the timing decided (timing.py). Every person's poses are kept, so changing
the endpoints, the subject or the timing rules never re-runs pose estimation.
"""

import cv2

import endpoints
import people
import timing
import video_io
from job import (FrameResult, STATUS_FAILED, STATUS_INCOMPLETE,
                 STATUS_NEEDS_INPUT, STATUS_NO_BODY, STATUS_OK)
from tracking import GroundTracker


def process_video(job, model_path, backend, matte_crop=True, on_progress=None):
    """
    The slow part of the analysis, with no user input: probe the video (first
    frame, matte crop), detect the ArUco finish marker, and run pose estimation
    (everyone in frame) and camera-drift tracking over every frame.

    Fills job.info, job.frames and job.aruco_finish. On failure sets
    job.status = STATUS_FAILED with job.error.

    Args:
        job: the VideoJob to fill in.
        model_path: pose model path/alias (backend-specific).
        backend: pose backend module (see pose_backend.get_backend).
        matte_crop: auto-detect and crop letterbox / pillarbox bars.
        on_progress: optional callback(fraction, frame_bgr, poses), called per frame.
    """
    try:
        job.info = video_io.probe_video(job.path, matte_crop)
    except video_io.VideoError as e:
        job.status, job.error = STATUS_FAILED, str(e)
        return
    try:
        tracker = GroundTracker(job.info.first_frame)
    except RuntimeError as e:
        job.status, job.error = STATUS_FAILED, str(e)
        return

    job.aruco_finish = endpoints.detect_near_endpoint(job.info.first_frame)
    landmarker = backend.create_landmarker(model_path, num_poses=people.MAX_PEOPLE)
    try:
        job.frames = track_frames(job, tracker, landmarker, backend, on_progress)
    finally:
        landmarker.close()


def interpret(job):
    """
    The fast part (seconds), from the per-frame results: follow people and
    choose the subject, set the endpoints, and decide the walk timing.

    Sets job.status to STATUS_OK / STATUS_INCOMPLETE / STATUS_NEEDS_INPUT /
    STATUS_NO_BODY (unless processing already failed).
    """
    if job.status == STATUS_FAILED:
        return
    shape = job.info.first_frame.shape
    job.tracks = people.build_tracks(job.frames, shape, job.info.fps)
    subject = people.choose_subject(job.tracks, job.frames, shape)
    if subject is None:
        job.status, job.error = STATUS_NO_BODY, "no body detected in any frame"
        return
    people.set_subject(job, subject)
    job.subject_start = people.subject_start(job)
    n = len(people.candidates(job.tracks, job.frames))
    if n > 1:
        print(f"  {n} people tracked; subject grew {subject.growth:.1f}x in apparent "
              f"size (walking toward the camera)")

    problem = ("no ArUco marker" if job.aruco_finish is None
               else "subject never fully located" if job.subject_start is None else "")
    if problem:
        job.status, job.endpoint_problem = STATUS_NEEDS_INPUT, problem
        print(f"  Rope endpoints need clicking at review: {problem}")
        return

    job.far_ep, job.near_ep = job.subject_start, job.aruco_finish
    job.endpoint_source = "auto"
    job.far_ep_is_standing_spot = True
    timing.update_timing(job)
    job.status = STATUS_OK if job.duration is not None else STATUS_INCOMPLETE


def track_frames(job, tracker, landmarker, backend, on_progress=None):
    """
    Run pose estimation (everyone in frame) and ground tracking over every frame
    of the job's video.

    Returns:
        List of FrameResult, one per frame read, with `people` filled in.
    """
    info = job.info
    # Frames to process: the container's count, less the black frames skipped at
    # the start. Containers can overstate it, so progress is capped at 1.
    expected = max(1, info.total_frames - info.first_frame_idx)
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

            frame_bgr_last = frame_bgr
            H = tracker.update(frame_bgr)
            poses = backend.detect_poses(landmarker, frame_bgr, ts_ms)
            frames.append(FrameResult(frame_idx=frame_idx, time_s=ts_ms / 1000.0,
                                      H=H, people=list(poses)))
            if on_progress is not None:
                on_progress(min(1.0, len(frames) / expected), frame_bgr, poses)
        if on_progress is not None and frames:
            on_progress(1.0, frame_bgr_last, poses)   # the video may hold fewer frames than stated
    finally:
        cap.release()
    return frames
