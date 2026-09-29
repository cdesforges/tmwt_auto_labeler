"""
Phase 1 of a run: analyse one video without any user input.

For each frame this records every person's pose and the camera-drift homography
(job.FrameResult). People are then linked into tracks and the walking subject is
chosen (people.py). None of that depends on the rope endpoints, and every
person's poses are kept, so changing the endpoints or the subject later only
recomputes the timing — pose estimation never runs twice.
"""

import cv2

import endpoints
import people
import timing
import video_io
from job import (FrameResult, STATUS_FAILED, STATUS_INCOMPLETE,
                 STATUS_NEEDS_INPUT, STATUS_NO_BODY, STATUS_OK)
from tracking import GroundTracker


def analyze_job(job, model_path, backend, matte_crop=True, on_progress=None):
    """
    Analyse one video: first frame and matte crop, every person's pose and the
    camera drift over every frame, the subject, the rope endpoints, and the
    automatic walk timing.

    Sets job.status to one of STATUS_OK / STATUS_INCOMPLETE / STATUS_NEEDS_INPUT
    / STATUS_NO_BODY / STATUS_FAILED (the last two with job.error).

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

    landmarker = backend.create_landmarker(model_path, num_poses=people.MAX_PEOPLE)
    try:
        job.frames = track_frames(job, tracker, landmarker, backend, on_progress)
    finally:
        landmarker.close()

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

    near_ep = endpoints.detect_near_endpoint(job.info.first_frame)
    problem = ("no ArUco marker" if near_ep is None
               else "subject never fully located" if job.subject_start is None else "")
    if problem:
        job.status, job.endpoint_problem = STATUS_NEEDS_INPUT, problem
        print(f"  Rope endpoints need clicking at review: {problem}")
        return

    job.far_ep, job.near_ep = job.subject_start, near_ep
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
            frames.append(FrameResult(frame_idx=frame_idx, time_s=ts_ms / 1000.0,
                                      H=H, people=list(poses)))
            if on_progress is not None:
                total = info.total_frames
                on_progress(len(frames) / total if total > 0 else 0.0, frame_bgr, poses)
    finally:
        cap.release()
    return frames
