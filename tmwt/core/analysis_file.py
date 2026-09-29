"""
Analysis files: the slow part of the analysis (analysis.process_video), saved so
the review can happen later or elsewhere — process.py on a cluster, then
review.py locally.

One file per video, next to the videos:

    <videos folder>/tmwt_analysis/<video file name>.npz

A NumPy .npz archive (FORMAT_VERSION 1) with:

  meta     JSON: format_version, video (file name), video_size and
           video_fingerprint (to check the local video is the one analysed),
           status and error (if processing failed), first_frame_idx, crop, fps,
           total_frames, frame_w, frame_h, aruco_finish, backend, model, device,
           max_people, library versions and when it was created.
  times    (n,) float64          timestamp of each frame, seconds
  H        (n, 3, 3) float64     reference -> frame homography (NaN where lost)
  people   (n,) int16            number of people detected in each frame
  poses    (n, P, L, 4) float64  x, y, z, visibility of landmark l (pose_common
                                 layout) of person p; NaN where missing

It holds coordinates only, no images, so it's de-identified like the CSVs.
Values are stored at full precision, so a review run from the file gives exactly
the same results as one straight after processing.
"""

import hashlib
import json
import os
from datetime import datetime

import numpy as np

from tmwt.pose import pose_common
from tmwt.core import video_io
from tmwt.core.job import STATUS_FAILED, FrameResult
from tmwt.core.video_io import VideoInfo

FORMAT_VERSION = 1
ANALYSIS_DIR = "tmwt_analysis"
# Bytes read from each end of a video for its fingerprint.
_FINGERPRINT_CHUNK = 1 << 20


class AnalysisFileError(Exception):
    """An analysis file is missing or can't be used; the message says why."""


def analysis_path(video_path):
    """Where a video's analysis file lives: <folder>/tmwt_analysis/<file name>.npz."""
    folder, name = os.path.split(os.path.abspath(video_path))
    return os.path.join(folder, ANALYSIS_DIR, name + ".npz")


def fingerprint(video_path):
    """A quick identity check for a video: its size plus a hash of its first and last MiB."""
    size = os.path.getsize(video_path)
    h = hashlib.sha1(str(size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(_FINGERPRINT_CHUNK))
        if size > 2 * _FINGERPRINT_CHUNK:
            f.seek(-_FINGERPRINT_CHUNK, os.SEEK_END)
        h.update(f.read(_FINGERPRINT_CHUNK))
    return h.hexdigest()


# --- Saving --------------------------------------------------------------------

def save(job, provenance):
    """
    Write the job's processing results to its analysis file.

    Args:
        job: a VideoJob after analysis.process_video (possibly failed).
        provenance: dict of backend, model, device and library versions.
    """
    path = analysis_path(job.path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    info = job.info
    meta = {
        "format_version": FORMAT_VERSION,
        "video": os.path.basename(job.path),
        "video_size": os.path.getsize(job.path),
        "video_fingerprint": fingerprint(job.path),
        "status": STATUS_FAILED if job.status == STATUS_FAILED else "processed",
        "error": job.error,
        "first_frame_idx": info.first_frame_idx if info else None,
        "crop": list(info.crop) if info and info.crop else None,
        "fps": info.fps if info else None,
        "total_frames": info.total_frames if info else None,
        "frame_w": int(info.first_frame.shape[1]) if info else None,
        "frame_h": int(info.first_frame.shape[0]) if info else None,
        "aruco_finish": list(job.aruco_finish) if job.aruco_finish else None,
        "created": datetime.now().isoformat(timespec="seconds"),
        **provenance,
    }

    n = len(job.frames)
    max_people = max((len(f.people) for f in job.frames), default=0)
    times = np.array([f.time_s for f in job.frames], dtype=np.float64)
    H = np.full((n, 3, 3), np.nan)
    counts = np.zeros(n, dtype=np.int16)
    poses = np.full((n, max_people, pose_common.NUM_LANDMARKS, 4), np.nan)
    for k, f in enumerate(job.frames):
        if f.H is not None:
            H[k] = f.H
        counts[k] = len(f.people)
        for p, pose in enumerate(f.people):
            for i, lm in enumerate(pose):
                if lm is not None:
                    vis = getattr(lm, "visibility", None)
                    poses[k, p, i] = (lm.x, lm.y, lm.z, 1.0 if vis is None else vis)

    # Write to a temporary name first, so an interrupted run never leaves a
    # half-written file that looks finished.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), times=times, H=H,
                        people=counts, poses=poses)
    os.replace(tmp, path)
    return path


# --- Loading -------------------------------------------------------------------

def read_meta(video_path):
    """The meta dict of a video's analysis file, or None if there isn't a usable one."""
    path = analysis_path(video_path)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as data:
            return json.loads(str(data["meta"]))
    except (OSError, ValueError, KeyError):
        return None


def load(job):
    """
    Fill a job's processing results from its analysis file: job.info (with the
    first frame read from the local video), job.frames, job.aruco_finish and
    job.analysis_meta. A file recording a processing failure sets
    job.status = STATUS_FAILED with its error.

    Raises:
        AnalysisFileError: no file, an unknown format, or the local video isn't
            the one that was analysed.
    """
    path = analysis_path(job.path)
    if not os.path.exists(path):
        raise AnalysisFileError("not processed yet (run process.py)")
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        if meta.get("format_version") != FORMAT_VERSION:
            raise AnalysisFileError(f"analysis file format {meta.get('format_version')} "
                                    f"not supported (expected {FORMAT_VERSION}); re-process")
        if (meta["video_size"] != os.path.getsize(job.path)
                or meta["video_fingerprint"] != fingerprint(job.path)):
            raise AnalysisFileError("this video differs from the one analysed; re-process it")
        job.analysis_meta = meta
        if meta["status"] == STATUS_FAILED:
            job.status, job.error = STATUS_FAILED, meta["error"]
            return
        times, H, counts, poses = data["times"], data["H"], data["people"], data["poses"]

    crop = tuple(meta["crop"]) if meta["crop"] else None
    cap = video_io.open_video(job.path, crop, meta["first_frame_idx"])
    ok, first_frame = cap.read()
    cap.release()
    if not ok:
        raise AnalysisFileError("can't read the video's first frame")
    if first_frame.shape[:2] != (meta["frame_h"], meta["frame_w"]):
        raise AnalysisFileError("frame size differs from the one analysed; re-process it")
    job.info = VideoInfo(meta["first_frame_idx"], crop, meta["fps"], meta["total_frames"], first_frame)
    job.aruco_finish = tuple(meta["aruco_finish"]) if meta["aruco_finish"] else None
    job.frames = [
        FrameResult(frame_idx=k, time_s=float(times[k]),
                    H=None if np.isnan(H[k]).any() else H[k],
                    people=[_pose(poses[k, p]) for p in range(counts[k])])
        for k in range(len(times))
    ]


def _pose(values):
    """One person's (L, 4) array -> a layout pose of Landmarks (None where NaN)."""
    return [None if np.isnan(v[0]) else pose_common.Landmark(float(v[0]), float(v[1]),
                                                             float(v[2]), float(v[3]))
            for v in values]
