"""
Processing a folder of videos: the slow part of the analysis
(analysis.process_video) for each video, written to one analysis file per video
(analysis_file.py).

Used by process_videos.py (e.g. on a cluster: console output only) and by
label.py (locally, with progress in the window). Videos that already have a
usable analysis file made with the same settings are skipped, so an interrupted
run can simply be started again.
"""

import os
import traceback

import cv2
import numpy as np

from tmwt.detection import analysis
from tmwt.core import analysis_file
from tmwt.detection import people
from tmwt.pose import pose_common
from tmwt.core.job import STATUS_FAILED, VideoJob
from tmwt.ui.labeler_ui import DONE, FAILED, WORKING, WindowClosed

# Video file extensions to look for.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


class Cancelled(Exception):
    """The user cancelled processing from the window."""


def find_videos(input_dir):
    """Sorted paths of the video files directly inside `input_dir`."""
    return sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)


def settings(backend_name, model_path, matte_crop):
    """The processing settings recorded in (and compared against) analysis files."""
    return {"backend": backend_name, "model": model_path, "matte_crop": matte_crop,
            "max_people": people.MAX_PEOPLE}


def needs_processing(video_path, wanted):
    """
    Whether a video needs (re)processing with the `wanted` settings.

    Returns:
        (needed, reason): reason says why, or is "" when the file can be reused.
    """
    meta = analysis_file.read_meta(video_path)
    if meta is None:
        return True, "not processed yet"
    if meta.get("format_version") != analysis_file.FORMAT_VERSION:
        return True, "older analysis file format"
    if meta.get("status") == STATUS_FAILED:
        return True, f"previous attempt failed ({meta.get('error')})"
    if any(meta.get(k) != v for k, v in wanted.items()):
        return True, "processed with different settings"
    if (meta.get("video_size") != os.path.getsize(video_path)
            or meta.get("video_fingerprint") != analysis_file.fingerprint(video_path)):
        return True, "video has changed"
    return False, ""


def load_pose_model(backend, backend_name, model_path, ui=None):
    """
    Load the pose model up front, showing a message meanwhile: it takes several
    seconds (downloading it first if needed), and rtmlib and mmpose then reuse
    it for every video.
    """
    print(f"Loading pose model ({backend_name}: {model_path})...")
    if ui is not None:
        ui.show_status("Loading pose model", [f"{backend_name}: {model_path}",
                                              "This takes a few seconds."])
    backend.create_landmarker(model_path, num_poses=people.MAX_PEOPLE).close()


def process_all(videos, backend, backend_name, model_path, matte_crop=True,
                reprocess=False, ui=None):
    """
    Process every video that needs it and write its analysis file.

    Args:
        videos: video paths.
        backend: pose backend module; backend_name / model_path identify it.
        matte_crop: auto-detect and crop letterbox / pillarbox bars.
        reprocess: process even videos whose analysis file could be reused.
        ui: optional LabelerUI for progress (its files are `videos`, in order).

    Returns:
        True if the user cancelled part-way.
    """
    wanted = settings(backend_name, model_path, matte_crop)
    todo = []
    for i, video in enumerate(videos):
        needed, why = needs_processing(video, wanted)
        if needed or reprocess:
            todo.append(i)
            print(f"  {os.path.basename(video)}: {why or 'reprocessing'}")
        elif ui is not None:
            ui.set_state(i, DONE, "already processed")
    if not todo:
        print("All videos already processed.")
        return False

    load_pose_model(backend, backend_name, model_path, ui)
    provenance = dict(wanted, **backend.provenance())
    provenance["versions"].update({"opencv": cv2.__version__, "numpy": np.__version__})

    for n, i in enumerate(todo):
        job = VideoJob(path=videos[i], output_path="", name=os.path.basename(videos[i]))
        print(f"\n{'=' * 60}\nProcessing ({n + 1}/{len(todo)}): {job.name}\n{'=' * 60}")
        title, subtitle = f"Analysing {job.name}", f"{n + 1} of {len(todo)}"
        if ui is not None:
            ui.active = i
            ui.set_state(i, WORKING, "starting...")
            ui.show_progress(title, subtitle, 0.0, force=True, cancellable=True)
        try:
            analysis.process_video(job, model_path, backend, matte_crop,
                                   _progress_callback(ui, i, title, subtitle))
        except Cancelled:
            print("  Processing cancelled by user.")
            for k in todo[n:]:
                if ui is not None:
                    ui.set_state(k, FAILED, "not processed (cancelled)")
            return True
        except WindowClosed:
            raise
        except Exception as e:
            traceback.print_exc()
            job.status, job.error = STATUS_FAILED, f"error: {e}"

        path = analysis_file.save(job, provenance)
        if job.status == STATUS_FAILED:
            print(f"  FAILED: {job.error}")
        else:
            print(f"  Saved {len(job.frames)} frames to {path}")
        if ui is not None:
            ui.set_state(i, *((FAILED, job.error) if job.status == STATUS_FAILED
                              else (DONE, "processed")))
    return False


def _progress_callback(ui, i, title, subtitle):
    """Per-frame progress: window progress bar, or console every 10%."""
    if ui is None:
        next_percent = [10]   # whole percentages, so steps don't drift (0.1 * 9 != 0.9)

        def console(fraction, frame_bgr, poses):
            while fraction * 100 >= next_percent[0] and next_percent[0] <= 100:
                print(f"    {next_percent[0]:3d}%", flush=True)
                next_percent[0] += 10
        return console

    def window(fraction, frame_bgr, poses):
        ui.notes[i] = f"analysing {fraction * 100:.0f}%"

        def preview():
            img = frame_bgr.copy()
            for pose in poses:
                pose_common.draw_pose(img, pose)
            return img
        if ui.show_progress(title, subtitle, fraction, preview, cancellable=True):
            raise Cancelled()
    return window
