"""
TMWT Labeler — times 10 m walk test videos from pose tracking.

Processes every video in an input directory in four phases, in one window
(labeler_ui.py) that lists every file colour-coded by state:

  1. Analyse (analysis.py) — unattended. Per video: find the rope endpoints,
     run pose estimation + ground tracking over every frame, and decide the
     walk start and end in hindsight (timing.py).
  2. Review (review.py) — each video plays back in real time and pauses for
     the user to approve it, fix the endpoints, time it manually, or reject it.
     Nothing is written yet, so there's no wait between videos.
  3. Save (data_export.py) — write the outputs of every video that wasn't
     rejected, all in one go.
  4. Report (report.py) — labeling_report.csv / .md summarise every video.

--no_display skips the window and the review: automatic results are saved
unreviewed, and videos that need clicks are reported as failed.

Usage:
    python label.py --input_dir <dir> [--output_dir <dir>]
                    [--backend {mediapipe,mmpose,rtmlib}] [--model <path_or_alias>]
                    [--no_matte_crop] [--no_display]

Outputs are described in data_export.py and report.py.
"""

import argparse
import os
import sys
import traceback

import analysis
import data_export
import pose_common
import review
from job import (REVIEW_APPROVED, REVIEW_REJECTED, STATUS_FAILED, STATUS_INCOMPLETE,
                 STATUS_NEEDS_INPUT, STATUS_OK, VideoJob)
from labeler_ui import (DONE, FAILED, GREEN, GREY, KEY_ENTER, KEY_ESC, NEEDS_INPUT,
                        ORANGE, RED, UNREVIEWED, WHITE, WORKING, LabelerUI)
from pose_backend import BACKENDS, get_backend
import report

# Video file extensions to look for.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


class QuitRequested(Exception):
    """Raised from the progress callback when the user presses Esc during analysis."""


def find_videos(input_dir):
    """Sorted paths of the video files directly inside `input_dir`."""
    return sorted(os.path.join(input_dir, f) for f in os.listdir(input_dir)
                  if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS)


def make_jobs(videos, output_dir):
    """One VideoJob per video, with its CSV output path in `output_dir`."""
    return [VideoJob(path=v,
                     output_path=os.path.join(output_dir, os.path.splitext(os.path.basename(v))[0] + ".csv"),
                     name=os.path.basename(v))
            for v in videos]


# --- Phase 1 -------------------------------------------------------------------

def run_analysis(jobs, ui, model_path, backend, matte_crop):
    """
    Analyse every job in turn, showing progress in the window (or the console
    when ui is None). Returns True if the user cancelled part-way.
    """
    n = len(jobs)
    for i, job in enumerate(jobs):
        print(f"\n{'=' * 60}\nAnalysing ({i + 1}/{n}): {job.name}\n{'=' * 60}")
        title, subtitle = f"Analysing {job.name}", f"{i + 1} of {n}"
        if ui is not None:
            ui.active = i
            ui.set_state(i, WORKING, "starting...")
            ui.show_progress(title, subtitle, 0.0, force=True, cancellable=True)
        try:
            analysis.analyze_job(job, model_path, backend, matte_crop,
                                 _progress_callback(ui, i, title, subtitle))
        except QuitRequested:
            print("  Analysis cancelled by user.")
            for k in range(i, n):
                jobs[k].frames = []
                jobs[k].status, jobs[k].error = STATUS_FAILED, "not analysed (run cancelled)"
                if ui is not None:
                    ui.set_state(k, FAILED, jobs[k].error)
            return True
        except Exception as e:
            traceback.print_exc()
            job.status, job.error = STATUS_FAILED, f"error: {e}"
        if ui is not None:
            ui.set_state(i, *_analysis_state(job))
    return False


def _progress_callback(ui, i, title, subtitle):
    """Per-frame analysis progress: window progress bar, or console every 10%."""
    if ui is None:
        next_report = [0.1]

        def console(fraction, frame_bgr, pose):
            if fraction >= next_report[0]:
                print(f"    {int(next_report[0] * 100):3d}%")
                next_report[0] += 0.1
        return console

    def window(fraction, frame_bgr, pose):
        ui.notes[i] = f"analysing {fraction * 100:.0f}%"

        def preview():
            img = frame_bgr.copy()
            if pose is not None:
                pose_common.draw_pose(img, pose)
            return img
        if ui.show_progress(title, subtitle, fraction, preview, cancellable=True):
            raise QuitRequested()
    return window


def _analysis_state(job):
    """(sidebar state, note) for a job after analysis."""
    if job.status == STATUS_OK:
        return DONE, f"auto  {job.duration:.2f}s"
    if job.status == STATUS_NEEDS_INPUT:
        return NEEDS_INPUT, f"needs endpoints: {job.endpoint_problem}"
    if job.status == STATUS_INCOMPLETE:
        return FAILED, review.summary_lines(job)[1]
    return FAILED, job.error


# --- Phases 2 to 4 -------------------------------------------------------------

def ask_to_review(ui, jobs):
    """
    "Analysis complete" screen with what was found. Returns True if the user
    chose to review, False to save everything without reviewing.
    """
    ui.active = None
    counts = [
        (sum(j.status == STATUS_OK for j in jobs), "timed automatically", GREEN),
        (sum(j.status == STATUS_NEEDS_INPUT for j in jobs), "need rope endpoints clicked", ORANGE),
        (sum(j.status == STATUS_INCOMPLETE for j in jobs), "with incomplete timing", RED),
        (sum(j.status == STATUS_FAILED for j in jobs), "failed", RED),
    ]
    lines = [("Analysis complete", GREEN), (f"{len(jobs)} video(s) analysed", WHITE)]
    lines += [(f"{count} {text}", color) for count, text, color in counts if count]
    choice = ui.show_message(lines, [("Start review", "review", KEY_ENTER),
                                     ("Save all without reviewing", "skip", ())])
    return choice == "review"


def run_review(jobs, ui):
    """Review every job that has something to review, until the user quits."""
    for i, job in enumerate(jobs):
        if job.status == STATUS_FAILED:
            continue
        if review.review_job(job, ui, i) == review.QUIT:
            print("  Review stopped by user.")
            return


def save_outputs(jobs, ui):
    """
    Phase 3: write the outputs of every job that has endpoints and wasn't
    rejected — approved ones and, if review was skipped or stopped early, the
    automatic results of the rest (marked unreviewed).
    """
    to_save = [(i, job) for i, job in enumerate(jobs)
               if job.review != REVIEW_REJECTED and job.far_ep is not None and job.frames]
    for k, (i, job) in enumerate(to_save):
        approved = job.review == REVIEW_APPROVED
        print(f"\n  Saving ({k + 1}/{len(to_save)}): {job.name}"
              + ("" if approved else " (not reviewed)"))
        if ui is None:
            data_export.save_job(job)
            continue
        ui.active = i
        note = ui.notes[i]
        ui.set_state(i, WORKING, "saving...")
        data_export.save_job(job, on_progress=lambda frac: ui.show_progress(
            f"Saving {job.name}", f"{k + 1} of {len(to_save)}", frac))
        if approved:
            ui.set_state(i, DONE, note)
        else:
            ui.set_state(i, UNREVIEWED, "saved (not reviewed)")


def show_summary(ui, jobs, report_path):
    """Final screen: result counts and where the report is."""
    counts = {}
    for job in jobs:
        result = report.job_result(job)[0]
        counts[result] = counts.get(result, 0) + 1
    ui.active = None
    ui.show_message(
        [("All done", GREEN)]
        + [(f"{result}: {count}", RED if result in (report.REJECTED, report.FAILED) else WHITE)
           for result, count in counts.items()]
        + [(f"Report: {os.path.basename(report_path)} in the output folder", GREY)],
        [("Close", "close", KEY_ENTER + (KEY_ESC,))])


# --- CLI -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TMWT Labeler — time 10 m walk test videos from pose tracking.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing the video files to process.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for the outputs (default: <input_dir>/output).")
    parser.add_argument("--model", default=None,
                        help="Pose model. mediapipe: .task file path "
                             "(default: models/pose_landmarker_full.task); mmpose: config "
                             "path or alias (default: 'human'); rtmlib: 'balanced' | "
                             "'performance' | 'lightweight' (default: 'balanced').")
    parser.add_argument("--backend", choices=BACKENDS, default="mediapipe",
                        help="Pose backend to use (default: mediapipe).")
    parser.add_argument("--no_matte_crop", action="store_true",
                        help="Don't crop solid-colour mattes (letterbox / pillarbox bars).")
    parser.add_argument("--no_display", action="store_true",
                        help="Run unattended: no window and no review. Automatic results "
                             "are saved unreviewed; videos that need manual endpoints "
                             "are reported as failed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: '{args.input_dir}' is not a valid directory.")

    backend = get_backend(args.backend)
    model_path = args.model or backend.DEFAULT_MODEL_PATH
    print(f"Backend: {args.backend}  |  Model: {model_path}")

    output_dir = args.output_dir or os.path.join(args.input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    videos = find_videos(args.input_dir)
    if not videos:
        sys.exit(f"No video files found in '{args.input_dir}'.")
    print(f"Found {len(videos)} video(s) in '{args.input_dir}':")
    for v in videos:
        print(f"  - {os.path.basename(v)}")

    jobs = make_jobs(videos, output_dir)
    ui = None if args.no_display else LabelerUI([job.name for job in jobs])
    cancelled = run_analysis(jobs, ui, model_path, backend, not args.no_matte_crop)
    if ui is not None and not cancelled and ask_to_review(ui, jobs):
        run_review(jobs, ui)
    save_outputs(jobs, ui)
    report_path, _ = report.write_report(jobs, output_dir, args.backend)
    if ui is not None:
        show_summary(ui, jobs, report_path)
        ui.close()
    print(f"\nAll done! Output files are in '{output_dir}'.")


if __name__ == "__main__":
    main()
