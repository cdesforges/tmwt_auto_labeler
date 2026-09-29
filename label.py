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
import people
import pose_common
import review
from job import (END_BEHAVIORS, END_FIRST_FOOT, REVIEW_APPROVED, REVIEW_REJECTED,
                 REVIEW_UNREVIEWED, STATUS_FAILED, STATUS_INCOMPLETE, STATUS_NEEDS_INPUT,
                 STATUS_OK, VideoJob)
from labeler_ui import (DONE, FAILED, GREEN, GREY, KEY_ENTER, KEY_ESC, NEEDS_INPUT,
                        ORANGE, RED, UNREVIEWED, WHITE, WORKING, JumpTo, LabelerUI)
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


def make_jobs(videos, output_dir, end_behavior):
    """One VideoJob per video, with its CSV output path in `output_dir`."""
    return [VideoJob(path=v,
                     output_path=os.path.join(output_dir, os.path.splitext(os.path.basename(v))[0] + ".csv"),
                     name=os.path.basename(v),
                     end_behavior=end_behavior)
            for v in videos]


# --- Phase 1 -------------------------------------------------------------------

def load_pose_model(ui, backend, backend_name, model_path):
    """
    Load the pose model before analysis starts, showing a message meanwhile
    (it takes several seconds; rtmlib and mmpose then reuse it for every video).
    """
    print(f"Loading pose model ({backend_name}: {model_path})...")
    if ui is not None:
        ui.show_status("Loading pose model", [f"{backend_name}: {model_path}",
                                              "This takes a few seconds."])
    backend.create_landmarker(model_path, num_poses=people.MAX_PEOPLE).close()


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

        def console(fraction, frame_bgr, poses):
            if fraction >= next_report[0]:
                print(f"    {int(next_report[0] * 100):3d}%")
                next_report[0] += 0.1
        return console

    def window(fraction, frame_bgr, poses):
        ui.notes[i] = f"analysing {fraction * 100:.0f}%"

        def preview():
            img = frame_bgr.copy()
            for pose in poses:
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
        what = "finish point" if job.subject_start is not None else "endpoints"
        return NEEDS_INPUT, f"needs {what}: {job.endpoint_problem}"
    if job.status == STATUS_INCOMPLETE:
        return FAILED, review.summary_lines(job)[1]
    return FAILED, job.error


# --- Phases 2 to 4 -------------------------------------------------------------

def reviewable(jobs):
    """Indices of the jobs that can be reviewed (analysis didn't fail outright)."""
    return [i for i, job in enumerate(jobs) if job.status != STATUS_FAILED]


def next_unreviewed(jobs, after):
    """
    The next reviewable job after index `after` that hasn't been approved or
    rejected yet, wrapping round to the start; None when every one is decided.
    """
    pending = [i for i in reviewable(jobs) if jobs[i].review == REVIEW_UNREVIEWED]
    later = [i for i in pending if i > after]
    return (later or pending or [None])[0]


def ask_to_review(ui, jobs):
    """
    "Analysis complete" screen with what was found.

    Returns:
        The index of the job to start reviewing — the first one, or whichever
        the user clicked in the sidebar — or None to save everything without
        reviewing.
    """
    ui.active = None
    counts = [
        (sum(j.status == STATUS_OK for j in jobs), "timed automatically", GREEN),
        (sum(j.status == STATUS_NEEDS_INPUT for j in jobs), "need the finish point clicked", ORANGE),
        (sum(j.status == STATUS_INCOMPLETE for j in jobs), "with incomplete timing", RED),
        (sum(j.status == STATUS_FAILED for j in jobs), "failed", RED),
    ]
    lines = [("Analysis complete", GREEN), (f"{len(jobs)} video(s) analysed", WHITE)]
    lines += [(f"{count} {text}", color) for count, text, color in counts if count]
    lines += [("Or click a video in the list to start reviewing there.", GREY)]
    ui.review_targets = set(reviewable(jobs))
    try:
        choice = ui.show_message(lines, [("Start review", "review", KEY_ENTER),
                                         ("Save all without reviewing", "skip", (KEY_ESC,))])
    except JumpTo as jump:
        return jump.index
    finally:
        ui.review_targets = set()
    return next_unreviewed(jobs, -1) if choice == "review" else None


def run_review(jobs, ui, first):
    """
    Review jobs, starting with index `first`, until every reviewable job has
    been approved or rejected, or the user quits.

    After each decision the next unreviewed job follows (wrapping round to
    earlier ones that were skipped); jobs already decided are never revisited
    automatically. Clicking a file in the sidebar at any point leaves the
    current review undecided and switches to that file — including one already
    reviewed, whose result the new review replaces.
    """
    ui.review_targets = set(reviewable(jobs))
    current = first
    try:
        while current is not None:
            job = jobs[current]
            state_before = (ui.states[current], ui.notes[current])
            try:
                if review.review_job(job, ui, current) == review.QUIT:
                    print("  Review stopped by user.")
                    return
            except JumpTo as jump:
                ui.set_state(current, *state_before)   # left undecided (or as before)
                print(f"  Switching to {jobs[jump.index].name}")
                current = jump.index
                continue
            current = next_unreviewed(jobs, current)
    finally:
        ui.review_targets = set()


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
    parser.add_argument("--endpoint_behavior", choices=END_BEHAVIORS, default=END_FIRST_FOOT,
                        help="What ends the walk at the finish line: the first foot to cross "
                             "it (first_foot, default) or the midpoint of the two ankles "
                             "(ankle_midpoint).")
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

    jobs = make_jobs(videos, output_dir, args.endpoint_behavior)
    ui = None if args.no_display else LabelerUI([job.name for job in jobs])
    load_pose_model(ui, backend, args.backend, model_path)
    cancelled = run_analysis(jobs, ui, model_path, backend, not args.no_matte_crop)
    if ui is not None and not cancelled:
        first = ask_to_review(ui, jobs)
        if first is not None:
            run_review(jobs, ui, first)
    save_outputs(jobs, ui)
    report_path, _ = report.write_report(jobs, output_dir, args.backend)
    if ui is not None:
        show_summary(ui, jobs, report_path)
        ui.close()
    print(f"\nAll done! Output files are in '{output_dir}'.")


if __name__ == "__main__":
    main()
