"""
The review session: load each video's analysis file and interpret it, then,
with a window, let the user review the videos, save their outputs and write the
report; without one, save the automatic results unreviewed.

Used by review_videos.py (analysis files made elsewhere, e.g. on a cluster) and
label.py (processing and review in one go).
"""

import os

from tmwt.detection import analysis
from tmwt.core import analysis_file
from tmwt.core import data_export
from tmwt.core import report
from tmwt.session import review
from tmwt.session import review_progress
from tmwt.core.job import (REVIEW_APPROVED, REVIEW_REJECTED, REVIEW_UNREVIEWED, STATUS_FAILED,
                 STATUS_INCOMPLETE, STATUS_NEEDS_INPUT, STATUS_NO_BODY, STATUS_OK,
                 VideoJob)
from tmwt.ui.labeler_ui import (APPROVED_MARK, DONE, FAILED, NEEDS_INPUT, REJECTED_MARK, SAVED_MARK,
                        UNREVIEWED, WORKING, JumpTo, WindowClosed)
from tmwt.ui.widgets import GREEN, GREY, KEY_ENTER, KEY_ESC, ORANGE, RED, WHITE


def make_jobs(videos, output_dir, endpoint_behavior):
    """One VideoJob per video, with its CSV output path in `output_dir`."""
    return [VideoJob(path=v,
                     output_path=os.path.join(output_dir, os.path.splitext(os.path.basename(v))[0] + ".csv"),
                     name=os.path.basename(v),
                     endpoint_behavior=endpoint_behavior)
            for v in videos]


def load_jobs(jobs, ui=None):
    """
    Load every job's analysis file and interpret it (subject, endpoints,
    timing). A missing or unusable file marks the job failed with the reason.
    """
    for i, job in enumerate(jobs):
        print(f"\n{'=' * 60}\nLoading ({i + 1}/{len(jobs)}): {job.name}\n{'=' * 60}")
        if ui is not None:
            ui.active = i
            ui.show_progress("Loading analysis", f"{job.name} ({i + 1} of {len(jobs)})",
                             i / len(jobs), force=True)
        try:
            analysis_file.load(job)
            analysis.interpret(job)
        except analysis_file.AnalysisFileError as e:
            job.status, job.error = STATUS_FAILED, str(e)
            print(f"  {e}")
        if ui is not None:
            ui.set_state(i, *_analysis_state(job))


def run(jobs, ui, output_dir, review_first=True):
    """
    Review (if there's a window and review_first), save the outputs and write
    the report; with a window, finish on a summary screen.

    With a window, an unfinished earlier review of the folder can be continued
    (review_progress.py). After reviewing, the user confirms saving first; exiting
    without saving, or closing the window, keeps the review progress for next
    time.

    Raises:
        WindowClosed: the user closed the window (progress is saved first).
    """
    if ui is not None and review_first:
        try:
            if not _review(jobs, ui):
                print("\nReview progress saved; no outputs written yet. Run the review "
                      "again to continue where you left off (or start over).")
                return
        except WindowClosed:
            review_progress.save(jobs, ui.active)
            print("\nWindow closed. Your review progress is kept: run the review "
                  "again to continue where you left off, or start over.")
            raise
    save_outputs(jobs, ui)
    if ui is not None:
        review_progress.clear(jobs)
    report_path, _ = report.write_report(jobs, output_dir, _pose_models(jobs))
    if ui is not None:
        show_summary(ui, jobs, report_path)


def _review(jobs, ui):
    """
    The interactive part: continue or start the review, review, then confirm.
    Returns True to save the outputs.
    """
    progress = review_progress.load(jobs)
    if progress is not None and ask_resume(ui, progress):
        last = review_progress.restore(jobs, progress)
        _show_restored_states(jobs, ui)
        # Pick up the video that was open when the review stopped, if it's still
        # undecided; otherwise the next one to review.
        if last is not None and jobs[last].review == REVIEW_UNREVIEWED:
            first = last
        else:
            first = next_unreviewed(jobs, last if last is not None else -1)
        print(f"  Continuing the review saved {progress['saved']}.")
    else:
        review_progress.clear(jobs)
        first = ask_to_review(ui, jobs)
        if first is None:
            return True   # "Save all without reviewing"
    if first is not None and run_review(jobs, ui, first) == review.SAVE_AND_QUIT:
        return False
    return confirm_save(jobs, ui)


def ask_resume(ui, progress):
    """"Continue your previous review?" screen. Returns True to continue it."""
    approved, skipped, todo = review_progress.counts(progress)
    ui.active = None
    choice = ui.show_message([
        ("Continue your previous review?", WHITE),
        (f"Saved {progress['saved'].replace('T', ' at ')}: {approved} approved, "
         f"{skipped} skipped, {todo} still to review.", GREY),
        ("Start over discards those decisions.", GREY),
    ], [("Continue", "continue", KEY_ENTER + (KEY_ESC,)), ("Start over", "restart", ())])
    return choice == "continue"


def _show_restored_states(jobs, ui):
    """Update the sidebar for decisions and edits restored from saved progress."""
    for i, job in enumerate(jobs):
        if job.review == REVIEW_APPROVED:
            ui.set_state(i, DONE, f"approved  {job.duration:.2f}s")
            ui.mark_reviewed(i, APPROVED_MARK)
        elif job.review == REVIEW_REJECTED:
            ui.set_state(i, FAILED, f"rejected: {job.review_note}")
            ui.mark_reviewed(i, REJECTED_MARK)
        elif job.status != STATUS_FAILED and job.duration is not None:
            ui.set_state(i, DONE, f"{job.timing_source}  {job.duration:.2f}s")


def _pose_models(jobs):
    """The pose model(s) the analysis files were made with, for the report."""
    models = sorted({f"{m['backend']}: {m['model']} ({m.get('device', '?')})"
                     for m in (job.analysis_meta for job in jobs) if m.get("backend")})
    return ", ".join(models) or "unknown"


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
        (sum(j.status == STATUS_NO_BODY for j in jobs), "with no body detected", RED),
        (sum(j.status == STATUS_FAILED for j in jobs), "failed", RED),
    ]
    lines = [("Analysis complete", GREEN), (f"{len(jobs)} video(s) analysed", WHITE)]
    lines += [(f"{count} {text}", color) for count, text, color in counts if count]
    lines += [("Or click a video in the list to start reviewing there.", GREY)]
    ui.review_targets = set(reviewable(jobs))
    try:
        choice = ui.show_message(lines, [("Start review", "review", KEY_ENTER),
                                         ("Save all without reviewing", "skip", ())])
    except JumpTo as jump:
        return jump.index
    finally:
        ui.review_targets = set()
    return next_unreviewed(jobs, -1) if choice == "review" else None


def confirm_save(jobs, ui):
    """
    "Review complete" screen: what will be saved, with Save results and Save
    progress and quit (no outputs written; the review can be continued later).
    Clicking a video in the sidebar reviews just that video, then returns here.

    Returns:
        True to save.
    """
    while True:
        counts = {}
        for job in jobs:
            result = report.job_result(job)[0]
            counts[result] = counts.get(result, 0) + 1
        lines = [("Review complete", GREEN)]
        for result, text, color in (
                (report.APPROVED, "approved", GREEN),
                (report.REJECTED, "skipped (not saved)", RED),
                (report.UNREVIEWED, "not reviewed (saved with their automatic timing)", WHITE),
                (report.FAILED, "can't be saved (no timing or analysis)", RED)):
            if counts.get(result):
                lines.append((f"{counts[result]} {text}", color))
        lines.append(("Click a video in the list to change it before saving.", GREY))

        ui.active = None
        ui.review_targets = set(reviewable(jobs))
        try:
            choice = ui.show_message(lines, [("Save results", "save", KEY_ENTER),
                                             ("Save progress and quit", "exit", (KEY_ESC,))])
        except JumpTo as jump:
            if run_review(jobs, ui, jump.index, only_one=True) == review.SAVE_AND_QUIT:
                return False
            continue
        finally:
            ui.review_targets = set()
        return choice == "save"


def run_review(jobs, ui, first, only_one=False):
    """
    Review jobs, starting with index `first`, until every reviewable job has
    been approved or rejected, or the user quits.

    After each decision the next unreviewed job follows (wrapping round to
    earlier ones that were skipped); jobs already decided are never revisited
    automatically. Clicking a file in the sidebar at any point leaves the
    current review undecided and switches to that file — including one already
    reviewed, whose result the new review replaces.

    With only_one, stop after the first decision (used when changing one video
    from the "Review complete" screen).

    Returns:
        review.QUIT or review.SAVE_AND_QUIT if the user stopped the review,
        else None.
    """
    ui.review_targets = set(reviewable(jobs))
    current = first
    try:
        while current is not None:
            job = jobs[current]
            state_before = (ui.states[current], ui.notes[current])
            # A video already approved or skipped opens at the options, not a replay.
            at_menu = job.review != REVIEW_UNREVIEWED
            try:
                result = review.review_job(job, ui, current, start_at_menu=at_menu)
            except JumpTo as jump:
                ui.set_state(current, *state_before)   # left undecided (or as before)
                review_progress.save(jobs, current)
                print(f"  Switching to {jobs[jump.index].name}")
                current = jump.index
                continue
            review_progress.save(jobs, current)       # after every decision
            if result in (review.QUIT, review.SAVE_AND_QUIT):
                print("  Review stopped by user.")
                return result
            current = None if only_one else next_unreviewed(jobs, current)
    finally:
        ui.review_targets = set()


def save_outputs(jobs, ui):
    """
    Phase 3: write the outputs of every job that has endpoints and wasn't
    rejected — approved ones and, if review was skipped or stopped early, the
    automatic results of the rest (marked unreviewed).
    """
    to_save = [(i, job) for i, job in enumerate(jobs)
               if job.frames and job.review != REVIEW_REJECTED
               and (job.far_ep is not None or job.review == REVIEW_APPROVED)]
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
            ui.mark_reviewed(i, SAVED_MARK)   # the check turns green once it's on disk
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
