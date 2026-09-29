"""
Phase 2 of a run: review each analysed video with the user.

Each video plays back in real time with its detection drawn on, then pauses on
a prompt (LabelerUI.ask_review):
  - Looks good               -> outputs are saved
  - Rope endpoints inaccurate -> click new endpoints; timing is recomputed from
                                the cached analysis and the video replays
  - Walk start/stop inaccurate -> replay with SPACE marking the start and stop
  - Body not detected        -> the file is rejected; nothing is saved
Videos whose endpoints couldn't be found automatically ask for clicks first.
"""

import annotate
import data_export
import timing
import video_io
from job import COURSE_M, REVIEW_APPROVED, REVIEW_REJECTED
from labeler_ui import DONE, FAILED, GREY, KEY_ENTER, KEY_ESC, WHITE, WORKING

QUIT = "quit"


def review_job(job, ui, i, n):
    """
    Review one analysed job (index i of n) in the window.

    Returns:
        QUIT if the user asked to stop reviewing, else None.
    """
    print(f"\n  Reviewing {job.name}")
    header = f"Reviewing {job.name} ({i + 1} of {n})"
    ui.active = i
    ui.set_state(i, WORKING)

    if job.far_ep is None:
        reason = f"Automatic detection failed: {job.endpoint_problem}."
        if not _set_manual_endpoints(job, ui, header, reason):
            _reject(job, ui, i, "rope endpoints not set")
            return None

    last_frame = job.info.first_frame
    note = None
    replay = True
    while True:
        if replay:
            result, _, _, frame = playback(job, ui, header)
            last_frame = frame if frame is not None else last_frame
            if result == QUIT:
                return QUIT
        replay = True

        choice = ui.ask_review(last_frame, summary_lines(job), note)
        note = None
        if choice == "approve":
            if job.duration is None:
                note = "Timing is incomplete: press 3 to time it manually, or 4 to skip."
                replay = False
                continue
            job.review = REVIEW_APPROVED
            save_with_progress(job, ui, i)
            ui.set_state(i, DONE, f"approved  {job.duration:.2f}s")
            return None
        if choice == "endpoints":
            replay = _set_manual_endpoints(job, ui, header, previous=(job.far_ep, job.near_ep))
        elif choice == "timing":
            ui.show_message([
                ("Manual timing", WHITE),
                ("The video will replay in real time.", GREY),
                ("Press SPACE when the walk starts, and again when it ends.", GREY),
                ("Press any key to begin.", GREY),
            ], background=last_frame)
            result, start, end, frame = playback(job, ui, header + "  -  MANUAL TIMING",
                                                 manual_timing=True)
            last_frame = frame if frame is not None else last_frame
            if result == QUIT:
                return QUIT
            if start is not None and end is not None:
                job.walk_start, job.walk_end = start, end
                job.timing_source, job.timing_detail = "manual", "spacebar"
            else:
                note = "Manual timing needs both a start and a stop press. Timing unchanged."
            replay = False
        elif choice == "body":
            _reject(job, ui, i, "body not detected")
            return None
        elif choice == "quit":
            return QUIT
        # "replay" loops round and plays again.


def playback(job, ui, header, manual_timing=False):
    """
    Play the job back in real time with its detection drawn on.

    Normal playback: SPACE pauses, ENTER skips to the review prompt.
    Manual timing: SPACE marks the start and then the stop, P pauses.
    Esc quits in both.

    Returns:
        (result, start, end, last_frame): result is QUIT or "done"; start/end
        are the timing shown (the SPACE marks, in manual timing); last_frame is
        the last annotated frame shown, or None.
    """
    if manual_timing:
        start = end = None
        controls = "space = mark start/stop | p = pause"
        waiting = annotate.WAITING_SPACEBAR
        pause_key = ord("p")
    else:
        start, end = job.walk_start, job.walk_end
        controls = "space = pause | enter = review"
        waiting = annotate.WAITING_AUTO
        pause_key = ord(" ")

    cap = job.open_capture()
    last = None
    clock = video_io.PlaybackClock()
    result = "done"
    try:
        for f in job.frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            last, _ = annotate.render_frame(frame_bgr, f, start, end, waiting, controls)
            key = ui.show_frame(last, clock.ms_until(f.time_s), header)

            if key == pause_key:
                key = _pause(ui, last, header, pause_key)
                clock.restart()
            if key == KEY_ESC:
                result = QUIT
                break
            if not manual_timing and key in KEY_ENTER:
                break
            if manual_timing and key == ord(" "):
                if start is None:
                    start = f.time_s
                    print(f"  Walk STARTED (manual) at {start:.3f}s")
                elif end is None and f.time_s > start:
                    end = f.time_s
                    print(f"  Walk FINISHED (manual) at {end:.3f}s")
    finally:
        cap.release()
    return result, start, end, last


def _pause(ui, frame, header, pause_key):
    """Hold on `frame` until the pause key or Esc; returns that key."""
    while True:
        key = ui.show_frame(frame, 50, header + "   [PAUSED]")
        if key in (pause_key, KEY_ESC):
            return key


def summary_lines(job):
    """Two lines describing the job's endpoints and timing, for the review prompt."""
    lines = [f"Endpoints: {job.endpoint_source or 'none'}    "
             f"Timing: {job.timing_source or 'none'}"
             + (f" ({job.timing_detail})" if job.timing_detail else "")]
    if job.duration is not None:
        lines.append(f"Start {job.walk_start:.2f}s    End {job.walk_end:.2f}s    "
                     f"Duration {job.duration:.2f}s ({COURSE_M / job.duration:.2f} m/s)")
    else:
        start = f"{job.walk_start:.2f}s" if job.walk_start is not None else "not found"
        end = f"{job.walk_end:.2f}s" if job.walk_end is not None else "not found"
        lines.append(f"Timing incomplete: start {start}, end {end}")
    return lines


def save_with_progress(job, ui, i):
    """Save the job's outputs while showing a progress bar."""
    ui.set_state(i, WORKING, "saving...")
    data_export.save_job(job, on_progress=lambda frac: ui.show_progress(
        f"Saving {job.name}", "Writing CSV, timing and annotated videos", frac))


def _set_manual_endpoints(job, ui, header, reason=None, previous=None):
    """
    Ask for new endpoints in the window, then recompute the timing.
    Returns False if the user cancelled.
    """
    picked = ui.pick_endpoints(job.info.first_frame, header, reason, previous)
    if picked is None:
        return False
    job.far_ep, job.near_ep = picked
    job.endpoint_source = "manual"
    job.far_ep_is_standing_spot = False  # a clicked far endpoint is a start line
    print(f"  Manual endpoints: far {job.far_ep}, near {job.near_ep}")
    timing.update_timing(job)
    return True


def _reject(job, ui, i, reason):
    job.review, job.review_note = REVIEW_REJECTED, reason
    ui.set_state(i, FAILED, f"rejected: {reason}")
    print(f"  Rejected ({reason}); no outputs written.")
