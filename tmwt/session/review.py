"""
Phase 2 of a run: review each analysed video with the user.

Each video plays back in real time with its detection drawn on, then pauses on
a prompt (LabelerUI.ask_review):
  - Looks good               -> approved (outputs are written after review)
  - Rope endpoints inaccurate -> re-place the endpoints; timing is recomputed
                                from the cached analysis and the video replays
  - Walk start/stop inaccurate -> replay, marking the start and stop with a button
Playback has a seek bar and frame-step buttons, so any moment can be checked.
  - Wrong person tracked     -> (only with several people) click the walker;
                                timing is recomputed and the video replays
  - Skip this file           -> the file is rejected; nothing is saved
A video in which nobody was detected at all only offers "Skip this file".
Videos whose endpoints couldn't be found automatically ask for clicks first,
with the start point pre-placed where the subject was detected standing. If new
endpoints don't give a complete timing, the user is asked straight away to redo
them, time the video manually, or skip it.
"""

import cv2

from tmwt.ui import annotate
from tmwt.detection import people
from tmwt.measurement import timing
from tmwt.core import video_io
from tmwt.core.job import COURSE_M, REVIEW_APPROVED, REVIEW_REJECTED, STATUS_NO_BODY
from tmwt.ui.labeler_ui import APPROVED_MARK, DONE, FAILED, REJECTED_MARK, WORKING
from tmwt.ui.player import Player
from tmwt.ui.widgets import GREEN, GREY, KEY_ENTER, KEY_ESC, KEY_SPACE, ORANGE, RED, WHITE

# review_job results: finish the review now (go to "Review complete"), or stop
# and come back later (progress is saved, no outputs are written).
QUIT = "quit"
SAVE_AND_QUIT = "save_quit"
# Outcomes of setting endpoints (_set_endpoints).
_REPLAY = "replay"        # complete timing found: play the video with it
_PROMPT = "prompt"        # back to the review prompt without replaying
_CANCELLED = "cancelled"  # the user cancelled endpoint picking
_SKIP = "skip"            # the user chose to skip the file


def review_job(job, ui, i, start_at_menu=False):
    """
    Review one analysed job (index i in the batch) in the window. With
    start_at_menu (a video already reviewed), open at the options instead of
    replaying it first.

    Returns:
        QUIT (finish the review now), SAVE_AND_QUIT (stop, to continue later),
        or None once this video is decided.
    """
    print(f"\n  Reviewing {job.name}")
    ui.active = i
    ui.set_state(i, WORKING)

    if job.status == STATUS_NO_BODY:
        ui.show_message([
            ("No body detected", ORANGE),
            (f"{job.name}: no person was found in any frame of this video.", GREY),
        ], [("Skip this file", "skip", KEY_ENTER + (KEY_ESC,))], background=job.info.first_frame)
        _reject(job, ui, i, "no body detected")
        return None

    note = None
    replay = True
    if job.far_ep is None or job.near_ep is None:
        reason = f"{job.name}: automatic detection failed ({job.endpoint_problem})"
        outcome, note = _set_endpoints(job, ui, reason)
        if outcome == _CANCELLED:
            # Back to the options, where the user can retry, time it, or skip it.
            note = "Rope endpoints not set: set them (2), time it manually (3), or skip it (4)."
        elif outcome == _SKIP:
            _reject(job, ui, i, "no walk timing found")
            return None
        replay = outcome == _REPLAY

    several_people = people.people_on_screen(job)[0] is not None
    last_frame = job.info.first_frame
    if start_at_menu:
        replay = False
    while True:
        if replay:
            _, _, frame = playback(job, ui)
            last_frame = frame if frame is not None else last_frame
        replay = True

        choice = ui.ask_review(last_frame, summary_lines(job), note, wrong_person=several_people)
        note = None
        if choice == "approve":
            if job.duration is None:
                note = "Timing is incomplete: time it manually (3), or skip the file (4)."
                replay = False
                continue
            job.review = REVIEW_APPROVED
            ui.set_state(i, DONE, f"approved  {job.duration:.2f}s")
            ui.mark_reviewed(i, APPROVED_MARK)
            return None
        if choice == "endpoints":
            outcome, note = _set_endpoints(job, ui)
            if outcome == _SKIP:
                _reject(job, ui, i, "no walk timing found")
                return None
            replay = outcome == _REPLAY
        elif choice == "person":
            replay = _pick_subject(job, ui)
        elif choice == "timing":
            note = _time_manually(job, ui, last_frame)
            replay = False
        elif choice == "skip":
            _reject(job, ui, i, "skipped at review")
            return None
        elif choice == "quit":
            return QUIT
        elif choice == "save_quit":
            return SAVE_AND_QUIT
        # "replay" loops round and plays again.


def _time_manually(job, ui, background):
    """
    Replay the video for the user to mark the start and stop. Updates the job's
    timing if both were marked. Returns a note for the review prompt, or None.
    """
    choice = ui.show_message([
        ("Manual timing", WHITE),
        ("The video will replay in real time.", GREY),
        ("Click Mark start when the walk starts, then Mark stop when it ends.", GREY),
        ("(Space works too.)", GREY),
    ], [("Start manual timing", "start", KEY_ENTER), ("Cancel", "cancel", (KEY_ESC,))],
        background=background)
    if choice == "cancel":
        return None
    start, end, _ = playback(job, ui, manual_timing=True)
    if start is None or end is None:
        return "Manual timing needs both a start and a stop mark. Timing unchanged."
    job.walk_start, job.walk_end = start, end
    job.timing_source, job.timing_detail = "manual", "marked during replay"
    return None


def playback(job, ui, manual_timing=False):
    """
    Play the job back in real time with its detection drawn on. Below the
    picture: a seek bar (drag or click to scrub; the walk's start and end are
    marked), frame-back / play-pause / frame-forward buttons (player.py), Skip to
    review, and in manual timing Mark start then Mark stop, and Done.

    Frames come from a video_io.FrameSource, so scrubbing and stepping back show
    exactly the frames the analysis used. Marks use the frame on screen when the
    button was pressed (or Space was hit), not when it was released.

    Returns:
        (start, end, last_frame): the timing shown (the marks, in manual
        timing), and the last annotated frame shown (or None).
    """
    if manual_timing:
        start = end = None
        waiting = annotate.WAITING_MANUAL
    else:
        start, end = job.walk_start, job.walk_end
        waiting = annotate.WAITING_AUTO
    label = "MANUAL TIMING" if manual_timing else None
    # In manual timing Space marks, so P pauses.
    player = Player([f.time_s for f in job.frames],
                    pause_keys=(ord("p"), ord("P")) if manual_timing else (KEY_SPACE,))
    source = video_io.FrameSource(job)
    ui.start_playback()
    last = None
    try:
        while True:
            frame = source.get(player.k)
            if frame is None:
                break
            last, _ = annotate.render_frame(frame, job.frames[player.k], start, end, waiting)
            specs = player.transport() + [
                ("Done (Enter)" if manual_timing else "Skip to review (Enter)", "skip", KEY_ENTER)]
            if manual_timing and end is None:
                specs.insert(0, ("Mark start (Space)" if start is None else "Mark stop (Space)",
                                 "mark", (KEY_SPACE,)))
            value, pressed_at = player.show(ui, last, specs, label, marks=[(start, GREEN), (end, RED)])

            if value == "skip":
                break
            if value == "mark":
                t = player.time_on_screen(pressed_at)
                if start is None:
                    start = t
                    print(f"  Walk STARTED (manual) at {start:.3f}s")
                elif t > start:
                    end = t
                    print(f"  Walk FINISHED (manual) at {end:.3f}s")
            if not player.advance():
                break
    finally:
        source.close()
    return start, end, last


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


def _set_endpoints(job, ui, reason=None):
    """
    Let the user (re)place the endpoints, with the start point pre-placed at the
    current start (or where the subject was detected standing), then recompute
    the timing. If that timing is incomplete, ask straight away whether to redo
    the endpoints, time the video manually, or skip it.

    Returns:
        (outcome, note): outcome is _REPLAY, _PROMPT, _CANCELLED or _SKIP; note
        is a message for the review prompt, or None.
    """
    start = job.far_ep or job.subject_start
    finish = job.near_ep
    while True:
        picked = ui.pick_endpoints(job.info.first_frame, reason, start, finish)
        if picked is None:
            return _CANCELLED, None
        start, finish, moved = picked
        job.far_ep, job.near_ep = start, finish
        # Unmoved start at the detected standing spot keeps the standing-start
        # timing; a start the user placed is a start line.
        job.far_ep_is_standing_spot = not moved and start == job.subject_start
        job.endpoint_source = "auto start, manual finish" if job.far_ep_is_standing_spot else "manual"
        print(f"  Endpoints: start {start} ({'detected' if not moved else 'clicked'}), finish {finish}")
        timing.update_timing(job)
        if job.duration is not None:
            return _REPLAY, None

        hint = ([("If the subject was already walking when the video starts, use", GREY),
                 ("Redo endpoints > Move start point and click the start line.", GREY)]
                if job.far_ep_is_standing_spot else [])
        choice = ui.show_message([
            ("No walk start / end found", ORANGE),
            (summary_lines(job)[1], GREY),
            ("Redo the endpoints, or time this video manually.", GREY),
        ] + hint, [("Redo endpoints", "redo", ()), ("Time manually", "timing", KEY_ENTER),
            ("Skip this file", "skip", (KEY_ESC,))], background=job.info.first_frame)
        if choice == "skip":
            return _SKIP, None
        if choice == "timing":
            return _PROMPT, _time_manually(job, ui, job.info.first_frame)
        reason = None


def _pick_subject(job, ui):
    """
    Show a frame with the tracked people and let the user click the walker, then
    recompute the timing for that person. Returns False if nothing changed.
    """
    frame_idx, present = people.people_on_screen(job)
    if frame_idx is None:
        return False
    cap = job.open_capture()
    cap.set(cv2.CAP_PROP_POS_FRAMES, job.info.first_frame_idx + frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    k = ui.pick_person(frame, [pose for _, pose in present],
                       f"{job.name}: {len(present)} people in view")
    if k is None:
        return False
    people.set_subject(job, present[k][0])
    job.subject_start = people.subject_start(job)
    if job.far_ep_is_standing_spot and job.subject_start is not None:
        job.far_ep = job.subject_start   # the start point is where this person stood
    print(f"  Subject changed to person {k + 1} of {len(present)}")
    timing.update_timing(job)
    return True


def _reject(job, ui, i, reason):
    job.review, job.review_note = REVIEW_REJECTED, reason
    ui.set_state(i, FAILED, f"rejected: {reason}")
    ui.mark_reviewed(i, REJECTED_MARK)
    print(f"  Rejected ({reason}); no outputs written.")
