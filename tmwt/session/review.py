"""
Phase 2 of a run: review each analysed video with the user.

Each video plays back (review_playback.py) with its detection drawn on: a seek bar with
the walk's start and stop marks (drag their tabs to move them), the mark
buttons (green dot: walk start, red dot: walk stop, M presses them in turn),
frame-step and play buttons, then:

  - Confirm (Enter) -> the final confirmation screen (_final_confirm): what
    will be saved, with Next video (the video is approved and the next one
    starts) or Go back (to the playback, where it was).
  - ☰ (Esc)         -> the menu (_menu):
        Change rope endpoints        re-place them; timing is recomputed and
                                     the video replays
        Mark this file to be skipped rejected; nothing is saved
        Review flagged points        (only with flagged pose points)
        Wrong person tracked         (only with several people) click the walker
        Back to review               back to the playback
        Finish all (save all results) go to "Review complete"

Marks changed in the playback replace the automatic timing (it becomes
manual). The video doesn't end the playback: it pauses on the last frame.

A video in which nobody was detected at all only offers "Skip this file".
Videos whose endpoints couldn't be found automatically ask for clicks first,
with the start point pre-placed where the subject was detected standing. If new
endpoints don't give a complete timing, the user is asked straight away to adjust
them, mark the timing in the playback, or skip the video.

If the pose check (pose_check.py) flagged points the heavier model couldn't
fix, opening the video shows a full-screen notice about them, then goes into
flagged-points mode (flagged_points.py): paused on the first flagged
frame, with buttons to jump between flagged frames, Smooth points / Unsmooth
(pose_smoothing.py; smoothed points turn yellow) and Confirm, which goes on to
the playback. Flagged points are drawn in orange everywhere and left out of
the timing until smoothed; skipping a video with unconfirmed flags records
"pose detection anomalies" as the reason.
"""

import cv2

from tmwt.core.job import COURSE_M, REVIEW_APPROVED, REVIEW_REJECTED, STATUS_NO_BODY
from tmwt.detection import people
from tmwt.measurement import timing
from tmwt.session.flagged_points import alert_pose_flags, review_flagged_points
from tmwt.session.review_playback import playback
from tmwt.ui.sidebar import APPROVED_MARK, DONE, FAILED, REJECTED_MARK, WORKING
from tmwt.ui.widgets import GREY, KEY_ENTER, KEY_ESC, ORANGE, WHITE

# review_job result: finish the review now (go to "Review complete"). (Stopping
# to continue later is the top bar's job: see ui/events.py.)
QUIT = "quit"
# Outcomes of setting endpoints (_set_endpoints).
_REPLAY = "replay"        # endpoints set: play the video from the start
_CANCELLED = "cancelled"  # the user cancelled endpoint picking
_SKIP = "skip"            # the user chose to skip the file
def review_job(job, ui, i):
    """
    Review one analysed job (index i in the batch) in the window (see the
    module docs). A video reviewed before plays back with the timing it was
    given.

    Returns:
        QUIT (finish the review now), or None once this video is decided.
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

    if job.pose_flags:
        alert_pose_flags(job, ui)
        review_flagged_points(job, ui)

    note = None
    if job.far_ep is None or job.near_ep is None:
        reason = f"{job.name}: automatic detection failed ({job.endpoint_problem})"
        outcome, note = _set_endpoints(job, ui, reason)
        if outcome == _SKIP:
            _reject(job, ui, i, "no walk timing found")
            return None
        if outcome == _CANCELLED:
            note = "Rope endpoints not set: use the menu (Esc) to set them, or to skip the video"

    k = None            # where the playback resumes; None = from the start
    while True:
        result = playback(job, ui, start_k=k, notice=note)
        k, note = result.k, None
        if (result.start, result.end) != (job.walk_start, job.walk_end):
            note = _apply_marks(job, result.start, result.end, "marked during review")

        if result.value == "confirm":
            if note:
                continue   # the new marks aren't a walk: back to the playback with why
            if job.pose_flags and not job.pose_confirmed:
                review_flagged_points(job, ui)   # e.g. after changing the subject
                continue
            decision = _final_confirm(job, ui, result.last)
            if decision == "next":
                job.review, job.saved = REVIEW_APPROVED, False   # saved again with the next save
                ui.set_state(i, DONE, f"approved  {job.duration:.2f}s")
                ui.mark_reviewed(i, APPROVED_MARK)
                return None
            if decision == "skip":
                _reject(job, ui, i, _skip_reason(job))
                return None
            continue   # "back": the playback, where it was

        choice = _menu(job, ui, result.last, note)
        note = None
        if choice == "endpoints":
            outcome, note = _set_endpoints(job, ui)
            if outcome == _SKIP:
                _reject(job, ui, i, "no walk timing found")
                return None
            if outcome == _REPLAY:
                k = None
        elif choice == "skip":
            _reject(job, ui, i, _skip_reason(job))
            return None
        elif choice == "flags":
            review_flagged_points(job, ui)
        elif choice == "person":
            if _pick_subject(job, ui):
                k = None
        elif choice == "quit":
            return QUIT
        # "back": the playback, where it was.


def _skip_reason(job):
    """Why a skipped video was rejected, for the report."""
    return "pose detection anomalies" if job.pose_flags and not job.pose_confirmed else "skipped at review"


def _menu(job, ui, background, note=None):
    """The playback's menu (see the module docs). Returns the chosen option's value."""
    options = [("1", "Change rope endpoints", "endpoints", (ord("1"),)),
               ("2", "Mark this file to be skipped", "skip", (ord("2"),))]
    if job.pose_flags:
        options.append(("3", "Review flagged points (orange)", "flags", (ord("3"),)))
    if people.people_on_screen(job)[0] is not None:
        options.append(("4", "Wrong person tracked (pick the walker)", "person", (ord("4"),)))
    options += [("Esc", "Back to review", "back", (KEY_ESC,)),
                ("F", "Finish all (save all results)", "quit", (ord("f"), ord("F")))]
    return ui.ask_menu(background, "Menu", summary_lines(job), options, note)


def _final_confirm(job, ui, background):
    """
    The final confirmation screen: what will be saved for this video.

    Returns:
        "next" (approve it and go on), "back" (to the playback), or "skip"
        (only offered when the timing is incomplete).
    """
    lines = [(text, GREY) for text in summary_lines(job)]
    if job.pose_flags:
        smoothed = f", {len(job.pose_edits)} smoothed" if job.pose_edits else ""
        lines.append((f"Flagged pose points checked ({job.pose_flagged_frames} frame(s){smoothed}).", GREY))
    if job.timing_source == "auto" and job.timing_note:
        lines.append((job.timing_note, ORANGE))
    if job.duration is None:
        return ui.show_message(
            [("Timing incomplete", ORANGE)] + lines
            + [("Mark the walk start and stop (green and red dots), or skip this video.", GREY)],
            [("Go back", "back", KEY_ENTER + (KEY_ESC,)), ("Skip this file", "skip", ())],
            background=background)
    return ui.show_message([("Confirm this video?", WHITE)] + lines,
                           [("Next video", "next", KEY_ENTER), ("Go back", "back", (KEY_ESC,))],
                           background=background)


def _apply_marks(job, start, end, detail):
    """
    Make the user's marks the job's timing (manual), if they make a walk.
    Returns a note for the review prompt if they don't, else None.
    """
    if start is None or end is None:
        return "Timing needs both a start and a stop mark. Timing unchanged."
    if end <= start:
        return "The stop mark was before the start mark. Timing unchanged."
    job.walk_start, job.walk_end = start, end
    job.timing_source, job.timing_detail = "manual", detail
    print(f"  Timing set by hand: {start:.3f}s to {end:.3f}s")
    return None


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
    the timing. If that timing is incomplete, ask straight away whether to
    adjust the endpoints again, mark the timing in the playback, or skip the video.

    Returns:
        (outcome, note): outcome is _REPLAY, _CANCELLED or _SKIP; note is a
        message to show in the playback, or None.
    """
    start = job.far_ep or job.subject_start
    finish = job.near_ep
    while True:
        picked = ui.pick_endpoints(job.info.first_frame, reason, start, finish,
                                   auto_start=job.subject_start)
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

        hint = ([("If the subject was already walking when the video starts,", GREY),
                 ("drag the start point onto the start line.", GREY)]
                if job.far_ep_is_standing_spot else [])
        choice = ui.show_message([
            ("No walk start / end found", ORANGE),
            (summary_lines(job)[1], GREY),
            ("Adjust the endpoints, or mark the start and stop yourself in the playback.", GREY),
        ] + hint, [("Adjust endpoints", "adjust", ()),
            ("Mark it in the playback", "mark", KEY_ENTER),
            ("Skip this file", "skip", (KEY_ESC,))], background=job.info.first_frame)
        if choice == "skip":
            return _SKIP, None
        if choice == "mark":
            return _REPLAY, "Mark the walk start and stop with the green and red dots"
        reason = None   # "adjust": back to the picker


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
    job.pose_confirmed = False   # a different person: their pose hasn't been looked at
    job.subject_start = people.subject_start(job)
    if job.far_ep_is_standing_spot and job.subject_start is not None:
        job.far_ep = job.subject_start   # the start point is where this person stood
    print(f"  Subject changed to person {k + 1} of {len(present)}")
    timing.update_timing(job)
    return True


def _reject(job, ui, i, reason):
    job.review, job.review_note = REVIEW_REJECTED, reason
    job.saved = False
    ui.set_state(i, FAILED, f"rejected: {reason}")
    ui.mark_reviewed(i, REJECTED_MARK)
    print(f"  Rejected ({reason}); no outputs written.")
