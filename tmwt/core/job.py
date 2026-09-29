"""
Data model for a labeling run: one VideoJob per video, holding one FrameResult
per analysed frame.

A job moves through the run like this:

  analysis.process_video -> the slow part (pose estimation, camera tracking,
                            ArUco); fills info, frames and aruco_finish. Saved to
                            and loaded from an analysis file (analysis_file.py),
                            so it can run on another machine.
  analysis.interpret     -> the fast part: people tracks, subject, endpoints and
                            automatic timing; sets `status`
  review.review_job     -> may change endpoints / timing; sets `review`
  data_export.save_job  -> writes the outputs after all reviews; sets `saved`
  report.write_report   -> summarises every job
"""

from dataclasses import dataclass, field
import os
from typing import List, Optional

import numpy as np

from tmwt.core import video_io
from tmwt.core.video_io import VideoInfo

# Real length of the walking course, in metres. far_ep is 0 m, near_ep is COURSE_M.
COURSE_M = 10.0

# Which body point decides when the subject crosses the start and finish lines
# (label.py --endpoint_behavior; see timing.py).
END_FIRST_FOOT = "first_foot"           # the first ankle to cross the line
END_ANKLE_MIDPOINT = "ankle_midpoint"   # the midpoint of the two ankles
END_BEHAVIORS = (END_FIRST_FOOT, END_ANKLE_MIDPOINT)

# Analysis status (VideoJob.status).
STATUS_PENDING = "pending"            # not analysed yet
STATUS_OK = "ok"                      # endpoints and full timing found automatically
STATUS_INCOMPLETE = "incomplete"      # endpoints found, but no walk start or end
STATUS_NEEDS_INPUT = "needs_input"    # endpoint(s) must be clicked at review
STATUS_NO_BODY = "no_body"            # nobody detected in any frame; can only be skipped
STATUS_FAILED = "failed"              # can't be used at all (see VideoJob.error)

# Review outcome (VideoJob.review).
REVIEW_UNREVIEWED = "unreviewed"
REVIEW_APPROVED = "approved"
REVIEW_REJECTED = "rejected"


Point = tuple  # an (x, y) pixel position


@dataclass
class FrameResult:
    """Everything the analysis learned about one video frame."""
    frame_idx: int                         # 0 = the video's first content frame
    time_s: float                          # video timestamp
    H: Optional[np.ndarray]                # reference -> this frame homography (None if lost)
    people: list = field(default_factory=list)   # every person's pose (pose_common layout)

    # The subject (set by people.set_subject): their pose and ankle midpoint in
    # this frame's pixels, or None where they weren't seen.
    pose: Optional[list] = None
    body_px: Optional[Point] = None

    # The subject in REFERENCE-frame pixels (see tracking.to_reference_frame).
    # Set only when both the ankles and the head were found; such frames make
    # up the walk track used for distance measurement.
    ref_foot: Optional[Point] = None
    ref_head: Optional[Point] = None
    ref_left_ankle: Optional[Point] = None
    ref_right_ankle: Optional[Point] = None

    # Derived from the rope endpoints by timing.apply_endpoints.
    far_ep: Optional[Point] = None         # rope endpoints in this frame's pixels
    near_ep: Optional[Point] = None
    t_along: Optional[float] = None        # body position along the rope (0 far, 1 near)
    t_smooth: Optional[float] = None       # EMA-smoothed t_along


@dataclass
class VideoJob:
    """One video's progress through analysis, review and saving."""
    path: str
    output_path: str                        # CSV path; the other outputs sit next to it
    name: str

    info: Optional[VideoInfo] = None        # set by analysis
    frames: List[FrameResult] = field(default_factory=list)
    tracks: list = field(default_factory=list)   # people.Track per person followed
    subject: Optional[int] = None           # id of the subject's track

    # Where the subject stood at the start (reference-frame pixels): the far
    # endpoint when it's detected automatically, and the suggested start point
    # when the endpoints have to be clicked.
    subject_start: Optional[Point] = None
    # The finish point from the ArUco marker in the first frame (reference-frame
    # pixels), or None if there wasn't exactly one marker.
    aruco_finish: Optional[Point] = None
    # Where the analysis came from: backend, model, device and library versions
    # (see analysis_file.py).
    analysis_meta: dict = field(default_factory=dict)

    # Rope endpoints, in reference-frame pixels.
    far_ep: Optional[Point] = None
    near_ep: Optional[Point] = None
    endpoint_source: str = ""               # "auto" | "manual"
    endpoint_problem: str = ""              # why automatic detection failed
    # True when far_ep is the subject's standing position (auto-detected) rather
    # than a start line the user clicked; changes how the start is decided.
    far_ep_is_standing_spot: bool = False

    endpoint_behavior: str = END_FIRST_FOOT      # one of END_BEHAVIORS
    walk_start: Optional[float] = None      # seconds
    walk_end: Optional[float] = None
    timing_source: str = ""                 # "auto" | "manual"
    timing_detail: str = ""                 # how the start was found
    timing_note: str = ""                   # automatic start uncertain / missing: why (for the reviewer)

    status: str = STATUS_PENDING
    error: str = ""                         # reason for STATUS_FAILED
    review: str = REVIEW_UNREVIEWED
    review_note: str = ""
    saved: bool = False

    @property
    def track(self):
        """Frames usable for distance measurement (subject fully located)."""
        return [f for f in self.frames if f.ref_foot is not None]

    @property
    def duration(self):
        """Walk time in seconds, or None if the start or end is missing."""
        if self.walk_start is None or self.walk_end is None:
            return None
        return self.walk_end - self.walk_start

    @property
    def speed(self):
        """Walking speed in m/s, or None."""
        return COURSE_M / self.duration if self.duration else None

    def output_file(self, suffix):
        """Path of an output next to the CSV, e.g. output_file('_annotated.mp4')."""
        return os.path.splitext(self.output_path)[0] + suffix

    def open_capture(self):
        """Reopen the video exactly as analysed: same crop, starting at frame 0."""
        return video_io.open_video(self.path, self.info.crop, self.info.first_frame_idx)
