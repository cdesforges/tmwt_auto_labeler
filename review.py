"""
TMWT review — the interactive part of the labeling, run locally after
process.py.

Loads each video's analysis file from <folder>/tmwt_analysis/, works out the
subject, endpoints and walk timing (in seconds), then opens the review window:
play each video back, approve it, fix its endpoints or timing, or skip it.
Outputs (CSV, timing, annotated videos) and the labeling report are written once
review is finished. No pose model is needed here.

The videos must be the same files that were processed (they're fingerprinted);
the folder can be a different path from the one used on the cluster.

Usage:
    python review.py --input_dir <videos folder> [--output_dir <dir>]
                            [--endpoint_behavior {first_foot,ankle_midpoint}]
                            [--no_display]
"""

import argparse
import os
import sys

from tmwt.detection import processing
from tmwt.session import review_session
from tmwt.core.job import END_BEHAVIORS, END_FIRST_FOOT
from tmwt.ui.events import WindowClosed
from tmwt.ui.labeler_ui import LabelerUI
from tmwt.ui.top_bar import folder_title


def parse_args():
    parser = argparse.ArgumentParser(
        description="TMWT review: review processed videos and save their timing.")
    parser.add_argument("--input_dir", required=True,
                        help="Folder of videos, with the tmwt_analysis/ subfolder that "
                             "process.py wrote.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for the outputs (default: <input_dir>/output).")
    parser.add_argument("--endpoint_behavior", choices=END_BEHAVIORS, default=END_FIRST_FOOT,
                        help="What counts as crossing the start line (when one is clicked) "
                             "and the finish line: the first toe to cross, big or small, on "
                             "either foot (first_foot, default), or the midpoint of the two "
                             "ankles (ankle_midpoint).")
    parser.add_argument("--no_display", action="store_true",
                        help="No window and no review: save the automatic results unreviewed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: '{args.input_dir}' is not a valid directory.")
    videos = processing.find_videos(args.input_dir)
    if not videos:
        sys.exit(f"No video files found in '{args.input_dir}'.")
    output_dir = args.output_dir or os.path.join(args.input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    jobs = review_session.make_jobs(videos, output_dir, args.endpoint_behavior)
    ui = None if args.no_display else LabelerUI([job.name for job in jobs],
                                                heading=folder_title(args.input_dir))
    try:
        review_session.load_jobs(jobs, ui)
        saved = review_session.run(jobs, ui, output_dir)
    except WindowClosed:
        return
    finally:
        if ui is not None:
            ui.close()
    if saved:
        print(f"\nAll done! Output files are in '{output_dir}'.")


if __name__ == "__main__":
    main()
