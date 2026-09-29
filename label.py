"""
TMWT Labeler — processing and review in one go, locally.

This is the local shortcut for the two-step workflow:

  1. Process (processing.py, as process.py does, e.g. on a cluster):
     pose estimation and tracking for every video that isn't processed yet,
     written to <folder>/tmwt_analysis/, with progress in the window.
  2. Review (review_session.py, as review.py does): load the analysis
     files, work out the subject, endpoints and timing, review each video,
     save the outputs and write the report.

Because review always reads the analysis files, running label.py again on the
same folder skips straight to review, and the results are the same as
processing on a cluster and reviewing locally.

--no_display runs without a window: videos are processed, then the automatic
results are saved unreviewed.

Usage:
    python label.py --input_dir <dir> [--output_dir <dir>]
                    [--backend {mediapipe,mmpose,rtmlib}] [--model <path_or_alias>]
                    [--device {auto,cpu,cuda,mps}] [--no_matte_crop] [--reprocess]
                    [--endpoint_behavior {first_foot,ankle_midpoint}] [--no_display]

Outputs are described in data_export.py and report.py.
"""

import argparse
import os
import sys

from tmwt.detection import processing
from tmwt.session import review_session
from tmwt.core.job import END_BEHAVIORS, END_FIRST_FOOT
from tmwt.ui.labeler_ui import LabelerUI, WindowClosed
from tmwt.ui.top_bar import folder_title
from tmwt.pose.pose_backend import BACKENDS, DEVICES, get_backend


def parse_args():
    parser = argparse.ArgumentParser(
        description="TMWT Labeler — process and review 10 m walk test videos locally.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing the video files to process.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for the outputs (default: <input_dir>/output).")
    parser.add_argument("--backend", choices=BACKENDS, default="rtmlib",
                        help="Pose backend to use (default: rtmlib).")
    parser.add_argument("--model", default=None,
                        help="Pose model. rtmlib: 'balanced' | 'performance' | "
                             "'lightweight' (default: 'balanced'); mediapipe: .task file "
                             "path (default: models/pose_landmarker_full.task); mmpose: "
                             "config path or alias (default: 'body26').")
    parser.add_argument("--device", choices=DEVICES, default="auto",
                        help="Where to run the pose model (default: auto).")
    parser.add_argument("--no_matte_crop", action="store_true",
                        help="Don't crop solid-colour mattes (letterbox / pillarbox bars).")
    parser.add_argument("--reprocess", action="store_true",
                        help="Process every video again, even those already processed.")
    parser.add_argument("--endpoint_behavior", choices=END_BEHAVIORS, default=END_FIRST_FOOT,
                        help="What counts as crossing the start line (when one is clicked) "
                             "and the finish line: the first toe to cross, big or small, on "
                             "either foot (first_foot, default), or the midpoint of the two "
                             "ankles (ankle_midpoint).")
    parser.add_argument("--no_display", action="store_true",
                        help="Run unattended: no window and no review. Automatic results "
                             "are saved unreviewed; videos that need manual endpoints are "
                             "reported as failed.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: '{args.input_dir}' is not a valid directory.")
    backend = get_backend(args.backend)
    backend.set_device(args.device)
    model_path = args.model or backend.DEFAULT_MODEL_PATH
    print(f"Backend: {args.backend}  |  Model: {model_path}")

    videos = processing.find_videos(args.input_dir)
    if not videos:
        sys.exit(f"No video files found in '{args.input_dir}'.")
    print(f"Found {len(videos)} video(s) in '{args.input_dir}':")
    for v in videos:
        print(f"  - {os.path.basename(v)}")
    output_dir = args.output_dir or os.path.join(args.input_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    ui = None if args.no_display else LabelerUI([os.path.basename(v) for v in videos],
                                                heading=folder_title(args.input_dir))
    try:
        cancelled = processing.process_all(videos, backend, args.backend, model_path,
                                           matte_crop=not args.no_matte_crop,
                                           reprocess=args.reprocess, ui=ui)
        jobs = review_session.make_jobs(videos, output_dir, args.endpoint_behavior)
        review_session.load_jobs(jobs, ui)
        saved = review_session.run(jobs, ui, output_dir, review_first=not cancelled)
    except WindowClosed:
        print("Window closed; stopping. Processed videos keep their analysis files.")
        return
    finally:
        if ui is not None:
            ui.close()
    if saved:
        print(f"\nAll done! Output files are in '{output_dir}'.")


if __name__ == "__main__":
    main()
