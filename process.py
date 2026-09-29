"""
TMWT processing — the slow part of the labeling, with no window, e.g. on a cluster.

For every video in a folder: pose estimation (everyone in view) and camera-drift
tracking over every frame, plus the ArUco finish marker. Each video's results
are written to <folder>/tmwt_analysis/<video file name>.npz (see
analysis_file.py). Then review them locally with review.py.

Videos that already have a usable analysis file made with the same settings are
skipped, so an interrupted run can simply be started again.

Usage:
    python process.py --input_dir <videos folder>
                             [--backend {mediapipe,mmpose,rtmlib}] [--model <name>]
                             [--device {auto,cpu,cuda,mps}] [--no_matte_crop]
                             [--reprocess] [--download_models]

On a cluster whose compute nodes have no internet, run once with
--download_models on a node that has, so the model files are cached (for rtmlib
in $TORCH_HOME/hub, else $XDG_CACHE_HOME/rtmlib/hub, else ~/.cache/rtmlib/hub).
That fetches the heavier model used to re-run videos with pose anomalies too.
"""

import argparse
import os
import sys

from tmwt.core import analysis_file
from tmwt.detection import processing
from tmwt.pose.pose_backend import BACKENDS, DEVICES, get_backend, heavier_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="TMWT processing: pose estimation and tracking for a folder of videos.")
    parser.add_argument("--input_dir",
                        help="Folder of videos (required unless --download_models). Analysis "
                             "files are written to its tmwt_analysis/ subfolder.")
    parser.add_argument("--backend", choices=BACKENDS, default="rtmlib",
                        help="Pose backend (default: rtmlib).")
    parser.add_argument("--model", default=None,
                        help="Pose model. rtmlib: 'balanced' | 'performance' | "
                             "'lightweight' (default: 'balanced'); mediapipe: .task file "
                             "path; mmpose: config path or alias (default: 'body26').")
    parser.add_argument("--device", choices=DEVICES, default="auto",
                        help="Where to run the pose model (default: auto — CUDA if "
                             "available, else CoreML on Apple Silicon, else CPU).")
    parser.add_argument("--no_matte_crop", action="store_true",
                        help="Don't crop solid-colour mattes (letterbox / pillarbox bars).")
    parser.add_argument("--reprocess", action="store_true",
                        help="Process every video, even those already processed.")
    parser.add_argument("--download_models", action="store_true",
                        help="Only download / load the pose model (and the heavier one used "
                             "for videos with pose anomalies), then exit.")
    args = parser.parse_args()
    if args.input_dir is None and not args.download_models:
        parser.error("the following arguments are required: --input_dir")
    return args


def main():
    args = parse_args()
    backend = get_backend(args.backend)
    backend.set_device(args.device)
    model_path = args.model or backend.DEFAULT_MODEL_PATH
    print(f"Backend: {args.backend}  |  Model: {model_path}  |  "
          f"Device: {backend.provenance()['device']}")

    if args.download_models:
        processing.load_pose_model(backend, args.backend, model_path)
        heavier = heavier_model(backend, model_path)
        if heavier is not None:
            processing.load_pose_model(backend, args.backend, heavier)
        print("Pose models ready.")
        return

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Error: '{args.input_dir}' is not a valid directory.")
    videos = processing.find_videos(args.input_dir)
    if not videos:
        sys.exit(f"No video files found in '{args.input_dir}'.")
    print(f"Found {len(videos)} video(s) in '{args.input_dir}'.")

    processing.process_all(videos, backend, args.backend, model_path,
                           matte_crop=not args.no_matte_crop, reprocess=args.reprocess)
    print(f"\nDone. Analysis files are in "
          f"'{os.path.join(args.input_dir, analysis_file.ANALYSIS_DIR)}'.")


if __name__ == "__main__":
    main()
