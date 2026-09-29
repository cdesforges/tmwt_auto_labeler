# TMWT Auto-Labeler

Automated labeling of Ten-Meter Walk / Ten-Meter Run Tests (10MWT / 10MRT) from
video. Detects the subject's pose, tracks their position along a rope (either
printed-and-cut or defined by an ArUco marker), and reports walk time, average
speed, and per-frame skeleton data.

The output CSVs are **de-identified** — they contain landmark coordinates and
rope positions only, no video frames. A companion `view.py` renders them as
skeleton-only playback for review.

---

## Workflow

Labeling has two steps, which can run on different machines:

1. **Process** — `process_videos.py`: pose estimation and camera tracking for every
   video, with no window. This is the slow part (about 1.4× the video's length
   per video with rtmlib on Apple Silicon), so it can run on a cluster. Each
   video's results are written to `<videos folder>/tmwt_analysis/<video>.npz`
   (coordinates only, no images; about 100 KB per video).
2. **Review** — `review_videos.py`: run locally on the same folder (copy the
   videos and the `tmwt_analysis/` folder back). It works out the subject,
   endpoints and timing in seconds, opens the review window, and writes the
   outputs and the report. It needs no pose model.

`label.py` does both in one go locally, processing only videos that aren't
processed yet, so running it again on a folder goes straight to review.

```bash
# On the cluster (or any machine)
python process_videos.py --input_dir /data/session1

# Locally, with the videos and tmwt_analysis/ copied back
python review_videos.py --input_dir media/session1

# Or both at once, locally
python label.py --input_dir media/session1
```

Things to know:

- **Same videos.** Review checks each video against a fingerprint saved at
  processing, and refuses a video that differs ("re-process it"). The folder's
  path can differ between machines.
- **Resuming.** `process_videos.py` skips videos that already have an analysis
  file made with the same backend, model and matte setting, so an interrupted
  run can be restarted. `--reprocess` forces everything to run again.
- **Same OpenCV version.** Review re-reads frames and must see the same frames as
  processing did, so use the same OpenCV version on both machines.
- **Cluster results vs. local.** A GPU or CPU gives very slightly different
  numbers from CoreML on a Mac, which can shift timings a little. Worth one
  check on a few videos processed both ways.
- **No internet on compute nodes?** Run `python process_videos.py --download_models`
  once on a node that has internet. rtmlib caches models in `$TORCH_HOME/hub`, else
  `$XDG_CACHE_HOME/rtmlib/hub`, else `~/.cache/rtmlib/hub`; point these at a shared
  folder if home directories aren't shared.

An example SLURM job for one folder:

```bash
#!/bin/bash
#SBATCH --job-name=tmwt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
##SBATCH --gres=gpu:1          # uncomment for a GPU node (with onnxruntime-gpu)
source ~/tmwt/.venv/bin/activate
cd ~/tmwt
python process_videos.py --input_dir /data/session1 --device auto
```

---

## Installation

Locally (processing and review):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install rtmlib onnxruntime        # the default pose backend
```

On a cluster (processing only; no window, so OpenCV's headless build):

```bash
pip install -r requirements-cluster.txt
```

The review window uses [pygame](https://www.pygame.org), installed by
`requirements.txt`. OpenCV's own windows aren't used: on macOS they report mouse
clicks in the wrong place. The window runs in its own small process, because
OpenCV and pygame bundle different versions of SDL2 that conflict when loaded
together. It can be resized freely; the picture scales to fit and clicks stay
accurate.

### Pose backends

| Backend    | Install                                                                                     | Notes                                                                                                          |
|------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `rtmlib` (default) | `pip install rtmlib onnxruntime` (`onnxruntime-gpu` for NVIDIA GPUs)             | RTMPose "body with feet" (Halpe-26) via ONNX Runtime: body plus big toe, small toe and heel per foot. CoreML on Apple Silicon, CUDA on NVIDIA GPUs, else CPU. Best accuracy at distance. |
| `mediapipe`| installed by `requirements.txt`                                                             | Fastest, lightest; CPU only. Heel and big toe, no small toe. Best for well-lit, close-range footage.            |
| `mmpose`   | `pip install openmim mmengine` then `MMCV_WITH_OPS=1 FORCE_CUDA=0 pip install --no-build-isolation "mmcv>=2.0.1,<2.2.0" && pip install mmpose mmdet` | Same RTMPose weights as `rtmlib` (`body26`). Heavier install; CPU or CUDA (no MPS ops in mmcv). |

---

## Usage

### `process_videos.py`

| Flag                | Default              | Description |
|---------------------|----------------------|-------------|
| `--input_dir`       | _required_           | Folder of videos (`.mp4`, `.mov`, `.avi`, `.mkv`, `.wmv`, `.m4v`). Analysis files go in its `tmwt_analysis/` subfolder. |
| `--backend`         | `rtmlib`             | Pose backend: `rtmlib`, `mediapipe` or `mmpose`. |
| `--model`           | _(backend-specific)_ | Pose model (see below). |
| `--device`          | `auto`               | `auto` (CUDA if available, else CoreML on Apple Silicon, else CPU), `cpu`, `cuda` or `mps`. |
| `--no_matte_crop`   | _off_                | Don't crop solid-colour mattes (letterbox / pillarbox bars). |
| `--reprocess`       | _off_                | Process every video, even ones already processed. |
| `--download_models` | _off_                | Only download / load the pose model, then exit. |

### `review_videos.py`

| Flag                  | Default              | Description |
|-----------------------|----------------------|-------------|
| `--input_dir`         | _required_           | Folder of videos with its `tmwt_analysis/` subfolder. |
| `--output_dir`        | `<input_dir>/output` | Where to write the CSVs, timing files, videos and report. |
| `--endpoint_behavior` | `first_foot`         | What counts as crossing the start line (when one is clicked) and the finish line: the first toe to cross, big or small, on either foot (`first_foot`), or the midpoint of the two ankles (`ankle_midpoint`). |
| `--no_display`        | _off_                | No window and no review: save the automatic results unreviewed. |

### `label.py`

Takes all the flags of both scripts (`--input_dir`, `--output_dir`, `--backend`,
`--model`, `--device`, `--no_matte_crop`, `--reprocess`, `--endpoint_behavior`,
`--no_display`).

#### `--model` values by backend

| Backend    | Value type              | Default                              | Examples                                          |
|------------|-------------------------|--------------------------------------|---------------------------------------------------|
| `rtmlib`   | Mode name               | `balanced`                           | `balanced`, `performance`, `lightweight`          |
| `mediapipe`| Path to a `.task` file  | `models/pose_landmarker_full.task`   | `models/pose_landmarker_heavy.task`               |
| `mmpose`   | Alias or config path    | `body26`                             | `body26`, `/path/to/config.py`                    |

#### Environment variables

| Variable         | Applies to  | Description |
|------------------|-------------|-------------|
| `RTMLIB_DEVICE`  | `rtmlib`    | Overrides `--device` for rtmlib. |
| `TORCH_HOME` / `XDG_CACHE_HOME` | `rtmlib` | Where rtmlib caches downloaded models. |

---

## Matte cropping

Video shot or exported on phones is often padded with solid black, grey, or
white bars (letterbox / pillarbox) so the active picture fills only part of the
frame. Before processing, the labeler inspects the first frame, detects any such
matte, and crops every frame to the active-picture rectangle — recovering pose-
detector resolution and keeping ArUco and ground tracking on real content.

Detection is automatic and a no-op when no matte is present. Pass
`--no_matte_crop` to disable it. All output coordinates and videos are in the
cropped frame's space.

`view.py` applies the same idea at playback. It has no video to inspect — it
draws on a generated canvas — so instead it crops to the bounding box of
everything actually recorded (skeleton, body point, rope endpoints) plus a small
margin. This removes dead margins and also tightens up CSVs recorded before
matte cropping existed. Pass `--no_content_crop` to see the full recorded frame.

## How a run works

A `label.py` run processes every video in `--input_dir` in four phases, all in
one resizable window. (`review_videos.py` starts at the review, loading the
analysis files instead of analysing.)
Every choice is an on-screen button: a click counts when the mouse is released
over the same button it was pressed on. Most buttons also have a keyboard
shortcut, shown on the button. The right-hand sidebar lists every file (scroll
it with the mouse wheel or trackpad when there are many),
colour-coded: **white** waiting, **yellow** being analysed / reviewed / saved,
**green** done, **orange** needs your input at review, **red** failed or rejected.

1. **Analyse (unattended).** The pose model loads first (a few seconds, shown
   in the window). Then each video in turn: find the rope endpoints, run
   pose estimation and ground tracking over every frame, and detect the walk
   start and end. The window shows `Analysing <file> (n of N)` with a progress
   bar and a **Cancel** button. Nothing asks for input here, so the whole batch
   can run unattended. When every video is done, an **Analysis complete**
   screen summarises what was found, with **Start review** and **Save all
   without reviewing**; clicking a video in the sidebar starts the review there.
2. **Review.** Each video is played back in real time with its detection
   overlaid, with **Pause / Resume** and **Skip to review** buttons below it.
   Once a video has been reviewed, the sidebar splits into **Unreviewed** and
   **Reviewed** sections; reviewed videos move down, dimmed, with a green check
   (approved) or a red cross (skipped).
   You can click any video in the sidebar at any time to review it instead,
   including one already reviewed (the new review replaces its result). After
   each decision the next unreviewed video follows, wrapping round to any you
   skipped; videos already decided are never revisited automatically.
   It then asks whether the detection was successful:

   | Button (key) | What happens |
   |---|---|
   | Looks good (`1` / Enter) | The video is marked approved and the next one starts right away. |
   | Rope endpoints inaccurate (`2`) | Re-place the endpoints on the first frame (see [Endpoint detection](#endpoint-detection)). Timing is recomputed from the cached analysis and the video replays. |
   | Walk start/stop inaccurate (`3`) | The video replays in real time. Click **Mark start** when the walk starts and **Mark stop** when it ends (Space also works). The mark uses the frame on screen when the button was pressed. |
   | Wrong person tracked (`5`) | Only shown when several people were tracked. Click the person doing the walk test on a frame showing everyone; timing is recomputed for them and the video replays. |
   | Skip this file (`4`) | The file is skipped: no outputs are written, and it's reported as rejected. |
   | Replay (`R`) | Play the video again. |
   | Quit review (Esc) | Review stops; the remaining videos keep their automatic results, marked unreviewed. |

3. **Save.** Once review is finished (or you quit it), a **Review complete**
   screen shows what will be saved. **Save results** writes the outputs of every
   video that wasn't skipped, in one go, with a progress bar; **Exit without
   saving** writes nothing (the analysis files stay, so you can review again
   without re-processing). Click a video in the sidebar to change it first; you
   come back to this screen afterwards. Nothing is written during review, so
   there's no wait between videos.
4. **Report.** `labeling_report.csv` and `labeling_report.md` summarise every
   video (see [Output](#output)).

## Endpoint detection

For each video, the labeler establishes two rope endpoints:

- **Near endpoint (finish)** — a corner of the ArUco marker at the finish line.
- **Far endpoint (start)** — where the subject stands at the start of the walk:
  their feet in the first frame they're fully seen.

Without a single ArUco marker, the video is marked orange during analysis. When
its review comes up, the start point is already placed where the subject was
standing and you click only the **finish** point, then **Confirm**. If the
start is wrong, **Move start point** lets you click it; a start you place is
treated as a start line (timing starts when the subject crosses it, or at their
first movement if they're already on or past it).

If the endpoints don't give both a start and an end time, you're asked straight
away to **Redo endpoints**, **Time manually** or **Skip this file**. A common
cause: the subject is already walking when the video begins, so there's no
standing start. Use **Move start point** and click the start line.

If you cancel endpoint picking, you're taken back to the review options, where
you can try again, time the video manually, or skip it. A video in which nobody
was detected at all only offers **Skip this file**.

## Foot points

Every backend reports the same 35 landmarks (MediaPipe's 33 plus a small toe
per foot), saved as `lm_00` to `lm_34` in each CSV. Foot points, left / right:
ankle 27 / 28, heel 29 / 30, big toe 31 / 32, small toe 33 / 34. rtmlib and
mmpose use Halpe-26 "body with feet" models, which provide all of them (not the
face or hand detail); MediaPipe provides all but the small toes. If no toe is
seen crossing a line, the crossing falls back to the ankles, then the ankle
midpoint.

## Multiple people

Everyone in view is detected and followed through the video (up to five
people), and the subject is chosen afterwards: in a 10 m walk test they walk
toward the camera, so they grow much larger in the picture than anyone standing
by or walking alongside. Duplicate detections of one person (common when they're
right in front of the camera) are merged. If the wrong person was chosen, use
**Wrong person tracked** at review; every person's poses are kept, so switching
takes no re-analysis.

## Walk start detection

The walk timing is decided after each video has been fully analysed, so it can
look back over the whole walk.

The end is the moment the subject crosses the finish line (near endpoint). Line
crossings — the finish line, and the start line when one is clicked — count
the first toe to cross (big or small, either foot) by default, or the midpoint of the two ankles with
`--endpoint_behavior ankle_midpoint`. When the far endpoint is the subject's standing
position, the start is found by working backwards from that end:

1. **Confirm the walk.** Find the sustained forward advance of the ankles that
   leads into the end and trace it back to where it began. Fidgets, heel raises
   and weight shifts never produce a sustained advance, so they can't trigger
   the start.
2. **Find the onset.** The second before that advance is treated as the
   standstill. Each foot's baseline is the median position there, and its
   noise is the MAD. The start is the first frame a foot leaves that band and
   then stays clearly forward (`max(5 cm, 4 × noise)` for at least 0.15 s). The
   look-back includes the heel lift that begins the step, up to 0.3 s before
   the forward swing.

Distances are perspective-corrected using the vanishing point of the subject's
own walk (see `metric.py`). Ankle positions are projected onto the walking line
first, so sideways sway does not read as forward movement. The detection and its
tuning constants live in `onset.py`; `timing.py` combines it with the line
crossings.

When the far endpoint was clicked as a start line, the start is the moment the
ankles cross it if the subject was standing behind it. If they were already on or
past the line when they started moving, it's their first foot movement, as above.

## Output

For each saved video (`<basename>` = filename without extension), four files
are written to `--output_dir`. Videos rejected at review get no outputs.

| File                        | Contents                                                                                              |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| `<basename>.csv`            | Frame-by-frame body position, rope endpoints, normalized rope position (`t_along`), 33 pose landmarks.|
| `<basename>_timing.json`    | The walk timing the labeler decided (start, end, duration, speed), how the start was found, and the review outcome. `view.py` reads it. |
| `<basename>_annotated.mp4`  | Source frames with skeleton, rope, and info panel overlaid.                                           |
| `<basename>_skeleton.mp4`   | Black canvas with skeleton, rope, and info panel only — de-identified for sharing.                    |

The info panel on both output videos shows walk status, timer, and distance
from the camera (assuming a 10 m course).

Each run also writes `labeling_report.csv` and `labeling_report.md` with one row
per video: its result (`approved`, `auto (not reviewed)`, `rejected`,
`failed`), the reason, whether the endpoints and timing were automatic or
manual, how the start was found, the start / end / duration / speed, and
whether outputs were saved.

---

## Reviewing results

The skeleton viewer plays back a saved CSV as a skeleton-only visualization —
no video frames required. Its timer uses the `<basename>_timing.json` saved next
to each CSV, so it shows exactly the timing approved at review. Older CSVs
without that file fall back to the start-line and finish-line crossings.

```bash
python view.py --input_dir <output folder> [--no_content_crop]
```

| Flag                | Default                 | Description                                                                     |
|---------------------|-------------------------|---------------------------------------------------------------------------------|
| `--no_content_crop` | _off (cropping enabled)_| Show the full recorded frame instead of cropping to the drawn-content bounds.    |

It uses the same window as the review: every recording in the folder is listed
in the sidebar, coloured by its saved result (green approved, grey not
reviewed, red incomplete timing, white an older CSV with no timing file) with
its walk time. Click a recording to play it. The buttons below the picture:

| Button | Key   | Effect |
|--------|-------|--------|
| `|◀◀`  | P     | Previous recording |
| `|◀`   | ←     | Step back one frame (pauses) |
| `▶` / `‖` | Space | Play / pause |
| `▶|`   | →     | Step forward one frame (pauses) |
| `▶▶|`  | N     | Next recording |

At the end of a recording the next one plays; after the last, playback pauses on
its final frame. Close the window (or press Esc) to quit.

---

## Repository layout

```
process_videos.py    # Step 1 (e.g. on a cluster): processing, writes tmwt_analysis/*.npz
review_videos.py     # Step 2 (locally): review from the analysis files
label.py             # Both steps in one go, locally
processing.py        # Processing a folder: skip / process / save analysis files
analysis_file.py     # The analysis file format: save, load, video fingerprint
review_session.py    # Loading, reviewing, saving and reporting a folder
job.py               # Data model: VideoJob (one per video) and FrameResult (one per frame)
video_io.py          # Opening videos: first content frame, matte crop, playback clock
matte.py             # Letterbox / pillarbox detection and the cropping capture wrapper
endpoints.py         # Finish line from the ArUco marker
analysis.py          # Slow part (pose + tracking per frame) and fast part (subject, endpoints, timing)
tracking.py          # Ground-plane optical-flow tracker (camera drift)
metric.py            # Geometry: t_along and perspective-correct distance along the course
onset.py             # Hindsight walk-start detection on distance signals
timing.py            # Walk start/end from the analysed frames
review.py            # Phase 2: real-time playback and the review flow
people.py            # Following everyone in view and choosing the walking subject
labeler_ui.py        # The labeler's screens: progress, playback, buttons, endpoint picking, sidebar
window.py            # The window as the labeler sees it: show a canvas, poll for keys and clicks
window_server.py     # The window process: resizable pygame window, canvas scaled to fit
annotate.py          # Frame drawing (skeleton, rope, info panel) shared with view.py
data_export.py       # Per-video outputs (CSV, timing JSON, videos) and reading them back
report.py            # End-of-run labeling report
view.py              # Skeleton-only playback of saved CSVs
pose_backend.py      # Pose backend factory
pose_common.py       # Shared pose layout (body + feet), drawing and ankle helpers
pose_mediapipe.py    # MediaPipe backend
pose_mmpose.py       # MMPose backend
pose_rtmlib.py       # RTMLib (ONNX Runtime) backend
```
