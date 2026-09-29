# TMWT Auto-Labeler

Automated labeling of Ten-Meter Walk / Ten-Meter Run Tests (10MWT / 10MRT) from
video. Detects the subject's pose, tracks their position along a rope (either
printed-and-cut or defined by an ArUco marker), and reports walk time, average
speed, and per-frame skeleton data.

The output CSVs are **de-identified** — they contain landmark coordinates and
rope positions only, no video frames. A companion `view.py` renders them as
skeleton-only playback for review.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The labeler's window uses [pygame](https://www.pygame.org), installed by
`requirements.txt`. OpenCV's own windows aren't used for it: on macOS they
report mouse clicks in the wrong place. The window runs in its own small
process, because OpenCV and pygame bundle different versions of SDL2 that
conflict when loaded together. The window can be resized freely; the picture
scales to fit and clicks stay accurate.

### Optional pose backends

The default backend is MediaPipe and is installed by `requirements.txt`. Two
alternative backends are supported:

| Backend    | Install                                                                                     | Notes                                                                                                          |
|------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| `mediapipe`| _(default, already installed)_                                                              | Fastest, lightest. Best for well-lit, close-range footage.                                                     |
| `rtmlib`   | `pip install rtmlib onnxruntime`                                                            | RTMPose via ONNX Runtime. Uses CoreML on Apple Silicon by default. Best accuracy at distance; clean install.   |
| `mmpose`   | `pip install openmim mmengine` then `MMCV_WITH_OPS=1 FORCE_CUDA=0 pip install --no-build-isolation "mmcv>=2.0.1,<2.2.0" && pip install mmpose mmdet` | Same RTMPose weights as `rtmlib`. Heavier install; CPU-only on Apple Silicon (no MPS ops in mmcv). |

---

## Usage

```bash
python label.py --input_dir <videos_dir> [options]
```

### Flags

| Flag           | Required | Default                                | Description                                                                                                              |
|----------------|----------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `--input_dir`  | ✓        | —                                      | Directory containing video files to process. Supported extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.wmv`, `.m4v`.        |
| `--output_dir` |          | `<input_dir>/output`                   | Directory to write output CSVs and videos.                                                                               |
| `--backend`    |          | `mediapipe`                            | Pose backend. One of `mediapipe`, `mmpose`, `rtmlib`.                                                                    |
| `--model`      |          | _(backend-specific)_                   | Pose model. Interpretation depends on the backend (see below).                                                           |
| `--no_matte_crop` |       | _off (cropping enabled)_               | Disable automatic cropping of solid-color mattes (letterbox / pillarbox bars) around the active picture.                 |
| `--no_display` |          | _off (window + review enabled)_        | Run unattended: no window and no review. Automatic results are saved unreviewed; videos needing manual endpoints are reported as failed. |

#### `--model` values by backend

| Backend    | Value type              | Default                              | Examples                                          |
|------------|-------------------------|--------------------------------------|---------------------------------------------------|
| `mediapipe`| Path to a `.task` file  | `models/pose_landmarker_full.task`   | `models/pose_landmarker_heavy.task`               |
| `mmpose`   | Alias or config path    | `human`                              | `human`, `wholebody`, `/path/to/config.py`        |
| `rtmlib`   | Mode name               | `balanced`                           | `balanced`, `performance`, `lightweight`          |

#### Environment variables

| Variable         | Applies to  | Description                                                                                                       |
|------------------|-------------|-------------------------------------------------------------------------------------------------------------------|
| `RTMLIB_DEVICE`  | `rtmlib`    | Override the ONNX Runtime device. Auto-picks `mps` on Apple Silicon, else `cpu`. Set to `cpu` to force CPU.       |

---

## Examples

```bash
# Default: MediaPipe on all videos in a directory
python label.py --input_dir media/session1 --output_dir results/session1

# RTMLib backend (recommended for distant subjects on Apple Silicon)
python label.py --input_dir media/session1 --backend rtmlib

# RTMLib with the highest-accuracy model
python label.py --input_dir media/session1 --backend rtmlib --model performance

# Force CPU on rtmlib (troubleshooting CoreML issues)
RTMLIB_DEVICE=cpu python label.py --input_dir media/session1 --backend rtmlib
```

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

A run processes every video in `--input_dir` in four phases, all in one
resizable window.
Every choice is an on-screen button: a click counts when the mouse is released
over the same button it was pressed on. Most buttons also have a keyboard
shortcut, shown on the button. The right-hand sidebar lists every file (scroll
it with the mouse wheel or trackpad when there are many),
colour-coded: **white** waiting, **yellow** being analysed / reviewed / saved,
**green** done, **orange** needs your input at review, **red** failed or rejected.

1. **Analyse (unattended).** Each video in turn: find the rope endpoints, run
   pose estimation and ground tracking over every frame, and detect the walk
   start and end. The window shows `Analysing <file> (n of N)` with a progress
   bar and a **Cancel** button. Nothing asks for input here, so the whole batch
   can run unattended. When every video is done, an **Analysis complete**
   screen summarises what was found, with **Start review** and **Save all
   without reviewing**.
2. **Review.** Each video is played back in real time with its detection
   overlaid, with **Pause / Resume** and **Skip to review** buttons below it.
   It then asks whether the detection was successful:

   | Button (key) | What happens |
   |---|---|
   | Looks good (`1` / Enter) | The video is marked approved and the next one starts right away. |
   | Rope endpoints inaccurate (`2`) | Re-place the endpoints on the first frame (see [Endpoint detection](#endpoint-detection)). Timing is recomputed from the cached analysis and the video replays. |
   | Walk start/stop inaccurate (`3`) | The video replays in real time. Click **Mark start** when the walk starts and **Mark stop** when it ends (Space also works). The mark uses the frame on screen when the button was pressed. |
   | Wrong person tracked (`5`) | Only shown when several people were tracked. Click the person doing the walk test on a frame showing everyone; timing is recomputed for them and the video replays. |
   | Body not detected (`4`) | The file is skipped: no outputs are written, and it's reported as rejected. |
   | Replay (`R`) | Play the video again. |
   | Quit review (Esc) | Review stops; the remaining videos keep their automatic results, marked unreviewed. |

3. **Save.** Once review is finished, the outputs of every video that wasn't
   rejected are written in one go, with a progress bar. Nothing is written
   during review, so there's no wait between videos.
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

When the far endpoint is the subject's standing position (both endpoints
auto-detected), the end is the moment the ankles cross the near endpoint and the
start is found by working backwards from it:

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
python view.py <csv_directory> [--no_content_crop]
```

| Flag                | Default                 | Description                                                                     |
|---------------------|-------------------------|---------------------------------------------------------------------------------|
| `--no_content_crop` | _off (cropping enabled)_| Show the full recorded frame instead of cropping to the drawn-content bounds.    |

Playback controls:

| Key      | Effect                                       |
|----------|----------------------------------------------|
| Space    | Pause / resume                               |
| ← / →    | Step back / forward one frame (while paused) |
| `q`      | Next file (quits after the last one)         |

---

## Repository layout

```
label.py             # Entry point: CLI and the analyse -> review -> report batch driver
job.py               # Data model: VideoJob (one per video) and FrameResult (one per frame)
video_io.py          # Opening videos: first content frame, matte crop, playback clock
matte.py             # Letterbox / pillarbox detection and the cropping capture wrapper
endpoints.py         # Finish line from the ArUco marker
analysis.py          # Phase 1: pose + ground tracking over every frame
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
pose_common.py       # Shared 33-landmark pose layout, drawing and ankle helpers
pose_mediapipe.py    # MediaPipe backend
pose_mmpose.py       # MMPose backend
pose_rtmlib.py       # RTMLib (ONNX Runtime) backend
```
