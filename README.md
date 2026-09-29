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

A run processes every video in `--input_dir` in three phases, all in one window.
Its right-hand sidebar lists every file, colour-coded: **white** waiting,
**yellow** being analysed / reviewed / saved, **green** done, **orange** needs
your input at review, **red** failed or rejected.

1. **Analyse (unattended).** Each video in turn: find the rope endpoints, run
   pose estimation and ground tracking over every frame, and detect the walk
   start and end. The window shows `Analysing <file> (n of N)` with a progress
   bar. Nothing asks for input here, so the whole batch can run unattended.
2. **Review.** Each video is played back in real time with its detection
   overlaid, then pauses and asks whether the detection was successful:

   | Key | Choice | What happens |
   |---|---|---|
   | `1` / Enter | Looks good | Outputs are saved and the video is marked approved. |
   | `2` | Rope endpoints inaccurate | Click the far (start) and near (finish) endpoints. Timing is recomputed from the cached analysis and the video replays. |
   | `3` | Walk start/stop inaccurate | The video replays in real time. Press Space when the walk starts and again when it ends. |
   | `4` | Body not detected | The file is skipped: no outputs are written, and it's reported as rejected. |
   | `R` | Replay | Play the video again. |
   | Esc | Quit review | The remaining videos are saved with their automatic results, marked unreviewed. |

   During playback, Space pauses and Enter skips to the prompt. In manual timing,
   Space marks the start and stop, and P pauses.
3. **Report.** `labeling_report.csv` and `labeling_report.md` summarise every
   video (see [Output](#output)).

## Endpoint detection

For each video, the labeler establishes two rope endpoints:

- **Near endpoint** — position of an ArUco marker at the finish line. Auto-detected.
- **Far endpoint** — where the subject stands at the start of the walk. Auto-detected from the pose landmarker.

If there's no single ArUco marker, or not exactly one person in the first frame,
the video is marked orange during analysis. When its review comes up you click
both endpoints, and a clicked far endpoint is treated as the start line.

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
first, so sideways sway does not read as forward movement. Tuning constants live
at the top of the hindsight section in `metric.py`.

When the far endpoint was clicked as a start line, the start is the moment the
ankles cross it if the subject was standing behind it. If they were already on or
past the line when they started moving, it's their first foot movement, as above.

## Output

For each saved video (`<basename>` = filename without extension), three files
are written to `--output_dir`. Videos rejected at review get no outputs.

| File                        | Contents                                                                                              |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| `<basename>.csv`            | Frame-by-frame body position, rope endpoints, normalized rope position (`t_along`), 33 pose landmarks.|
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
no video frames required.

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
| `q`      | Quit / advance to next file                  |

---

## Repository layout

```
label.py             # Main labeler entry point (analyse, review, report)
labeler_ui.py        # Batch window: progress, playback, review prompt, file sidebar
report.py            # End-of-run labeling report
metric.py            # Perspective-correct distances and hindsight walk-start detection
label_legacy.py      # Older two-click-only variant (kept for reference)
view.py              # Skeleton-only playback of saved CSVs
pose.py              # MediaPipe backend
pose_mmpose.py       # MMPose backend
pose_rtmlib.py       # RTMLib (ONNX Runtime) backend
pose_backend.py      # Backend factory
manual_selection.py  # Automatic endpoint detection and rope-endpoint click UI
tracking.py          # Ground-plane optical-flow tracker
data_export.py       # CSV writer
```
