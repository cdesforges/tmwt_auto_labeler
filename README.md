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

## Endpoint detection

For each video, the labeler establishes two rope endpoints:

- **Near endpoint** — position of an ArUco marker at the finish line. Auto-detected.
- **Far endpoint** — where the subject stands at the start of the walk. Auto-detected from the pose landmarker.

If either is missing, the tool falls back gracefully:

| Situation                         | Behavior                                                                                                                              |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Both auto-detected                | Fully automatic run. Timer starts on the first foot to move `≥ 5 cm` from its standstill position.                                    |
| Pose missed, ArUco found          | User clicks the far endpoint. The main loop runs with the **spacebar as the manual start trigger**; pose-based stop is still applied. |
| ArUco missed (any pose result)    | User clicks both endpoints. Timer runs fully automatically as if both were auto-detected.                                             |
| No timer stop detected            | User is prompted to retry the same video in **full manual mode** — spacebar drives both start and stop.                               |

### Controls during playback

| Key   | Effect                                                                     |
|-------|----------------------------------------------------------------------------|
| `q`   | Stop the current video early (writes what's been recorded so far).         |
| Space | Manual start / stop trigger, when the run is in a manual-timer mode.       |
| `r`   | On the "no stop detected" retry prompt: retry the video in full-manual mode.|

---

## Output

For each source video (`<basename>` = filename without extension), three files
are written to `--output_dir`:

| File                        | Contents                                                                                              |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| `<basename>.csv`            | Frame-by-frame body position, rope endpoints, normalized rope position (`t_along`), 33 pose landmarks.|
| `<basename>_annotated.mp4`  | Source frames with skeleton, rope, and info panel overlaid.                                           |
| `<basename>_skeleton.mp4`   | Black canvas with skeleton, rope, and info panel only — de-identified for sharing.                    |

The info panel on both output videos shows walk status, live timer, distance
from the camera (assuming a 10 m course), and per-foot motion score for
debugging.

---

## Reviewing results

The skeleton viewer plays back a saved CSV as a skeleton-only visualization —
no video frames required.

```bash
python view.py <csv_directory>
```

Playback controls:

| Key      | Effect                                       |
|----------|----------------------------------------------|
| Space    | Pause / resume                               |
| ← / →    | Step back / forward one frame (while paused) |
| `q`      | Quit / advance to next file                  |

---

## Repository layout

```
label.py             # Main labeler entry point
label_legacy.py      # Older two-click-only variant (kept for reference)
view.py              # Skeleton-only playback of saved CSVs
pose.py              # MediaPipe backend
pose_mmpose.py       # MMPose backend
pose_rtmlib.py       # RTMLib (ONNX Runtime) backend
pose_backend.py      # Backend factory
manual_selection.py  # Rope-endpoint UI and ArUco detection
tracking.py          # Ground-plane optical-flow tracker
data_export.py       # CSV writer
```
