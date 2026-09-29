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

1. **Process** — `process.py`: pose estimation and camera tracking for every
   video, with no window. This is the slow part (about 1.4× the video's length
   per video with rtmlib on Apple Silicon), so it can run on a cluster. Each
   video's results are written to `<videos folder>/tmwt_analysis/<video>.npz`
   (coordinates only, no images; about 100 KB per video).
2. **Review** — `review.py`: run locally on the same folder (copy the
   videos and the `tmwt_analysis/` folder back). It works out the subject,
   endpoints and timing in seconds, opens the review window, and writes the
   outputs and the report. It needs no pose model.

`label.py` does both in one go locally, processing only videos that aren't
processed yet, so running it again on a folder goes straight to review.

```bash
# On the cluster (or any machine)
python process.py --input_dir /data/session1

# Locally, with the videos and tmwt_analysis/ copied back
python review.py --input_dir media/session1

# Or both at once, locally
python label.py --input_dir media/session1
```

Things to know:

- **Same videos.** Review checks each video against a fingerprint saved at
  processing, and refuses a video that differs ("re-process it"). The folder's
  path can differ between machines.
- **Resuming.** `process.py` skips videos that already have an analysis
  file made with the same backend, model and matte setting, so an interrupted
  run can be restarted. `--reprocess` forces everything to run again.
- **Same OpenCV version.** Review re-reads frames and must see the same frames as
  processing did, so use the same OpenCV version on both machines.
- **No internet on compute nodes?** Run `python process.py --download_models`
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
python process.py --input_dir /data/session1 --device auto
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
`requirements.txt`. The window runs in its own small process, because
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

### `process.py`

| Flag                | Default              | Description |
|---------------------|----------------------|-------------|
| `--input_dir`       | _required_           | Folder of videos (`.mp4`, `.mov`, `.avi`, `.mkv`, `.wmv`, `.m4v`). Analysis files go in its `tmwt_analysis/` subfolder. |
| `--backend`         | `rtmlib`             | Pose backend: `rtmlib`, `mediapipe` or `mmpose`. |
| `--model`           | _(backend-specific)_ | Pose model (see below). |
| `--device`          | `auto`               | `auto` (CUDA if available, else CoreML on Apple Silicon, else CPU), `cpu`, `cuda` or `mps`. |
| `--no_matte_crop`   | _off_                | Don't crop solid-colour mattes (letterbox / pillarbox bars). |
| `--reprocess`       | _off_                | Process every video, even ones already processed. |
| `--download_models` | _off_                | Only download / load the pose model (and the heavier one used for videos with pose anomalies), then exit. |

Videos whose pose check finds implausible points are processed again with the
heavier model automatically; see [Pose check](#pose-check). The results are in
`tmwt_analysis/processing_report.csv` / `.md`.

### `review.py`

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
one resizable window. (`review.py` starts at the review, loading the
analysis files instead of analysing.)
Every choice is an on-screen button: a click counts when the mouse is released
over the same button it was pressed on. Most buttons also have a keyboard
shortcut: hover over a button to see what it does and its shortcut, e.g.
"Confirm (Enter)" or "Back one frame (Left arrow)" (menu options show theirs
at the left, e.g. "[1]"). The right-hand sidebar lists every file (scroll
it with the mouse wheel or trackpad when there are many),
colour-coded: **white** waiting, **yellow** being analysed / reviewed / saved,
**green** done, **orange** needs your input at review, **red** failed or rejected.

The **top bar** is always there: an **X** at the top left, then (during a
review) a **save** button (floppy disk); the folder's name in the middle
(shortened in the middle if it's long); the University of Rochester shield
and the RNA Institute rosette at the right (hover over one to see its name).

- **Save** (floppy disk): save your review progress and quit, from any screen.
- **X**: quit. During a review it asks first: **Save progress & quit**, **Quit
  without saving** (discards this review's decisions, after asking whether
  you're sure), or **Cancel** (carry on where you were). Outside a review it quits straight away, like closing the
  window.

1. **Analyse (unattended).** The pose model loads first (a few seconds, shown
   in the window). Then each video in turn: find the rope endpoints, run
   pose estimation and ground tracking over every frame, and detect the walk
   start and end. The window shows `Analysing <file> (n of N)` with a progress
   bar and a **Cancel** button. Nothing asks for input here, so the whole batch
   can run unattended. When every video is done, an **Analysis complete**
   screen summarises what was found, with **Start review** and **Save all
   without reviewing**; clicking a video in the sidebar starts the review there.
2. **Review.** Each video is played back in real time with its detection
   overlaid. Below it: a seek bar (drag or click to scrub; the walk's start and
   end are marked in green and red), the mark buttons (a **green dot** marks the
   walk start and a **red dot** the stop, at the frame on screen; `M` presses
   them in turn), frame-back / play-pause / frame-forward buttons (←, Space, →;
   hold an arrow key to keep stepping), **Confirm** (Enter; hover to see
   shortcuts) and the menu button
   **☰** (Esc). Hover over an icon button to see what it does. The video
   doesn't end the playback: it pauses on the last frame.
   **Correcting the timing while you watch:** pause, then mark the start or
   stop again with the dot buttons, or drag a mark along the seek bar by the
   small tab under it (the picture follows, and the playhead hides while you
   drag so you can see where the mark will land). When the frame on screen is
   a mark's frame, the playhead fills with that mark's colour (green or red,
   inside a white rim), so you can land on it exactly. Pressing the track itself
   always scrubs, even where a mark is. Changed marks are shown at the top
   left, and they replace the automatic timing (it becomes manual). A stop
   before the start is flagged in red, and Confirm waits until it's fixed.

   **Confirm** opens the final confirmation screen: the endpoints, the timing
   (start, end, duration, speed) and any pose points checked. **Next video**
   (Enter) approves the video and moves on; **Go back** (Esc) returns to the
   playback, paused where you were. If the timing is incomplete, it says so
   and offers **Go back** or **Skip this file** instead.

   **☰** opens the menu:

   | Option (key) | What happens |
   |---|---|
   | Change rope endpoints (`1`) | Re-place the endpoints on the first frame (see [Endpoint detection](#endpoint-detection)). Timing is recomputed from the cached analysis and the video replays. |
   | Mark this file to be skipped (`2`) | No outputs are written, and it's reported as rejected. |
   | Review flagged points (`3`) | Only for videos with flagged pose points: back to flagged-points mode (see [Pose check](#pose-check)). |
   | Wrong person tracked (`4`) | Only when several people were tracked. On a frame showing everyone, people are drawn in blue (brighter under the pointer) and the person currently tracked in green; click the person doing the walk test (they turn green), then **Confirm**. Timing is recomputed for them and the video replays; confirming the person already tracked changes nothing. |
   | Back to review (Esc) | Back to the playback, where you were. |
   | Finish all (save all results) (`F`) | Go to the **Review complete** screen now; videos not reviewed yet keep their automatic results, marked unreviewed. |

   Once a video has been reviewed, the sidebar splits into **Unreviewed** and
   **Reviewed** sections; reviewed videos move down, dimmed, with a check
   (grey: approved, green: saved) or a red cross (skipped).
   You can click any video in the sidebar at any time to review it instead,
   including one already reviewed: it plays back with the timing you gave it,
   and the new review replaces its result. After each decision the next
   unreviewed video follows, wrapping round to any you skipped; videos
   already decided are never revisited automatically.

   To stop and come back later, use the top bar's **save** button (or **X**,
   then **Save progress & quit**): your decisions so far are saved, no outputs
   are written, and the next review of the folder offers to continue.

3. **Save.** Once every video is reviewed (or you choose **Finish all**), a
   **Review complete** screen shows what will be saved. **Save results** writes
   the outputs of every video that wasn't skipped, in one go, with a progress
   bar; the top bar's **save** button writes nothing yet and keeps your review
   progress for next time. Click a video in the sidebar to change it first; you
   come back to this screen afterwards. Nothing is written during review, so
   there's no wait between videos.
   **Progress is kept.** Review progress is saved as you go (after every
   decision and whenever you switch video) to `tmwt_analysis/review_progress.json`,
   and it's **kept after the results are saved**. The next review of the
   folder asks whether to **Continue** — where you left off, or, for a
   finished review, at **Review complete** so you can change videos and save
   again (saved videos show green checks; a video you change goes back to a
   grey check until the next save, and one you reject has its earlier outputs
   removed) — or **Start over** (which asks whether you're sure, since it
   discards those decisions). **Quit without saving** (top bar X) puts the
   progress back as it was when that review began. Progress for a video that
   has since been replaced or re-processed is ignored.
4. **Report.** `labeling_report.csv` and `labeling_report.md` summarise every
   video (see [Output](#output)).

## Endpoint detection

For each video, the labeler establishes two rope endpoints:

- **Near endpoint (finish)** — a corner of the ArUco marker at the finish line.
- **Far endpoint (start)** — where the subject stands at the start of the walk:
  their feet in the first frame they're fully seen.

Without a single ArUco marker, the video is marked orange during analysis. When
its review comes up, the start point is already placed where the subject was
standing and you click only the **finish** point, then **Confirm**.

On the endpoint screen (also **☰ > Change rope endpoints**):
- While a point is missing, **click** to place it (the start first, then the
  finish).
- **Drag** the START or FINISH point to move it (it's ringed when the pointer
  is on it); the line follows as you drag.
- **Auto start point** (shown once the start has been moved): puts the start
  back where the subject was detected standing.
- **Confirm** (hover to see its shortcut, Enter) or **Cancel** (Esc).
- A start at the detected standing spot is timed from the subject's first
  foot movement; a start you place anywhere else is treated as a start line
  (timing starts when the subject crosses it, or at their first movement if
  they're already on or past it).

If the endpoints don't give both a start and an end time, you're told exactly
which is missing — **Walk start not found**, **Walk end not found**, or both —
and why (for the start, e.g. the walk starts too soon after the recording
begins, or the subject wasn't seen crossing the start line; for the end, no
foot was seen crossing the finish line), and asked straight away to
**Reselect points**, **Time manually** (the playback, where you mark what's
missing with the green / red dot) or **Skip this file**. The final confirmation
screen says the same, with the same choices, if you confirm a video whose
timing is incomplete, and the playback shows an orange note saying what to
mark. A common cause of a missing start: the subject is already walking when
the video begins, so there's no standing start — drag the start point onto
the start line.

If you cancel endpoint picking, the video still plays, with a note: set them
from the menu (**☰ > Change rope endpoints**) or skip the video there. A video
in which nobody was detected at all only offers **Skip this file**.

## Foot points

Every backend reports the same 35 landmarks (MediaPipe's 33 plus a small toe
per foot), saved as `lm_00` to `lm_34` in each CSV. Foot points, left / right:
ankle 27 / 28, heel 29 / 30, big toe 31 / 32, small toe 33 / 34. rtmlib and
mmpose use Halpe-26 "body with feet" models, which provide all of them (not the
face or hand detail); MediaPipe provides all but the small toes. If no toe is
seen crossing a line, the crossing falls back to the ankles, then the ankle
midpoint.

## Pose check

After pose estimation, the walking subject's leg and foot points are checked
for ones that can't be right (`tmwt/detection/pose_check.py`):

- **foot_length** — a heel or toe further from its ankle than 0.6 × the
  subject's leg length (hip → knee → ankle, median over the surrounding
  second). Real ankle-to-toe distances are about a fifth of the leg, so this
  only catches points thrown elsewhere (onto the floor, up to the waist).
- **spike** — a point that jumps more than 0.35 body heights away for one
  frame and back.

The first and last second of each video aren't checked (people stepping into
or out of frame, the camera being picked up or covered). On the 15 control
videos the check flags nothing; on the backlit old control it flags exactly
the frames where toes were thrown across the floor.

**During processing** (`process.py`, `label.py`): if any points are flagged,
the video is processed again with the backend's heavier model (rtmlib
`balanced` → `performance`: RTMPose-x at 384×288 with the YOLOX-x detector,
about 3× slower, ~75 ms a frame on an M1 Max), and that output is kept. The
model used in the end is recorded as `model_strength`, and the checks' results
as `pose_check` (`ok`, `fixed by heavier model`, `anomalies remain`), in the
analysis file. `tmwt_analysis/processing_report.csv` / `.md` list every video,
and any whose anomalies remain are also printed at the end of the run — no
questions are asked, so it runs unattended (e.g. under SLURM).

**At review**, opening a video whose anomalies remain (every time, including
a video reviewed before) shows a full-screen notice listing the flagged points
and the model used; **Review flagged points** then opens **flagged-points
mode**, paused on the first flagged frame:
- the flagged points (and their lines) are **orange**, the timeline is orange
  over those frames, and the info panel names the points (it lists the model
  strength for every video);
- **◀● / ●▶** (or `[` / `]`) jump to the previous / next flagged frame; the
  usual frame-step and play controls work too;
- **Smooth points** replaces the flagged points by interpolation (below); they
  turn **yellow**, on the timeline too. **Unsmooth** puts them back as
  detected;
- **Confirm** (Enter) records that you've checked them and goes on to the
  usual timing playback, then the review prompt.

The review prompt has **Review flagged points (orange)** (`6`) for these
videos, to go back to that mode at any time. Flagged points are left out of
the timing until smoothed (a stray toe can't trigger a line crossing), and
automatic timing is recomputed when you smooth or unsmooth. **Skip this file**
removes the video; if its flagged points weren't confirmed, the reason is
recorded as "pose detection anomalies".

**Smoothing** replaces each flagged point by linear interpolation from the
nearest frames (within 0.5 s) where that point wasn't flagged. Heels and toes
are interpolated relative to their ankle, then placed on that frame's ankle,
so the foot moves with the step; hips, knees and ankles are interpolated in
position. Points with no good frames near enough are left as they were (still
orange, with a note). Smoothing is saved with the review progress.

Pose data is only changed by smoothing the reviewer chose, and always
labelled: the CSV's `pose_flags` column says which points were flagged in each
frame (e.g. `left_small_toe:foot_length`), `pose_smoothed` which of them now
hold smoothed values, and `<basename>_pose_corrections.csv` lists every
smoothed point with its original and new position. Unsmoothed flagged points
stay in the landmark columns as detected, so the gait analysis can decide what
to do with them.

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
`--endpoint_behavior ankle_midpoint`. The start and finish lines are
**horizontal** in the image through their endpoints: on the floor they're at
right angles to the course, which is horizontal when filming from the end of
the course with the camera held level (like the tape at the finish). A body
point's position along the course (`t_along`, 0 at the start line, 1 at the
finish line) is read off where the horizontal line through it meets the rope,
so a foot to one side of a tilted rope crosses at the right moment. Positions
are smoothed before finding crossings, forwards and backwards, so the smoothing
doesn't delay them. When the far endpoint is the subject's standing
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

   **Recordings that start just before the walk.** The standstill is best at a
   second or more; start recording at least 2 s before "go". With less, the
   labeler uses whatever standstill there is (if necessary running into the
   first moments of the walk), which makes its noise estimate rough and the
   start tend to come out a little late (about 0.2 s on `SV_10MWRT_string`).
   Those starts are flagged on the review prompt ("Start found from very
   little standing still…") and in the timing JSON (`timing_note`), so they
   can be checked and the green mark dragged if needed. With no standstill at
   all, the start isn't found and the prompt says why.

Distances are perspective-corrected using the vanishing point of the subject's
own walk (see `tmwt/measurement/metric.py`). Ankle positions are projected onto the walking line
first, so sideways sway does not read as forward movement. The detection and its
tuning constants live in `tmwt/measurement/onset.py`; `tmwt/measurement/timing.py`
combines it with the line crossings.

When the far endpoint was clicked as a start line, the start is the moment the
ankles cross it if the subject was standing behind it. If they were already on or
past the line when they started moving, it's their first foot movement, as above.

## Output

For each saved video (`<basename>` = filename without extension), four files
are written to `--output_dir`. Videos rejected at review get no outputs.

| File                        | Contents                                                                                              |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| `<basename>.csv`            | Frame-by-frame body position, rope endpoints, position along the course (`t_along`: 0 at the start line, 1 at the finish line), 35 pose landmarks (raw), and `pose_flags` (see [Pose check](#pose-check)).|
| `<basename>_timing.json`    | The walk timing the labeler decided (start, end, duration, speed), how the start was found, the review outcome, the pose model used (`model_strength`) and the pose check's outcome (`pose_check`). `view.py` reads it. |
| `<basename>_annotated.mp4`  | Source frames with skeleton, rope, and info panel overlaid.                                           |
| `<basename>_skeleton.mp4`   | Black canvas with skeleton, rope, and info panel only — de-identified for sharing.                    |
| `<basename>_pose_corrections.csv` | Only if flagged pose points were smoothed at review: each one's frame, landmark, flag, and original and smoothed position. |

The info panel on both output videos shows the walk status and timer.

Each run also writes `labeling_report.csv` and `labeling_report.md` with one row
per video: its result (`approved`, `auto (not reviewed)`, `rejected`,
`failed`), the reason, whether the endpoints and timing were automatic or
manual, how the start was found, the start / end / duration / speed, the
pose model used, the pose check's outcome, and whether outputs were saved.

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
| `|◀`   | ←     | Step back one frame (pauses); hold ← to keep stepping back |
| `▶` / `‖` | Space | Play / pause |
| `▶|`   | →     | Step forward one frame (pauses); hold → to keep stepping forward |
| `▶▶|`  | N     | Next recording |

Above the buttons, a seek bar: drag it (or click on it) to scrub through the
recording. The walk's start and end are marked on it in green and red, and the
time is shown as elapsed / total.

At the end of a recording the next one plays; after the last, playback pauses on
its final frame. Close the window (or press Esc) to quit.

---

## Testing

```bash
.venv/bin/python -m unittest discover tests
```

About 600 unit tests (standard-library `unittest`, no extra packages) cover
every module that doesn't need a real window, a video device or a pose model:
geometry and timing, pose check and smoothing, people tracking, the ArUco
finish, processing (with a stand-in pose backend, including the heavier-model
retry), analysis files and outputs, review progress and the review flows (with
a stand-in window), and the window's widgets, panels, seek bar, player and
input handling (clicks, drags, keys, sidebar and top bar). They run in a couple
of seconds. The key-repeat tests run in a separate process, because pygame and
OpenCV bundle clashing copies of SDL2 on macOS.

Not unit-tested: the pygame window process itself and pose inference (they
need a display and the models); those are checked by running the labeler.

## Repository layout

The four scripts you run are at the top level; everything else is in the
`tmwt/` package, grouped by what it does.

```
process.py                # Step 1 (e.g. on a cluster): processing, writes tmwt_analysis/*.npz
review.py                 # Step 2 (locally): review from the analysis files
label.py                  # Both steps in one go, locally
view.py                   # Skeleton-only playback of saved CSVs

media/                    # Input videos (not tracked by git)
models/                   # MediaPipe model file (the default --model path)
assets/                   # ArUco marker specification; the top bar's UR shield and RNA rosette
tests/                    # Unit tests (see Testing below)
runs/                     # Output folders from earlier runs (data; not tracked by git)

tmwt/
  core/                   # Data model, video access and file outputs
    job.py                #   VideoJob (one per video) and FrameResult (one per frame)
    video_io.py           #   Opening videos consistently; in-order frame cache for playback
    matte.py              #   Letterbox / pillarbox detection and cropping
    analysis_file.py      #   The analysis file format: save, load, video fingerprint
    data_export.py        #   Per-video outputs (CSV, timing JSON, videos) and reading them back
    report.py             #   End-of-run labeling report
  pose/                   # Pose estimation
    pose_backend.py       #   Backend factory
    pose_common.py        #   Shared landmark layout (body + feet), drawing and ankle helpers
    pose_rtmlib.py        #   RTMLib (ONNX Runtime) backend
    pose_mediapipe.py     #   MediaPipe backend
    pose_mmpose.py        #   MMPose backend
  detection/              # The analysis
    analysis.py           #   Slow part (pose + tracking per frame) and fast part (subject, endpoints, timing)
    processing.py         #   Processing a folder: skip / process / save analysis files
    tracking.py           #   Ground-plane optical-flow tracker (camera drift)
    people.py             #   Following everyone in view and choosing the walking subject
    pose_check.py         #   Flagging implausible leg / foot points (pose check)
    pose_smoothing.py     #   Replacing flagged points by interpolation, at review
    endpoints.py          #   Finish line from the ArUco marker
  measurement/            # Turning positions into timing
    metric.py             #   Geometry: t_along and perspective-correct distance along the course
    onset.py              #   Hindsight walk-start detection on distance signals
    timing.py             #   Walk start / end from the analysed frames
  session/                # The review
    review.py             #   Reviewing one video: the flow, the menu, the final confirmation
    review_playback.py    #   The review's playback: marks, Confirm, menu button
    flagged_points.py     #   The pose-anomaly notice and flagged-points mode (smooth / unsmooth)
    review_session.py     #   Loading, reviewing, saving and reporting a folder
    review_progress.py    #   Saving / restoring an unfinished review
  ui/                     # The window and what's drawn in it
    window.py             #   The window as the program sees it: show a canvas, poll for input
    window_server.py      #   The window process: resizable pygame window
    base_window.py        #   BaseWindow: the window, its panels and input handling
    pickers.py            #   PickerScreens: picking the rope endpoints and the walker
    labeler_ui.py         #   LabelerUI(PickerScreens, BaseWindow): progress, playback frame, menus
    events.py             #   Leaving a screen: JumpTo, and the UserQuit family (window closed, save / quit)
    panel.py              #   Panel: a region of the window (base of the top bar and sidebar)
    top_bar.py            #   Top bar: quit and save buttons, folder title, logos
    sidebar.py            #   Sidebar: the colour-coded file list, its states and marks
    widgets.py            #   Colours, key codes, buttons (Button and subclasses), icons, tooltips, badges
    seek_bar.py           #   SeekBar: scrubbing and draggable marks
    player.py             #   Playback controls shared by the review and the viewer
    annotate.py           #   Frame drawing (skeleton, rope, info panel)
```
