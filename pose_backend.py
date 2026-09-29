"""
Pose backend factory.

Each backend module exposes the same small inference API:

  DEFAULT_MODEL_PATH: str
  create_landmarker(model_path, num_poses=1) -> landmarker   (streaming frames)
  create_image_landmarker(model_path, num_poses=1) -> landmarker   (single images)
  detect_poses(landmarker, frame_bgr, timestamp_ms) -> list[pose]
  detect_poses_image(landmarker, frame_bgr) -> list[pose]

and every landmarker has close(). Backends also have set_device(device) — one of
"auto", "cpu", "cuda", "mps" — and provenance() -> {"device", "versions"}. Poses are lists of landmarks in the layout
described in pose_common.py (including the feet), which also holds the drawing
and ankle helpers.

Creating landmarkers is cheap to repeat: rtmlib and mmpose load each model once
per process and share it, and MediaPipe models load quickly (its VIDEO-mode
landmarker tracks between frames, so a fresh one is needed per video anyway).
"""

BACKENDS = ("mediapipe", "mmpose", "rtmlib")
DEVICES = ("auto", "cpu", "cuda", "mps")


def get_backend(name):
    """
    Return the pose backend module for `name`. Backends are imported lazily, so
    only the chosen one's dependencies need to be installed.

    Raises:
        ValueError: if `name` is not a known backend.
        ImportError: if the backend module's dependencies are missing.
    """
    if name == "mediapipe":
        import pose_mediapipe
        return pose_mediapipe
    if name == "mmpose":
        import pose_mmpose
        return pose_mmpose
    if name == "rtmlib":
        import pose_rtmlib
        return pose_rtmlib
    raise ValueError(f"Unknown pose backend: {name!r}. Use one of {BACKENDS}.")
