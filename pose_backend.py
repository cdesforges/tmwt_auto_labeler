"""
Pose backend factory.

Returns the module implementing the pose API for a given backend name.
All backends expose the same public interface (see pose.py for reference):

  DEFAULT_MODEL_PATH: str
  POSE_CONNECTIONS: list[tuple[int, int]]
  LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX, NOSE_IDX: int
  FACE_IDXS: set[int]

  create_landmarker(model_path, num_poses=1) -> landmarker
  create_image_landmarker(model_path, num_poses=1) -> landmarker
  detect_poses(landmarker, frame_bgr, timestamp_ms) -> list[Pose]
  detect_poses_image(landmarker, frame_bgr) -> list[Pose]
  draw_pose(frame_bgr, pose_landmarks, ...) -> None
  get_ankle_midpoint(pose_landmarks, frame_shape) -> (x, y) | None
"""


_BACKENDS = ("mediapipe", "mmpose", "rtmlib")


def get_backend(name):
    """
    Return the pose backend module for `name`.

    Args:
        name: one of "mediapipe", "mmpose", "rtmlib".

    Raises:
        ValueError: if `name` is not a known backend.
        ImportError: if the backend module's dependencies are missing.
    """
    if name == "mediapipe":
        import pose
        return pose
    if name == "mmpose":
        import pose_mmpose
        return pose_mmpose
    if name == "rtmlib":
        import pose_rtmlib
        return pose_rtmlib
    raise ValueError(
        f"Unknown pose backend: {name!r}. Use one of {_BACKENDS}."
    )
