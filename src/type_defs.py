from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoint:
    """One 3D joint sample from the MeTRAbs pose estimator.

    ``x``/``y``/``z`` are absolute METRIC coordinates in MILLIMETERS, in the
    camera's coordinate frame: x right, y down, z forward/away from the
    camera (a monocular pinhole-camera convention -- see
    ``src/perception/metrabs_model.py``). This replaces the previous
    MediaPipe-era convention of normalized [0,1] image coordinates plus a
    weak, unreliable depth channel.

    ``visibility`` is a PROXY in [0, 1] (in-frame/in-detection-box confidence
    x detection score), not a true per-joint occlusion estimate -- MeTRAbs
    does not provide one. See ``PoseEstimator._visibility_proxy``.
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass(frozen=True)
class PoseFrame:
    timestamp_s: float
    keypoints: dict[str, Keypoint]
    frame_index: int


@dataclass(frozen=True)
class JointCommand:
    timestamp_s: float
    joint_angles_rad: dict[str, float]
    frame_index: int
