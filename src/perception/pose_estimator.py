"""MeTRAbs-based human pose estimator.

Design goals (unchanged from the MediaPipe-era version this replaces):
- Loud, explicit failures (no silent fallback) when MeTRAbs cannot load or no
  GPU is visible.
- Optional synthetic fallback only when explicitly enabled in config.
- Returns a canonical `PoseFrame` per frame, with a visibility PROXY per
  landmark (see below -- MeTRAbs has no true per-joint confidence).

What changed vs. MediaPipe
---------------------------
MediaPipe returned normalized [0,1] image coordinates plus a weak, unreliable
depth channel. MeTRAbs returns absolute METRIC 3D coordinates in MILLIMETERS,
in the camera's coordinate frame (x right, y down, z forward/away from the
camera) -- see ``metrabs_model.py`` and the upstream docs/API.md. Downstream
code (retargeting, gait cues) now works in real 3D instead of reconstructing
it from a 2D projection.

MeTRAbs also has no per-joint visibility/confidence output (only one
detection-box confidence per person). ``visibility`` here is therefore a
PROXY: 1.0 for a joint whose 2D projection lands well inside the detected
person's box and the image frame, decaying to 0.0 near the image edge or
outside the box, and 0.0 everywhere when no person was detected at all. It is
not a measure of true occlusion -- flagged explicitly so downstream
`visibility >= threshold` gating (nao_retarget.py, gait_cues.py, mapper.py) is
understood to be an approximation, not truth.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.perception import metrabs_model
from src.perception.landmarks import POSE_LANDMARKS, build_raw_to_canonical_map
from src.type_defs import Keypoint, PoseFrame

logger = logging.getLogger(__name__)

# How close (in pixels) to the image/box edge a joint's 2D projection must be
# before its visibility proxy starts decaying to 0.
_EDGE_MARGIN_FRAC = 0.06


class PoseEstimatorError(RuntimeError):
    """Raised when MeTRAbs cannot be initialised and fallback is disabled."""


@dataclass
class PoseEstimator:
    """Wraps the MeTRAbs TF-Hub model with GPU checking and robust logging."""

    use_metrabs: bool = True
    model_url: str = metrabs_model.DEFAULT_MODEL_URL
    skeleton: str = metrabs_model.DEFAULT_SKELETON
    default_fov_degrees: float = 55.0
    detector_threshold: float = 0.3
    num_aug: int = 1
    max_detections: int = 1
    require_gpu: bool = True
    allow_synthetic_fallback: bool = False

    _model: object = field(default=None, init=False, repr=False)
    _tf: object = field(default=None, init=False, repr=False)
    _raw_names: List[str] = field(default_factory=list, init=False, repr=False)
    _raw_to_canonical: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _intrinsics_cache: Dict[Tuple[int, int], np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._model = None

        if not self.use_metrabs:
            self._warn_about_fallback("pose.use_metrabs is False")
            return

        try:
            if self.require_gpu:
                metrabs_model.require_gpu()
            self._tf = metrabs_model.require_tensorflow()
            self._model = metrabs_model.load_model(self.model_url)
            info = metrabs_model.skeleton_info(self.model_url, self.skeleton)
        except metrabs_model.MetrabsUnavailableError as exc:
            if self.allow_synthetic_fallback:
                logger.error(str(exc))
                self._warn_about_fallback("MeTRAbs unavailable")
                self._model = None
                return
            raise PoseEstimatorError(str(exc)) from exc

        self._raw_names = list(info.names)
        self._raw_to_canonical = build_raw_to_canonical_map(self._raw_names)
        matched_canonical = set(self._raw_to_canonical.values())
        missing = [n for n in POSE_LANDMARKS if n not in matched_canonical]
        if missing:
            msg = (
                f"MeTRAbs skeleton '{self.skeleton}' joint names did not match "
                f"{len(missing)} expected canonical landmark(s): {missing}. "
                f"Raw model joint names were: {self._raw_names}. "
                "src/perception/landmarks.py's CANONICAL_TO_RAW_ALIASES was "
                "written without access to a live model and needs correcting -- "
                "run scripts/inspect_metrabs_skeleton.py and update the alias "
                "table to match the raw names printed above."
            )
            if self.allow_synthetic_fallback:
                logger.error(msg)
                self._warn_about_fallback("landmark name mismatch")
                self._model = None
                return
            raise PoseEstimatorError(msg)

        logger.info(
            "MeTRAbs model initialised (skeleton=%s, %d joints, detector_threshold=%.2f).",
            self.skeleton, len(self._raw_names), self.detector_threshold,
        )

    # ------------------------------------------------------------------ public

    @property
    def is_real(self) -> bool:
        """True if the real MeTRAbs model is active (not the synthetic fallback)."""
        return self._model is not None

    def intrinsics_for(self, width: int, height: int) -> np.ndarray:
        """Pinhole intrinsic matrix used for this estimator's frames, cached
        per resolution. Exposed so the visualizer can project 3D points back
        to pixels for the overlay.
        """
        key = (width, height)
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = metrabs_model.intrinsic_matrix(
                width, height, self.default_fov_degrees
            )
        return self._intrinsics_cache[key]

    def estimate(
        self,
        image_bgr: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> PoseFrame:
        if self._model is None:
            return self._estimate_fallback(image_bgr, timestamp_s, frame_index)

        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self._tf.convert_to_tensor(rgb, dtype=self._tf.uint8)

        pred = self._model.detect_poses(
            image_tensor,
            default_fov_degrees=self.default_fov_degrees,
            detector_threshold=self.detector_threshold,
            max_detections=self.max_detections,
            num_aug=self.num_aug,
            skeleton=self.skeleton,
        )
        boxes = pred["boxes"].numpy()
        if boxes.shape[0] == 0:
            logger.debug("Frame %d: No human detected by MeTRAbs", frame_index)
            return PoseFrame(timestamp_s=timestamp_s, keypoints={}, frame_index=frame_index)

        poses3d = pred["poses3d"].numpy()[0]  # [num_joints, 3] mm, camera frame
        poses2d = pred["poses2d"].numpy()[0]  # [num_joints, 2] pixels
        box = boxes[0]  # [left, top, width, height, confidence]

        keypoints: Dict[str, Keypoint] = {}
        for j, raw_name in enumerate(self._raw_names):
            canonical = self._raw_to_canonical.get(raw_name)
            if canonical is None:
                continue
            vis = self._visibility_proxy(poses2d[j], box, width, height)
            x, y, z = poses3d[j]
            keypoints[canonical] = Keypoint(
                x=float(x), y=float(y), z=float(z), visibility=vis
            )

        visible_count = sum(1 for kp in keypoints.values() if kp.visibility > 0.3)
        logger.debug(
            "Frame %d: Human detected (box conf %.2f) with %d/%d visible landmarks (>0.3 threshold)",
            frame_index, float(box[4]), visible_count, len(keypoints),
        )
        return PoseFrame(timestamp_s=timestamp_s, keypoints=keypoints, frame_index=frame_index)

    def close(self) -> None:
        # The TF-Hub model has no explicit teardown; nothing to release.
        pass

    # ----------------------------------------------------------------- helpers

    def _visibility_proxy(
        self, px_py: np.ndarray, box: np.ndarray, width: int, height: int
    ) -> float:
        """1.0 well inside the image and detection box, decaying to 0.0 near
        the edges. See module docstring -- this is a stand-in for MediaPipe's
        real per-joint visibility, which MeTRAbs does not provide."""
        px, py = float(px_py[0]), float(px_py[1])
        left, top, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        margin = _EDGE_MARGIN_FRAC * max(width, height)

        def edge_score(pos: float, lo: float, hi: float) -> float:
            if pos < lo or pos > hi:
                return 0.0
            dist = min(pos - lo, hi - pos)
            return max(0.0, min(1.0, dist / max(margin, 1e-6)))

        image_score = min(
            edge_score(px, 0.0, width), edge_score(py, 0.0, height)
        )
        box_score = min(
            edge_score(px, left, left + w), edge_score(py, top, top + h)
        )
        return min(image_score, box_score) * float(box[4])

    def _warn_about_fallback(self, reason: str) -> None:
        logger.warning(
            "Using SYNTHETIC pose fallback (%s). The skeleton will NOT follow the "
            "human; fix MeTRAbs/GPU setup to enable real pose tracking.",
            reason,
        )

    def _estimate_fallback(
        self,
        image_bgr: np.ndarray,
        timestamp_s: float,
        frame_index: int,
    ) -> PoseFrame:
        """Synthetic deterministic pose generator for environments without a
        working MeTRAbs/GPU setup. Coordinates are in the same mm/camera-frame
        convention real frames use (a person ~1.7m tall standing ~2m from the
        camera), so downstream retargeting math stays consistent."""
        t = frame_index / 20.0
        swing = 120.0 * math.sin(t)  # mm
        z0 = 2000.0  # mm from camera

        def kp(x: float, y: float, z: float = z0) -> Keypoint:
            return Keypoint(x=x, y=y, z=z, visibility=1.0)

        base: Dict[str, Keypoint] = {
            "nose":            kp(0.0, -750.0),
            "left_eye":        kp(-20.0, -770.0),
            "right_eye":       kp(20.0, -770.0),
            "left_ear":        kp(-60.0, -750.0),
            "right_ear":       kp(60.0, -750.0),
            "neck":            kp(0.0, -600.0),
            "left_shoulder":   kp(-160.0, -580.0),
            "right_shoulder":  kp(160.0, -580.0),
            "left_elbow":      kp(-220.0, -300.0 + swing),
            "right_elbow":     kp(220.0, -300.0 - swing),
            "left_wrist":      kp(-260.0, -60.0 + swing * 1.3),
            "right_wrist":     kp(260.0, -60.0 - swing * 1.3),
            "pelvis":          kp(0.0, 0.0),
            "left_hip":        kp(-110.0, 20.0),
            "right_hip":       kp(110.0, 20.0),
            "left_knee":       kp(-115.0, 430.0),
            "right_knee":      kp(115.0, 430.0),
            "left_ankle":      kp(-120.0, 830.0),
            "right_ankle":     kp(120.0, 830.0),
        }
        return PoseFrame(timestamp_s=timestamp_s, keypoints=base, frame_index=frame_index)
