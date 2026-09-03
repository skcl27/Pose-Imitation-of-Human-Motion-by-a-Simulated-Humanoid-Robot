"""Skeleton overlay visualizer for live camera feed."""
from __future__ import annotations

import logging
from collections.abc import Iterable

import cv2
import numpy as np

from src.perception.landmarks import POSE_CONNECTIONS, POSE_LANDMARKS
from src.perception.metrabs_model import project_point
from src.type_defs import PoseFrame

logger = logging.getLogger(__name__)

LANDMARK_COLOR = (0, 200, 255)   # cyan-orange landmarks
SKELETON_COLOR = (255, 255, 255) # white bones
HUD_COLOR = (50, 220, 50)        # green HUD
LOW_VIS_THRESHOLD = 0.3          # Lowered threshold to show more detected landmarks


class SkeletonOverlay:
    """Draws keypoints, bones, and HUD onto a BGR frame.

    Keypoints are now 3D (mm, camera frame) rather than normalized [0,1] image
    coordinates, so drawing them requires projecting back to pixels with the
    same pinhole intrinsic matrix the estimator used for inference (see
    ``PoseEstimator.intrinsics_for``), passed into :meth:`draw`.
    """

    def __init__(
        self,
        window_name: str = "Pose Imitation - Camera Feed",
        show: bool = True,
    ) -> None:
        self.window_name = window_name
        self.show = show
        self._window_created = False

    def _ensure_window(self) -> None:
        if self.show and not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

    def draw(
        self,
        frame_bgr: np.ndarray,
        pose: PoseFrame,
        intrinsics: np.ndarray,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        extra_hud: Iterable[str] = (),
    ) -> np.ndarray:
        canvas = frame_bgr.copy()
        keypoints = pose.keypoints

        pixels = {
            name: project_point(
                np.array([kp.x, kp.y, kp.z], dtype=np.float32), intrinsics
            )
            for name, kp in keypoints.items()
        }

        # Draw bones
        for a_name, b_name in POSE_CONNECTIONS:
            a = keypoints.get(a_name)
            b = keypoints.get(b_name)
            if a is None or b is None:
                continue
            if a.visibility < LOW_VIS_THRESHOLD or b.visibility < LOW_VIS_THRESHOLD:
                continue
            pa = (int(pixels[a_name][0]), int(pixels[a_name][1]))
            pb = (int(pixels[b_name][0]), int(pixels[b_name][1]))
            cv2.line(canvas, pa, pb, SKELETON_COLOR, 2, cv2.LINE_AA)

        # Draw landmarks
        for name in POSE_LANDMARKS:
            kp = keypoints.get(name)
            if kp is None or kp.visibility < LOW_VIS_THRESHOLD:
                continue
            cx, cy = int(pixels[name][0]), int(pixels[name][1])
            cv2.circle(canvas, (cx, cy), 4, LANDMARK_COLOR, -1, cv2.LINE_AA)

        # HUD
        self._draw_hud(canvas, pose, fps, latency_ms, extra_hud)
        return canvas

    def _draw_hud(
        self,
        canvas: np.ndarray,
        pose: PoseFrame,
        fps: float,
        latency_ms: float,
        extra_hud: Iterable[str],
    ) -> None:
        detected = sum(1 for kp in pose.keypoints.values() if kp.visibility >= LOW_VIS_THRESHOLD)
        total = len(POSE_LANDMARKS)
        # Consider human detected if we see 20% or more of the body
        has_human = detected >= total * 0.2 and len(pose.keypoints) > 0
        status = "✓ HUMAN DETECTED" if has_human else "✗ NO HUMAN DETECTED"
        lines = [
            f"FPS: {fps:5.1f}   Latency: {latency_ms:5.1f} ms",
            f"Frame: {pose.frame_index}   Landmarks: {detected}/{total}",
            f"Status: {status}",
            *extra_hud,
        ]
        y = 28
        for line in lines:
            cv2.putText(
                canvas, line, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 2, cv2.LINE_AA,
            )
            y += 24

    def show_frame(self, canvas: np.ndarray) -> bool:
        """Display the canvas. Returns False if the user requested exit."""
        if not self.show:
            return True
        self._ensure_window()
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))  # ESC or 'q' quits

    def close(self) -> None:
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
