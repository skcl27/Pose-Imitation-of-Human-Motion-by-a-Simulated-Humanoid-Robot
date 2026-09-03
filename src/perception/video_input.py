"""Cross-platform video capture wrapper.

Selects the best OpenCV backend per OS (V4L2 on Linux, AVFoundation on macOS,
DirectShow on Windows) and falls back to the default backend if needed.
"""
from __future__ import annotations

import logging
import platform
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SourceType = int | str


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_s: float
    image_bgr: np.ndarray


def _preferred_backend() -> int:
    system = platform.system().lower()
    if system == "linux":
        return cv2.CAP_V4L2
    if system == "darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class VideoCaptureError(RuntimeError):
    """Raised when the camera or video file cannot be opened."""


class VideoSource:
    """Robust OpenCV-backed video source with platform-aware backend selection."""

    def __init__(
        self,
        source: SourceType,
        width: int = 1280,
        height: int = 720,
        preferred_fps: float = 30.0,
    ) -> None:
        self.source = source
        backend = _preferred_backend()
        logger.info("Opening video source %r with backend=%d", source, backend)

        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            logger.warning("Preferred backend failed; retrying with CAP_ANY")
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise VideoCaptureError(
                f"Cannot open video source {source!r}. "
                "Check camera index, device permissions, or file path."
            )

        # Configure capture parameters; some webcams ignore these silently.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, preferred_fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency where supported
        except cv2.error:
            pass

        logger.info(
            "Capture opened: %dx%d @ %.1f FPS",
            self.width, self.height, self.cap.get(cv2.CAP_PROP_FPS),
        )

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_loop(
        self,
        target_period_s: float | Callable[[], float] = 0.0,
    ) -> Generator[VideoFrame, None, None]:
        """Yield frames until the source ends or too many failures occur.

        ``target_period_s`` may be a number or a **callable** returning the
        current period. The callable form is what makes adaptive frame-rate
        control work at all: the pipeline used to pass
        ``fps_controller.target_period_s``, which is a property, so the period was
        read once when this generator was constructed and never again. The
        controller went on measuring latency and adjusting its target for the
        whole run while the loop slept to a constant period -- four config keys
        (min_fps, max_fps, fps_step, latency_budget_ms) had no effect on anything,
        and the HUD reported a target FPS that was not being applied.
        """
        period_of = (
            target_period_s if callable(target_period_s) else (lambda: target_period_s)
        )
        frame_index = 0
        consecutive_failures = 0
        max_failures = 30

        while True:
            start = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive read failures; stopping capture.")
                    break
                time.sleep(0.01)
                continue
            consecutive_failures = 0

            yield VideoFrame(
                frame_index=frame_index,
                timestamp_s=time.time(),
                image_bgr=frame,
            )
            frame_index += 1

            period = period_of()
            if period > 0:
                elapsed = time.perf_counter() - start
                sleep_s = period - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
