"""Loading and introspection helpers for the MeTRAbs TensorFlow-Hub model.

Kept separate from :mod:`pose_estimator` so that other modules (landmark
naming, the skeleton overlay) can ask "what joints/edges does this skeleton
have" without duplicating model-loading logic or forcing a GPU check just to
read metadata.

MeTRAbs ships no PyPI package: models are TensorFlow SavedModels distributed
via TensorFlow-Hub / a plain URL to a zipped SavedModel directory, loaded with
``tensorflow_hub.load(...)``. See https://github.com/isarandi/metrabs,
docs/API.md and docs/INFERENCE.md there for the authoritative reference.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# EfficientNetV2-S backbone: a reasonable default speed/accuracy trade-off for
# real-time use. Swap for a larger (``eff2l``, more accurate) or smaller
# (``mob3s``/``rn18``, faster) backbone via config -- see docs/MODELS_6_DATASETS.md
# in the metrabs repo for the full table and relative accuracy/speed.
DEFAULT_MODEL_URL = "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4.zip"
DEFAULT_SKELETON = "coco_19"


class MetrabsUnavailableError(RuntimeError):
    """Raised when TensorFlow/TensorFlow-Hub, the model, or a GPU is unavailable."""


@dataclass(frozen=True)
class SkeletonInfo:
    names: Tuple[str, ...]
    edges: Tuple[Tuple[int, int], ...]


def require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise MetrabsUnavailableError(
            "TensorFlow is not installed. Install a GPU-enabled build matching "
            "the target machine's CUDA/cuDNN driver version, plus "
            "tensorflow-hub:\n    pip install tensorflow tensorflow-hub\n"
            f"Original error: {exc}"
        ) from exc
    return tf


def require_gpu() -> None:
    """Raise loudly if TensorFlow cannot see a GPU.

    MeTRAbs is a full detector + 3D-pose network; on CPU it is far too slow for
    the pipeline's real-time budget (see ``runtime.latency_budget_ms`` in
    configs/default.yaml). Rather than silently degrading to CPU, fail clearly
    -- same philosophy as the old MediaPipe-era ``PoseEstimatorError``.
    """
    tf = require_tensorflow()
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise MetrabsUnavailableError(
            "No GPU visible to TensorFlow (tf.config.list_physical_devices('GPU') "
            "returned []). MeTRAbs needs a GPU for real-time inference. Either "
            "run this on a machine with a working CUDA-enabled TensorFlow "
            "install, or set pose.allow_synthetic_fallback: true in the config "
            "to run without real pose tracking."
        )
    logger.info("TensorFlow sees %d GPU(s): %s", len(gpus), [g.name for g in gpus])


@lru_cache(maxsize=4)
def load_model(model_url: str):
    """Load (and cache) a MeTRAbs SavedModel via TensorFlow-Hub.

    ``model_url`` may be the URL of a zipped SavedModel (as in
    ``DEFAULT_MODEL_URL``) or a local path to an already-extracted SavedModel
    directory. The first load downloads and extracts into TFHub's local cache
    (``$TFHUB_CACHE_DIR``, defaults to a temp dir) -- subsequent runs reuse it.
    """
    tfhub = _import_tfhub()
    logger.info("Loading MeTRAbs model from %s (first call may download it) ...", model_url)
    model = tfhub.load(model_url)
    logger.info("MeTRAbs model loaded.")
    return model


def _import_tfhub():
    require_tensorflow()
    try:
        import tensorflow_hub as tfhub  # type: ignore
    except ImportError as exc:
        raise MetrabsUnavailableError(
            "tensorflow-hub is not installed:\n    pip install tensorflow-hub\n"
            f"Original error: {exc}"
        ) from exc
    return tfhub


@lru_cache(maxsize=8)
def skeleton_info(model_url: str, skeleton: str) -> SkeletonInfo:
    """Joint names and kinematic-tree edges for ``skeleton``, read live from
    the model rather than hard-coded -- MeTRAbs exposes this via the
    ``per_skeleton_joint_names`` / ``per_skeleton_joint_edges`` attributes
    (see docs/API.md). Names are decoded to plain ``str``.
    """
    model = load_model(model_url)
    names = tuple(
        n.decode("utf-8") if isinstance(n, bytes) else str(n)
        for n in model.per_skeleton_joint_names[skeleton].numpy()
    )
    edges = tuple(
        (int(a), int(b)) for a, b in model.per_skeleton_joint_edges[skeleton].numpy()
    )
    return SkeletonInfo(names=names, edges=edges)


def intrinsic_matrix(width: int, height: int, fov_degrees: float) -> np.ndarray:
    """Pinhole camera intrinsic matrix for an uncalibrated camera.

    ``fov_degrees`` is the field of view along the LARGER image side, matching
    MeTRAbs' own ``default_fov_degrees`` convention (see docs/API.md), so the
    same value can be passed to both this function and ``model.detect_poses``.
    The principal point is assumed to be the image center. This is a rough
    approximation in the absence of real camera calibration -- fine for the
    angle-based retargeting math (the dominant use), less exact for absolute
    scale.
    """
    larger_side = max(width, height)
    focal = 0.5 * larger_side / math.tan(math.radians(fov_degrees) / 2.0)
    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def project_point(point_mm: np.ndarray, intrinsics: np.ndarray) -> Tuple[float, float]:
    """Project one camera-frame 3D point (mm) to pixel coordinates."""
    x, y, z = float(point_mm[0]), float(point_mm[1]), float(point_mm[2])
    z = z if abs(z) > 1e-6 else 1e-6
    px = intrinsics[0, 0] * x / z + intrinsics[0, 2]
    py = intrinsics[1, 1] * y / z + intrinsics[1, 2]
    return px, py
