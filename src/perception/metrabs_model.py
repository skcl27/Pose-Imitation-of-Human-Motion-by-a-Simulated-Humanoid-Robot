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
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# EfficientNetV2-S backbone: a reasonable default speed/accuracy trade-off for
# real-time use. Swap for a larger (``eff2l``, more accurate) or smaller
# (``mob3s``/``rn18``, faster) backbone via config -- see docs/MODELS_6_DATASETS.md
# in the metrabs repo for the full table and relative accuracy/speed.
DEFAULT_MODEL_URL = "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2s_y4.zip"
DEFAULT_SKELETON = "coco_19"

# Where downloaded/extracted SavedModels live. A persistent, per-user location
# on purpose: the model is ~320 MB compressed and the default TF-Hub cache is a
# TEMP directory, so every reboot silently threw the download away (NFR-5).
# Override with $METRABS_CACHE_DIR.
DEFAULT_CACHE_DIR = Path(
    os.environ.get("METRABS_CACHE_DIR", Path.home() / ".cache" / "metrabs")
)


class MetrabsUnavailableError(RuntimeError):
    """Raised when TensorFlow/TensorFlow-Hub, the model, or a GPU is unavailable."""


@dataclass(frozen=True)
class SkeletonInfo:
    names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]


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


def enable_memory_growth() -> None:
    """Stop TensorFlow claiming every byte of VRAM at startup.

    TF's default is to reserve the whole card. Measured here that is 22.6 GB of
    the 3090 Ti for a model that needs a fraction of it, which starves every
    other process on the machine: a second tool run alongside the pipeline got
    971 MB and could not load the same model. Growth must be requested BEFORE
    any op initialises the device, so this runs ahead of the placement probe
    below and ahead of ``load_model``.
    """
    tf = require_tensorflow()
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            # Only raised when the device is already initialised, in which case
            # the allocator is already fixed for this process -- not fatal.
            logger.debug("Memory growth unavailable for %s: %s", gpu.name, exc)


def require_gpu() -> None:
    """Raise loudly unless TensorFlow can actually EXECUTE on a GPU.

    MeTRAbs is a full detector + 3D-pose network; on CPU it is far too slow for
    the pipeline's real-time budget (see ``runtime.latency_budget_ms`` in
    configs/default.yaml). Rather than silently degrading to CPU, fail clearly
    -- same philosophy as the old MediaPipe-era ``PoseEstimatorError``.

    Visibility is NOT execution. ``list_physical_devices('GPU')`` returning a
    card only means the driver enumerated it; TF still silently places ops on
    CPU when the CUDA/cuDNN build does not match the driver, which is precisely
    the "why is it so slow" failure this guard exists to prevent. So this runs a
    real op and checks where it landed.
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

    enable_memory_growth()

    try:
        with tf.device("/GPU:0"):
            probe = tf.linalg.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
        placed_on = probe.device
    except Exception as exc:  # noqa: BLE001 - any failure here means "no usable GPU"
        raise MetrabsUnavailableError(
            f"TensorFlow lists {len(gpus)} GPU(s) but could not run an op on "
            f"one: {exc}. This is normally a CUDA/cuDNN version mismatch with "
            "the installed driver. Fix the TensorFlow build, or set "
            "pose.allow_synthetic_fallback: true to run without pose tracking."
        ) from exc

    if "GPU" not in placed_on.upper():
        raise MetrabsUnavailableError(
            "TensorFlow lists a GPU but placed a test op on "
            f"'{placed_on}' -- inference would silently run on CPU, far too "
            "slow for the real-time budget. This is normally a CUDA/cuDNN "
            "version mismatch with the installed driver."
        )

    logger.info(
        "TensorFlow sees %d GPU(s): %s; test op executed on %s.",
        len(gpus), [g.name for g in gpus], placed_on,
    )


def _saved_model_root(path: Path) -> Path:
    """Descend to the directory that actually contains ``saved_model.pb``.

    MeTRAbs' release zips wrap the SavedModel in a directory of the same name,
    so extracting ``metrabs_eff2s_y4.zip`` yields
    ``metrabs_eff2s_y4/saved_model.pb`` rather than ``saved_model.pb`` at the
    root. ``tfhub.load`` looks only at the root and rejects the result with the
    thoroughly misleading "does not appear to be a valid module" -- which is why
    the configured ``DEFAULT_MODEL_URL`` could not be loaded at all, and why the
    model had to be extracted by hand to make the pipeline run.
    """
    if (path / "saved_model.pb").is_file():
        return path
    children = [c for c in path.iterdir() if c.is_dir()]
    if len(children) == 1:
        return _saved_model_root(children[0])
    raise MetrabsUnavailableError(
        f"No saved_model.pb found under {path}. The download may be truncated; "
        f"delete it and retry, or point pose.model_url at a SavedModel directory."
    )


def _download_and_extract(url: str, dest: Path) -> Path:
    """Fetch a zipped SavedModel into ``dest`` and return its SavedModel root.

    Extraction goes to a sibling temp directory and is renamed into place only
    once complete, so an interrupted download can never leave a half-unpacked
    tree that later runs would happily load.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        archive = staging / "model.zip"
        logger.info("Downloading MeTRAbs model from %s (~320 MB, one time) ...", url)
        with urllib.request.urlopen(url) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        logger.info("Extracting to %s ...", dest)
        unpacked = staging / "unpacked"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(unpacked)
        archive.unlink()
        # Normalise the layout NOW so the cache always holds a directly-loadable
        # SavedModel, whatever nesting the archive happened to use.
        root = _saved_model_root(unpacked)
        root.rename(dest)
        return dest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def resolve_model_path(model_url: str, cache_dir: Path | None = None) -> str:
    """Turn a model URL or local path into a loadable SavedModel directory.

    Local paths are used as-is (descending into a nested SavedModel if needed);
    ``.zip`` URLs are downloaded once into :data:`DEFAULT_CACHE_DIR` and reused
    on every later run. Anything else (a real TF-Hub handle) is passed straight
    through to ``tfhub.load``.
    """
    local = Path(model_url).expanduser()
    if local.is_dir():
        return str(_saved_model_root(local))
    if not model_url.endswith(".zip"):
        return model_url

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    dest = cache_dir / Path(model_url).stem
    if dest.is_dir():
        return str(_saved_model_root(dest))
    return str(_download_and_extract(model_url, dest))


@lru_cache(maxsize=4)
def load_model(model_url: str):
    """Load (and cache) a MeTRAbs SavedModel.

    ``model_url`` may be the URL of a zipped SavedModel (as in
    ``DEFAULT_MODEL_URL``), a local path to an already-extracted SavedModel
    directory, or a TF-Hub handle. Zips are resolved by
    :func:`resolve_model_path` rather than by TF-Hub's own resolver -- see
    :func:`_saved_model_root` for why TF-Hub cannot load MeTRAbs' archives.
    """
    tfhub = _import_tfhub()
    path = resolve_model_path(model_url)
    logger.info("Loading MeTRAbs model from %s ...", path)
    model = tfhub.load(path)
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


def project_point(point_mm: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    """Project one camera-frame 3D point (mm) to pixel coordinates."""
    x, y, z = float(point_mm[0]), float(point_mm[1]), float(point_mm[2])
    z = z if abs(z) > 1e-6 else 1e-6
    px = intrinsics[0, 0] * x / z + intrinsics[0, 2]
    py = intrinsics[1, 1] * y / z + intrinsics[1, 2]
    return px, py
