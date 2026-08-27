"""MeTRAbs ``coco_19`` skeleton landmark definitions.

Reference: https://github.com/isarandi/metrabs, docs/API.md ("Skeleton
Conventions"). ``coco_19`` was picked because it is the closest existing
MeTRAbs convention to the joint set this project actually uses (shoulders,
elbows, wrists, hips, knees, ankles, face points) without SMPL's extra
spine/collar joints or hand joints we have no use for.

IMPORTANT -- this list is a best-effort match of MeTRAbs' own naming, written
without access to a live model (no GPU on the machine this was written on).
It is NOT hard-relied upon for correctness: at runtime, ``pose_estimator.py``
reads the actual joint names for the loaded model
(``metrabs_model.skeleton_info(...)``) and normalizes them against
``CANONICAL_TO_RAW_ALIASES`` below, logging a loud warning (and refusing to
start unless ``pose.allow_synthetic_fallback`` is set) for any canonical name
it cannot match. Run ``scripts/inspect_metrabs_skeleton.py`` once on the
target GPU machine to print the model's actual names/edges and correct this
file (and the alias table) if they differ.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical joint names used throughout this codebase (dict keys on
# PoseFrame.keypoints). Order is not semantically meaningful -- lookups are by
# name -- but is kept stable for iteration/logging.
POSE_LANDMARKS: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",     # coco_19 extension over plain 17-point COCO: shoulder midpoint
    "pelvis",   # coco_19 extension over plain 17-point COCO: hip midpoint
]

NUM_LANDMARKS: int = len(POSE_LANDMARKS)

# Best-effort mapping from OUR canonical name to the raw name(s) MeTRAbs might
# use for the same joint, tried in order, case-insensitively. Matching also
# falls back to normalizing the raw name (lowercasing, "l_"/"r_" -> "left_"/
# "right_", stripping underscores) before comparing, so small spelling
# differences (e.g. "lshoulder" vs "left_shoulder") still resolve.
CANONICAL_TO_RAW_ALIASES: Dict[str, Tuple[str, ...]] = {
    "nose": ("nose",),
    "left_eye": ("left_eye", "leye", "l_eye"),
    "right_eye": ("right_eye", "reye", "r_eye"),
    "left_ear": ("left_ear", "lear", "l_ear"),
    "right_ear": ("right_ear", "rear", "r_ear"),
    "left_shoulder": ("left_shoulder", "lshoulder", "l_shoulder", "lsho"),
    "right_shoulder": ("right_shoulder", "rshoulder", "r_shoulder", "rsho"),
    "left_elbow": ("left_elbow", "lelbow", "l_elbow"),
    "right_elbow": ("right_elbow", "relbow", "r_elbow"),
    "left_wrist": ("left_wrist", "lwrist", "l_wrist"),
    "right_wrist": ("right_wrist", "rwrist", "r_wrist"),
    "left_hip": ("left_hip", "lhip", "l_hip"),
    "right_hip": ("right_hip", "rhip", "r_hip"),
    "left_knee": ("left_knee", "lknee", "l_knee"),
    "right_knee": ("right_knee", "rknee", "r_knee"),
    "left_ankle": ("left_ankle", "lankle", "l_ankle"),
    "right_ankle": ("right_ankle", "rankle", "r_ankle"),
    "neck": ("neck",),
    "pelvis": ("pelvis", "root", "hip"),
}

# Bone connections for the skeleton overlay, as pairs of canonical names. Used
# as a fallback when the live model's own edge list (via
# ``metrabs_model.skeleton_info``) is unavailable (e.g. offline unit tests).
POSE_CONNECTIONS: Tuple[Tuple[str, str], ...] = (
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("neck", "nose"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("neck", "pelvis"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def _normalize(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    if n.startswith("l_") or n.startswith("l"):
        if n.startswith("l_"):
            n = "left_" + n[2:]
    if n.startswith("r_") or n.startswith("r"):
        if n.startswith("r_"):
            n = "right_" + n[2:]
    return n


def build_raw_to_canonical_map(raw_names: List[str]) -> Dict[str, str]:
    """Match a live model's raw joint names to our canonical names.

    Returns ``{raw_name: canonical_name}`` for every raw name that could be
    matched. Names that cannot be matched are omitted -- callers should treat
    an incomplete match against ``POSE_LANDMARKS`` as a loud, fatal error
    (see ``pose_estimator.PoseEstimator``), not something to silently ignore.
    """
    normalized_raw = {_normalize(r): r for r in raw_names}
    result: Dict[str, str] = {}
    for canonical, aliases in CANONICAL_TO_RAW_ALIASES.items():
        for alias in aliases:
            if alias in normalized_raw:
                result[normalized_raw[alias]] = canonical
                break
            norm_alias = _normalize(alias)
            if norm_alias in normalized_raw:
                result[normalized_raw[norm_alias]] = canonical
                break
    return result


def landmark_id(name: str) -> int:
    """Index of ``name`` in the canonical :data:`POSE_LANDMARKS` order."""
    return POSE_LANDMARKS.index(name)


def enumerate_landmarks() -> List[Tuple[int, str]]:
    return list(enumerate(POSE_LANDMARKS))
