from __future__ import annotations

from src.perception.landmarks import (
    CANONICAL_TO_RAW_ALIASES,
    NUM_LANDMARKS,
    POSE_LANDMARKS,
    build_raw_to_canonical_map,
    landmark_id,
)


def test_landmark_count_is_19() -> None:
    assert NUM_LANDMARKS == 19
    assert len(POSE_LANDMARKS) == 19


def test_landmark_id_round_trip() -> None:
    for idx, name in enumerate(POSE_LANDMARKS):
        assert landmark_id(name) == idx


def test_key_anatomy_present() -> None:
    expected = {
        "nose", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    }
    assert expected.issubset(set(POSE_LANDMARKS))


def test_every_canonical_landmark_has_an_alias() -> None:
    assert set(CANONICAL_TO_RAW_ALIASES.keys()) == set(POSE_LANDMARKS)


def test_build_raw_to_canonical_map_matches_exact_names() -> None:
    # The common case: a live model reports exactly our canonical names.
    mapping = build_raw_to_canonical_map(list(POSE_LANDMARKS))
    assert set(mapping.values()) == set(POSE_LANDMARKS)


def test_build_raw_to_canonical_map_matches_common_abbreviations() -> None:
    raw = ["lsho", "rsho", "lhip", "rhip", "nose"]
    mapping = build_raw_to_canonical_map(raw)
    assert mapping["lsho"] == "left_shoulder"
    assert mapping["rsho"] == "right_shoulder"
    assert mapping["lhip"] == "left_hip"
    assert mapping["rhip"] == "right_hip"
    assert mapping["nose"] == "nose"


def test_build_raw_to_canonical_map_ignores_unknown_names() -> None:
    mapping = build_raw_to_canonical_map(["left_shoulder", "some_unknown_joint"])
    assert mapping == {"left_shoulder": "left_shoulder"}


# The joint names a live metrabs_eff2s_y4 model reports for `coco_19`, in the
# model's own order (verified against the loaded SavedModel, and printable with
# scripts/inspect_metrabs_skeleton.py). Pinned here because the alias table was
# originally written by guesswork: nine of these matched nothing, so the
# pipeline refused to start on the very first frame it ever saw a GPU.
METRABS_COCO_19_RAW = [
    "neck", "nose", "pelv",
    "lsho", "lelb", "lwri", "lhip", "lkne", "lank",
    "rsho", "relb", "rwri", "rhip", "rkne", "rank",
    "leye", "lear", "reye", "rear",
]


def test_every_real_metrabs_joint_maps_to_a_canonical_name() -> None:
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    unmatched = [r for r in METRABS_COCO_19_RAW if r not in mapping]
    assert not unmatched, f"raw MeTRAbs names with no canonical match: {unmatched}"


def test_the_real_metrabs_skeleton_covers_every_canonical_landmark() -> None:
    """The check pose_estimator.py makes at startup, run offline."""
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    missing = [n for n in POSE_LANDMARKS if n not in set(mapping.values())]
    assert not missing, f"canonical landmarks the model cannot supply: {missing}"


def test_no_two_raw_names_collapse_onto_the_same_canonical_joint() -> None:
    """A sloppy alias (e.g. a bare 'l' prefix rule) can silently make 'lear'
    and 'lelb' fight over one slot; the loser is then read as a missing joint."""
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    assert len(set(mapping.values())) == len(mapping)


def test_left_and_right_are_not_swapped() -> None:
    mapping = build_raw_to_canonical_map(METRABS_COCO_19_RAW)
    for raw, canonical in mapping.items():
        if raw.startswith("l") and canonical != "neck":
            assert canonical.startswith("left_"), f"{raw} -> {canonical}"
        elif raw.startswith("r"):
            assert canonical.startswith("right_"), f"{raw} -> {canonical}"
