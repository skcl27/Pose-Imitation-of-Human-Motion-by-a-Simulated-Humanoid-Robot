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
