"""Tests for the synthetic pose generator (src/perception/synthetic_pose.py).

The generator's whole value is that it is a KNOWN human: if it silently produces
an impossible body (limbs changing length, feet through the floor, landmarks
missing) then every downstream measurement taken against it is meaningless, and
worse, meaningless in a way that looks like a robot bug.
"""
from __future__ import annotations

import math

import pytest

from src.perception.gait_cues import GaitCueExtractor
from src.perception.landmarks import POSE_LANDMARKS
from src.perception.synthetic_pose import (
    FOREARM,
    GROUND_Y,
    SHANK,
    THIGH,
    UPPER_ARM,
    BodyState,
    build_pose,
)

# Every scripted motion the replay driver can emit, sampled across a full cycle.
MOTIONS = [
    BodyState(),
    BodyState(right_arm_side=math.radians(115), right_elbow=math.radians(60)),
    BodyState(left_arm_fwd=math.radians(90), right_arm_fwd=math.radians(90)),
    BodyState(crouch=0.5),
    BodyState(crouch=1.0),
    BodyState(left_hip_flex=math.radians(65), left_knee_flex=math.radians(85)),
    BodyState(right_hip_flex=math.radians(65), right_knee_flex=math.radians(85)),
    BodyState(body_yaw=math.radians(75)),
    BodyState(body_yaw=math.radians(-75)),
    BodyState(head_yaw=math.radians(35), head_pitch=math.radians(25)),
]


def _dist(a, b) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


@pytest.mark.parametrize("state", MOTIONS)
def test_every_landmark_is_present_and_finite(state) -> None:
    kps = build_pose(state, 0.0, 0).keypoints
    assert set(kps) == set(POSE_LANDMARKS)
    for name, kp in kps.items():
        assert all(math.isfinite(v) for v in (kp.x, kp.y, kp.z)), name
        assert kp.visibility == 1.0


@pytest.mark.parametrize("state", MOTIONS)
def test_limb_lengths_are_rigid(state) -> None:
    """A retargeter calibrates its own limb-length scale from the stream. If the
    synthetic subject's bones changed length between frames it would mis-scale
    everything downstream and the fault would look like a robot bug."""
    kps = build_pose(state, 0.0, 0).keypoints
    for side in ("left", "right"):
        assert _dist(kps[f"{side}_shoulder"], kps[f"{side}_elbow"]) == pytest.approx(
            UPPER_ARM, abs=1e-6)
        assert _dist(kps[f"{side}_elbow"], kps[f"{side}_wrist"]) == pytest.approx(
            FOREARM, abs=1e-6)
        assert _dist(kps[f"{side}_hip"], kps[f"{side}_knee"]) == pytest.approx(
            THIGH, abs=1e-6)
        assert _dist(kps[f"{side}_knee"], kps[f"{side}_ankle"]) == pytest.approx(
            SHANK, abs=1e-6)


def test_limb_lengths_are_rigid_across_a_whole_squat() -> None:
    """The squat is the one motion that places the knee by circle intersection
    rather than forward kinematics, so it is the one that could drift."""
    for i in range(41):
        kps = build_pose(BodyState(crouch=i / 40.0), 0.0, i).keypoints
        for side in ("left", "right"):
            assert _dist(kps[f"{side}_hip"], kps[f"{side}_knee"]) == pytest.approx(
                THIGH, abs=1e-6)
            assert _dist(kps[f"{side}_knee"], kps[f"{side}_ankle"]) == pytest.approx(
                SHANK, abs=1e-6)


def test_planted_feet_stay_on_the_ground_through_a_squat() -> None:
    """Both feet are planted in a squat; a foot that sinks through the floor
    would read to the controller as a step."""
    for i in range(21):
        kps = build_pose(BodyState(crouch=i / 20.0), 0.0, i).keypoints
        assert kps["left_ankle"].y == pytest.approx(GROUND_Y, abs=1e-6)
        assert kps["right_ankle"].y == pytest.approx(GROUND_Y, abs=1e-6)


def test_a_squat_actually_lowers_the_hips() -> None:
    tall = build_pose(BodyState(crouch=0.0), 0.0, 0).keypoints
    deep = build_pose(BodyState(crouch=1.0), 0.0, 1).keypoints
    # y is DOWN, so sinking means a LARGER y.
    assert deep["pelvis"].y > tall["pelvis"].y + 100.0
    # ...and the knees must travel forward (toward the camera, -z) to allow it.
    assert deep["left_knee"].z < tall["left_knee"].z - 50.0


def test_a_leg_lift_raises_that_ankle_only() -> None:
    kps = build_pose(
        BodyState(left_hip_flex=math.radians(65), left_knee_flex=math.radians(85)), 0.0, 0
    ).keypoints
    assert kps["left_ankle"].y < GROUND_Y - 50.0    # lifted (y down)
    assert kps["right_ankle"].y == pytest.approx(GROUND_Y, abs=1e-6)   # still planted


def test_body_yaw_preserves_every_segment_length() -> None:
    """Turning must rotate the subject, not stretch them."""
    square = build_pose(BodyState(), 0.0, 0).keypoints
    turned = build_pose(BodyState(body_yaw=math.radians(75)), 0.0, 1).keypoints
    for a, b in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
                 ("neck", "pelvis")):
        assert _dist(turned[a], turned[b]) == pytest.approx(_dist(square[a], square[b]), abs=1e-6)


def test_body_yaw_changes_the_depth_spread_of_the_shoulders() -> None:
    """This is the signal the torso-yaw estimator actually reads."""
    square = build_pose(BodyState(), 0.0, 0).keypoints
    turned = build_pose(BodyState(body_yaw=math.radians(60)), 0.0, 1).keypoints
    assert abs(square["left_shoulder"].z - square["right_shoulder"].z) < 1e-6
    assert abs(turned["left_shoulder"].z - turned["right_shoulder"].z) > 100.0


def test_generation_is_deterministic() -> None:
    """The entire point: the same input must give byte-identical output, or
    'deterministic replay' (PRD US-2) is a lie."""
    a = build_pose(BodyState(crouch=0.3, body_yaw=0.2), 1.5, 9).keypoints
    b = build_pose(BodyState(crouch=0.3, body_yaw=0.2), 1.5, 9).keypoints
    for name in a:
        assert (a[name].x, a[name].y, a[name].z) == (b[name].x, b[name].y, b[name].z)


def test_the_arms_hang_by_default() -> None:
    kps = build_pose(BodyState(), 0.0, 0).keypoints
    for side in ("left", "right"):
        # Elbow directly below the shoulder, a full upper-arm length down.
        assert kps[f"{side}_elbow"].x == pytest.approx(kps[f"{side}_shoulder"].x, abs=1e-6)
        assert kps[f"{side}_elbow"].y == pytest.approx(
            kps[f"{side}_shoulder"].y + UPPER_ARM, abs=1e-6)


# ------------------------------------------------------- left/right convention
def _settled_yaw(state: BodyState, frames: int = 40) -> float:
    """Body yaw after the extractor's EMA has settled, in degrees."""
    gait = GaitCueExtractor()
    cue = None
    for i in range(frames):
        cue = gait.update(build_pose(state, i / 30.0, i))
    return math.degrees(cue.body_yaw_rad)


def test_a_front_facing_subject_has_their_left_on_the_positive_x_side() -> None:
    """When you face someone, your left is on their right. Landmark names are
    ANATOMICAL, so a subject facing the camera has left_shoulder at positive x."""
    kps = build_pose(BodyState(), 0.0, 0).keypoints
    assert kps["left_shoulder"].x > kps["right_shoulder"].x
    assert kps["left_hip"].x > kps["right_hip"].x
    assert kps["left_ear"].x > kps["right_ear"].x


def test_a_subject_facing_the_camera_reads_as_zero_yaw() -> None:
    """The check src/perception/gait_cues.py's docstring asks a human to perform
    ("stand facing the camera, confirm yaw reads ~0") and which had never been
    automated. Get the side convention backwards and someone standing perfectly
    still reads as turned 180 degrees, so the controller demands turn clips
    forever and never walks -- and nothing in the suite would have noticed."""
    assert abs(_settled_yaw(BodyState())) < 5.0


def test_turning_is_reported_with_the_right_sign_and_size() -> None:
    left = _settled_yaw(BodyState(body_yaw=math.radians(40.0)))
    right = _settled_yaw(BodyState(body_yaw=math.radians(-40.0)))
    assert left * right < 0, "turning each way must give opposite signs"
    assert 15.0 < abs(left) < 75.0, f"turn magnitude implausible: {left:.1f} deg"
    assert 15.0 < abs(right) < 75.0, f"turn magnitude implausible: {right:.1f} deg"


def test_the_side_convention_survives_a_mirrored_capture() -> None:
    """input.flip_horizontal mirrors the image before inference. That moves a
    shoulder to the other side AND makes the estimator label it as the other
    shoulder, so the two swaps cancel and the sign is unchanged. This pins that
    reasoning, because the alternative -- that the flip inverts the convention --
    is a very natural thing to conclude and would send someone inverting a
    correct sign in shipping robot code."""
    kps = build_pose(BodyState(), 0.0, 0).keypoints
    mirrored = {}
    for name, kp in kps.items():
        if name.startswith("left_"):
            other = "right_" + name[len("left_"):]
        elif name.startswith("right_"):
            other = "left_" + name[len("right_"):]
        else:
            other = name
        mirrored[other] = (-kp.x, kp.y, kp.z)
    assert (mirrored["left_shoulder"][0] - mirrored["right_shoulder"][0]) == pytest.approx(
        kps["left_shoulder"].x - kps["right_shoulder"].x, abs=1e-9)
