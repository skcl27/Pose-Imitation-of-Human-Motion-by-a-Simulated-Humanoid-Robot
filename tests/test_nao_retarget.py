"""Tests for full-body retargeting (main/libraries/nao_retarget.py).

The leg solve is checked by *round trip*: a synthetic camera-frame 3D pose
(mm, camera coordinates -- see nao_retarget.py's module docstring) is built
from known NAO joint angles via the same swing-twist forward-kinematics
relationship the retargeter inverts, and the retargeter has to recover those
angles from the projected landmarks. That is a much stronger check than
asserting on hand-picked numbers -- it verifies the actual geometry (NAO's
HipRoll -> HipPitch -> KneePitch chain, the ankle levelling, the mirrored
roll signs) rather than just that the code runs.

Unlike the MediaPipe-era version of this test file, landmarks here are
absolute 3D (mm) rather than a 2D image-plane projection, so there is no
approximation/foreshortening error to tolerate in the round trip -- the
recovered angles should match the input angles almost exactly.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from nao_retarget import (  # noqa: E402
    HEAD_PITCH_BASELINE,
    HeadGeometry,
    LowerBodyRetargeter,
    PeakHold,
    _side_sign,
    crouch_posture,
    retarget_full_body,
    retarget_upper_body,
)

# Synthetic subject geometry, in millimeters.
TORSO = 250.0
THIGH = 180.0
SHANK = 180.0
HIP_Y = 550.0
HALF_HIP = 40.0
HALF_SHOULDER = 60.0
CENTER_X = 500.0

# The synthetic shoulders/hips below are placed so that ``_torso_frame``
# computes exactly this identity basis: right=(1,0,0), up=(0,-1,0),
# forward=(0,0,-1). Note this is a construction choice for the test (it does
# not assert anything about which way a real subject faces the camera -- see
# gait_cues.py for that empirical, separately-flagged convention) -- it only
# needs to be internally self-consistent with the forward-kinematics helpers
# below, which is what the round trip actually checks.
def _leg_dir(side, roll, pitch):
    s = _side_sign(side)
    return (
        s * math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
        -math.sin(pitch),
    )


def _leg_landmarks(side, roll_mag, hip_pitch, knee_pitch):
    """Landmarks for one leg posed at the given NAO angles."""
    hip = (CENTER_X + _side_sign(side) * HALF_HIP, HIP_Y, 0.0)
    d1 = _leg_dir(side, roll_mag, hip_pitch)
    knee = tuple(hip[i] + THIGH * d1[i] for i in range(3))
    d2 = _leg_dir(side, roll_mag, hip_pitch + knee_pitch)
    ankle = tuple(knee[i] + SHANK * d2[i] for i in range(3))
    pre = "left_" if side == "L" else "right_"
    return {
        pre + "hip": [hip[0], hip[1], hip[2], 1.0],
        pre + "knee": [knee[0], knee[1], knee[2], 1.0],
        pre + "ankle": [ankle[0], ankle[1], ankle[2], 1.0],
    }


def figure(left=(0.0, 0.0, 0.0), right=(0.0, 0.0, 0.0)):
    """A whole synthetic subject; each leg is ``(roll_mag, hip_pitch, knee)``."""
    sh_y = HIP_Y - TORSO
    kps = {
        "left_shoulder": [CENTER_X - HALF_SHOULDER, sh_y, 0.0, 1.0],
        "right_shoulder": [CENTER_X + HALF_SHOULDER, sh_y, 0.0, 1.0],
        # Arms hanging down, elbows and wrists included so the upper-body
        # retarget has something to solve (not round-trip tested here).
        "left_elbow": [CENTER_X - HALF_SHOULDER - 10.0, sh_y + 110.0, 0.0, 1.0],
        "right_elbow": [CENTER_X + HALF_SHOULDER + 10.0, sh_y + 110.0, 0.0, 1.0],
        "left_wrist": [CENTER_X - HALF_SHOULDER - 20.0, sh_y + 220.0, 0.0, 1.0],
        "right_wrist": [CENTER_X + HALF_SHOULDER + 20.0, sh_y + 220.0, 0.0, 1.0],
        "nose": [CENTER_X, sh_y - 120.0, 0.0, 1.0],
    }
    kps.update(_leg_landmarks("L", *left))
    kps.update(_leg_landmarks("R", *right))
    return kps


def warm(retargeter, frames=80):
    """Let the geometry EMA settle on the standing reference lengths."""
    obs = None
    for _ in range(frames):
        obs = retargeter.observe(figure())
    return obs


# ---------------------------------------------------------------- calibration
def test_peak_hold_rises_fast_and_decays_slowly() -> None:
    ph = PeakHold(rise=0.5, decay=0.01)
    ph.update(1.0)
    assert ph.update(2.0) > 1.4          # rises quickly toward a new peak
    before = ph.value
    ph.update(0.1)                        # a low sample
    assert ph.value > 0.9 * before        # barely moves the reference
    assert ph.update(float("nan")) == ph.value


def test_calibration_recovers_segment_lengths() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    # Real mm measurements, EMA-smoothed but not foreshortened -- tight
    # tolerance compared to the old MediaPipe-era scale-recovery test.
    assert abs(r.geom.torso - TORSO) < 1e-2
    assert abs(r.geom.thigh - THIGH) < 1e-2
    assert abs(r.geom.shank - SHANK) < 1e-2
    assert r.geom.calibrated


def test_uncalibrated_observation_is_invalid() -> None:
    # One frame is not enough to trust the proportions... but it must never throw.
    obs = LowerBodyRetargeter().observe({})
    assert obs.valid is False
    assert obs.leg("L") is None


# --------------------------------------------------------------- the leg solve
def test_standing_leg_solves_to_zero() -> None:
    obs = warm(LowerBodyRetargeter())
    for side in ("L", "R"):
        leg = obs.leg(side)
        assert abs(leg.hip_pitch) < 0.02
        assert abs(leg.hip_roll) < 0.02
        assert abs(leg.knee_pitch) < 0.02
        assert leg.lift < 0.02
    assert obs.crouch_u == 0.0
    assert obs.stance_side == ""


def test_round_trip_recovers_hip_and_knee_angles() -> None:
    cases = [
        (0.0, -0.60, 1.20),     # knee lifted forward, shank folded under
        (0.0, -1.20, 1.40),     # high march step
        (0.35, -0.30, 0.50),    # abducted and flexed
        (0.60, 0.0, 0.0),       # pure abduction
        (0.0, -0.40, 0.80),     # shallow crouch on one leg
    ]
    for roll, hip, knee in cases:
        r = LowerBodyRetargeter()
        warm(r)
        obs = r.observe(figure(left=(roll, hip, knee)))
        leg = obs.left
        assert abs(leg.hip_roll - roll) < 0.01, (roll, hip, knee, leg.hip_roll)
        assert abs(leg.hip_pitch - hip) < 0.01, (roll, hip, knee, leg.hip_pitch)
        assert abs(leg.knee_pitch - knee) < 0.01, (roll, hip, knee, leg.knee_pitch)


def test_roll_signs_are_mirrored_between_legs() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    obs = r.observe(figure(left=(0.5, 0.0, 0.0), right=(0.5, 0.0, 0.0)))
    # Both legs abducted outward: NAO wants LHipRoll positive, RHipRoll negative.
    assert obs.left.hip_roll > 0.4
    assert obs.right.hip_roll < -0.4
    # The ankle cancels the hip so each sole stays level.
    assert abs(obs.left.ankle_roll + obs.left.hip_roll) < 1e-9
    assert abs(obs.right.ankle_roll + obs.right.hip_roll) < 1e-9


def test_ankle_keeps_the_sole_flat() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    obs = r.observe(figure(left=(0.0, -0.5, 1.0), right=(0.0, -0.5, 1.0)))
    for leg in (obs.left, obs.right):
        # Hip + knee + ankle == 0 => torso vertical and sole flat.
        assert abs(leg.hip_pitch + leg.knee_pitch + leg.ankle_pitch) < 1e-9


def test_knee_never_hyperextends() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    for hip in (-1.2, -0.6, 0.0):
        obs = r.observe(figure(left=(0.0, hip, 0.0)))
        assert obs.left.knee_pitch >= 0.0


# ------------------------------------------------------------------- lift/stance
def test_single_leg_lift_is_detected_on_the_right_side() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    # A high march step on the left: hip flexed, knee folded -> the foot rises.
    obs = r.observe(figure(left=(0.0, -1.1, 1.5)))
    assert obs.left.lift > 0.5
    assert obs.right.lift < 0.05
    # Stance is the OTHER foot -- this is what tells the controller which way to
    # transfer weight before the step.
    assert obs.stance_side == "R"


def test_both_feet_planted_is_not_a_step() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    obs = r.observe(figure(left=(0.0, -0.4, 0.8), right=(0.0, -0.4, 0.8)))
    assert obs.stance_side == ""
    assert obs.left.lift < 0.05 and obs.right.lift < 0.05


def test_symmetric_squat_reports_a_crouch() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    shallow = r.observe(figure(left=(0.0, -0.3, 0.6), right=(0.0, -0.3, 0.6)))
    deep = r.observe(figure(left=(0.0, -0.7, 1.4), right=(0.0, -0.7, 1.4)))
    assert 0.0 < shallow.crouch_u < deep.crouch_u
    from nao_retarget import MAX_CROUCH
    assert deep.crouch_u <= MAX_CROUCH


def test_the_crouch_keeps_the_ankle_under_the_hip_at_any_depth() -> None:
    """This is why the crouch cap is a joint-range limit and not a safety one:
    NAO's thigh and shank are within 3 mm of the same length, so the hip stays
    over the ankle however deep the squat goes."""
    from balance import THIGH_LENGTH, TIBIA_LENGTH
    for u in (0.0, 0.35, 0.70, 1.0):
        p = crouch_posture(u)
        # Forward offset of the ankle from the hip, from the two segment pitches.
        offset = (THIGH_LENGTH * math.sin(u)
                  + TIBIA_LENGTH * math.sin(-(u)))
        assert abs(offset) < 0.004
        for side in ("L", "R"):
            total = (p[f"{side}HipPitch"] + p[f"{side}KneePitch"]
                     + p[f"{side}AnklePitch"])
            assert abs(total) < 1e-12          # torso vertical, sole flat


def test_lift_is_invariant_to_camera_distance() -> None:
    """Real 3D coordinates are already metric, so moving the whole subject
    farther from the camera (a uniform z shift) must not read as a lift --
    unlike MediaPipe's foreshortened 2D projection, there is no scale
    ambiguity left to introduce spurious lift, but a uniform depth offset is
    still worth checking since the leg solve uses z for the fwd/backward
    sign."""
    r = LowerBodyRetargeter()
    warm(r)
    far = {}
    for name, v in figure().items():
        far[name] = [v[0], v[1], v[2] + 1500.0, v[3]]
    obs = r.observe(far)
    assert obs.left.lift < 0.1 and obs.right.lift < 0.1


# ------------------------------------------------------------------ robustness
def test_invisible_leg_is_omitted_not_guessed() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    kps = figure()
    for name in ("left_hip", "left_knee", "left_ankle"):
        kps[name] = kps[name][:3] + [0.1]      # visibility below threshold
    obs = r.observe(kps)
    assert obs.left is None
    assert obs.right is not None
    # A one-legged read is explicitly less trusted.
    assert obs.confidence <= 0.5


def test_garbage_landmarks_do_not_raise() -> None:
    r = LowerBodyRetargeter()
    for payload in ({}, {"left_hip": []}, {"left_hip": ["x", "y"]},
                    {"left_hip": [float("nan"), 0.0, 0.0, 1.0]}):
        assert r.observe(payload).valid is False


# ------------------------------------------------------------------ upper body
def test_upper_body_returns_no_leg_joints() -> None:
    targets = retarget_upper_body(figure())
    assert targets
    assert not any("Hip" in n or "Knee" in n or "Ankle" in n for n in targets)


def test_upper_body_needs_no_hips_visible() -> None:
    """A desk-framed webcam (waist up only) should still drive arms/head --
    see nao_retarget._torso_frame's camera-vertical fallback."""
    kps = figure()
    for name in ("left_hip", "right_hip", "left_knee", "right_knee",
                 "left_ankle", "right_ankle"):
        kps.pop(name, None)
    targets = retarget_upper_body(kps)
    assert "LShoulderPitch" in targets and "RShoulderPitch" in targets


def test_full_body_includes_legs_only_when_asked() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    without = retarget_full_body(figure(), drive_legs=False)
    with_legs = retarget_full_body(figure(left=(0.0, -0.5, 1.0)),
                                  drive_legs=True, retargeter=r)
    assert "LKneePitch" not in without
    assert "LKneePitch" in with_legs


def test_swap_sides_mirrors_every_joint() -> None:
    normal = retarget_upper_body(figure())
    mirrored = retarget_upper_body(figure(), swap_sides=True)
    assert abs(normal["LShoulderPitch"] - mirrored["RShoulderPitch"]) < 1e-9


def test_crouch_posture_is_statically_balanced() -> None:
    for u in (0.0, 0.1, 0.35):
        p = crouch_posture(u)
        for side in ("L", "R"):
            total = p[f"{side}HipPitch"] + p[f"{side}KneePitch"] + p[f"{side}AnklePitch"]
            assert abs(total) < 1e-12   # torso vertical, sole flat
        assert p["LHipRoll"] == p["RHipRoll"] == 0.0


# ------------------------------------------------- head-pitch self-calibration
# ``figure()``'s nose sits 120 mm above a 120 mm shoulder span, i.e. exactly
# 1.0 shoulder-widths -- a subject whose neck is longer than the population
# average encoded in HEAD_PITCH_BASELINE.
def _with_neck(nose_height: float):
    """``figure()`` with the nose at ``nose_height`` shoulder-widths up."""
    kps = figure()
    shoulder_w = 2.0 * HALF_SHOULDER
    kps["nose"] = [CENTER_X, (HIP_Y - TORSO) - nose_height * shoulder_w, 0.0, 1.0]
    return kps


def test_an_off_average_neck_biases_the_head_without_calibration() -> None:
    """The regression this class exists for: a subject looking straight ahead
    gets a standing head tilt purely because their neck is not average."""
    pitch = retarget_upper_body(_with_neck(1.30))["HeadPitch"]
    assert abs(pitch) > 0.5   # ~ -37 deg of permanent nod


def test_calibration_removes_the_bias_for_a_neutral_head() -> None:
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        targets = retarget_upper_body(_with_neck(1.30), head_geom=geom)
    assert geom.calibrated
    assert abs(targets["HeadPitch"]) < 1e-6
    assert abs(geom.baseline - 1.30) < 1e-6


def test_calibration_does_not_flatten_real_head_motion() -> None:
    """Calibrating away the OFFSET must not calibrate away the SIGNAL."""
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        retarget_upper_body(_with_neck(1.30), head_geom=geom)
    # Nose drops toward the shoulders -> looking down -> positive pitch.
    assert retarget_upper_body(_with_neck(1.10), head_geom=geom)["HeadPitch"] > 0.2
    # ... and the opposite way for looking up.
    geom_up = HeadGeometry()
    for _ in range(geom_up.warmup):
        retarget_upper_body(_with_neck(1.30), head_geom=geom_up)
    assert retarget_upper_body(_with_neck(1.50), head_geom=geom_up)["HeadPitch"] < -0.2


def test_the_first_frame_replaces_the_population_default() -> None:
    geom = HeadGeometry()
    assert geom.baseline == HEAD_PITCH_BASELINE
    assert geom.update(1.40) == 1.40      # not averaged with the default


def test_non_finite_samples_cannot_poison_the_neutral() -> None:
    geom = HeadGeometry()
    geom.update(1.20)
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert geom.update(bad) == 1.20


def test_a_brief_glance_does_not_move_a_settled_neutral() -> None:
    """After warm-up the neutral drifts slowly, so looking away for a second
    must not drag the robot's idea of 'straight ahead' with it."""
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        geom.update(1.00)
    for _ in range(50):                    # ~1.7 s of looking down
        geom.update(0.60)
    assert abs(geom.baseline - 1.00) < 0.05


def test_a_new_subject_eventually_recalibrates() -> None:
    """The slow decay is what stops the second person in front of the camera
    from inheriting the first one's neck."""
    geom = HeadGeometry()
    for _ in range(geom.warmup):
        geom.update(1.00)
    for _ in range(2000):
        geom.update(1.40)
    assert abs(geom.baseline - 1.40) < 0.05


# ------------------------------------------------- arm fore/aft axis (chirality)
# These use a subject built to face the camera (left shoulder at POSITIVE x --
# your left is on my right), because that is what a real estimator reports and
# it is the orientation under which the fore/aft sign actually matters.
#
# The suite could not catch a front-to-back inversion before: figure() hangs the
# arms straight down with z = 0, where the fore/aft component of the arm bone is
# zero and its sign therefore cannot change the answer.
def _facing_subject(left_arm_forward_rad: float = 0.0):
    """A camera-facing figure whose LEFT arm is swung forward by the given angle.

    Camera frame: x right, y DOWN, z away from the camera -- so "forward" for the
    subject (toward the camera) is NEGATIVE z.
    """
    sh_y, half_sh, upper, fore = -500.0, 190.0, 300.0, 260.0
    dy, dz = math.cos(left_arm_forward_rad), -math.sin(left_arm_forward_rad)
    ls = (half_sh, sh_y, 0.0)          # left on POSITIVE x: subject faces us
    rs = (-half_sh, sh_y, 0.0)
    return {
        "left_shoulder": [ls[0], ls[1], ls[2], 1.0],
        "right_shoulder": [rs[0], rs[1], rs[2], 1.0],
        "left_hip": [100.0, 0.0, 0.0, 1.0],
        "right_hip": [-100.0, 0.0, 0.0, 1.0],
        "left_elbow": [ls[0], ls[1] + upper * dy, ls[2] + upper * dz, 1.0],
        "left_wrist": [ls[0], ls[1] + (upper + fore) * dy, ls[2] + (upper + fore) * dz, 1.0],
        "right_elbow": [rs[0], rs[1] + upper, rs[2], 1.0],
        "right_wrist": [rs[0], rs[1] + upper + fore, rs[2], 1.0],
        "nose": [0.0, sh_y - 240.0, 0.0, 1.0],
    }


def test_a_hanging_arm_is_shoulder_pitch_ninety() -> None:
    """NAO's ShoulderPitch is +90 deg for an arm at the side."""
    t = retarget_upper_body(_facing_subject(0.0))
    assert math.degrees(t["LShoulderPitch"]) == pytest.approx(90.0, abs=1.0)


def test_an_arm_reaching_at_the_camera_is_shoulder_pitch_zero() -> None:
    """The regression this pair exists for. TorsoFrame.forward is right x up,
    which points out of the subject's BACK; NAO's ShoulderPitch is 0 for an arm
    held FORWARD, so the arm solve must read the negated axis. Without that, an
    arm pointing at the camera solved to atan2(0, -1) = 180 deg and the joint
    limit clamped it to 119.5 -- the arm hit its mechanical stop instead of
    reaching forward, on every frame of every forward reach."""
    t = retarget_upper_body(_facing_subject(math.radians(90.0)))
    assert math.degrees(t["LShoulderPitch"]) == pytest.approx(0.0, abs=1.0)


def test_the_forward_reach_is_monotonic_and_never_saturates() -> None:
    """A smooth human motion must produce a smooth robot one. The broken version
    was not merely offset: it saturated at +119.5 for most of the range and then
    flipped sign to -119.5 at the end."""
    limit = math.radians(119.5)
    pitches = [
        retarget_upper_body(_facing_subject(math.radians(d)))["LShoulderPitch"]
        for d in range(0, 91, 10)
    ]
    assert all(abs(p) < limit - 1e-3 for p in pitches), "a joint hit its limit"
    for earlier, later in zip(pitches, pitches[1:], strict=False):
        assert later < earlier + 1e-9, "shoulder pitch must fall as the arm rises"
    assert math.degrees(pitches[0] - pitches[-1]) == pytest.approx(90.0, abs=2.0)


def test_the_swing_is_defined_when_the_bone_lies_on_the_second_axis() -> None:
    """An arm straight out to the side puts the bone along the roll axis, so the
    swing angle is undefined and only atan2's SIGNED ZEROS would decide it --
    which is a coin flip between 0 and 180 deg, and 180 saturates the joint."""
    from nao_retarget import _swing_twist

    first, second = _swing_twist(0.0, 0.0, 1.0)
    assert first == 0.0
    assert second == pytest.approx(math.pi / 2.0)
    first_neg, second_neg = _swing_twist(-0.0, -0.0, -1.0)
    assert first_neg == 0.0
    assert second_neg == pytest.approx(-math.pi / 2.0)


def test_the_leg_solve_still_reads_the_unnegated_axis() -> None:
    """The arm fix must NOT be applied to the legs: NAO's HipPitch is NEGATIVE
    for a thigh swung forward, so the leg solve wants the back-pointing axis that
    TorsoFrame.forward already provides."""
    r = LowerBodyRetargeter()
    for _ in range(90):
        r.observe(figure())
    obs = r.observe(figure(left=(0.0, -0.6, 1.0)))
    leg = obs.leg("L")
    assert leg is not None
    assert leg.as_targets("L")["LHipPitch"] < 0.0
