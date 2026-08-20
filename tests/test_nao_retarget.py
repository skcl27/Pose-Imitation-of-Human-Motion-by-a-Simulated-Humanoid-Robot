"""Tests for full-body retargeting (main/libraries/nao_retarget.py).

The leg solve is checked by *round trip*: a synthetic camera projection is built
from known NAO joint angles, and the retargeter has to recover those angles from
the projected landmarks. That is a much stronger check than asserting on
hand-picked numbers -- it verifies the actual geometry (NAO's HipRoll -> HipPitch
-> KneePitch chain, the ankle levelling, the mirrored roll signs) rather than
just that the code runs.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from nao_retarget import (  # noqa: E402
    LowerBodyRetargeter,
    PeakHold,
    crouch_posture,
    retarget_full_body,
    retarget_upper_body,
)

# Synthetic subject geometry, in normalized image units.
TORSO = 0.25
THIGH = 0.18
SHANK = 0.18
HIP_Y = 0.55
HALF_HIP = 0.04
HALF_SHOULDER = 0.06
# Image axis mapping used by the projection below (see _project_leg):
#   NAO +y (robot's left) -> image -x        NAO +z (up) -> image -y
#   NAO +x (forward, toward camera) -> image -z
_IMG_OUT = {"L": -1.0, "R": +1.0}


def _segment(origin, out_dir, length, roll_mag, pitch, z_scale=1.0):
    """Project one limb segment onto the synthetic camera.

    The segment direction in the torso frame is
    ``R_x(roll) . R_y(pitch) . (0, 0, -1)``
    ``= (-sin(pitch), sin(roll)cos(pitch), -cos(roll)cos(pitch))``.
    """
    x, y, z = origin
    lateral = math.sin(roll_mag) * math.cos(pitch)
    vertical = math.cos(roll_mag) * math.cos(pitch)
    forward = -math.sin(pitch)
    return (
        x + out_dir * length * lateral,
        y + length * vertical,
        z - z_scale * length * forward,
    )


def _leg_landmarks(side, roll_mag, hip_pitch, knee_pitch):
    """Landmarks for one leg posed at the given NAO angles."""
    out = _IMG_OUT[side]
    hip = (0.5 + out * HALF_HIP, HIP_Y, 0.0)
    knee = _segment(hip, out, THIGH, roll_mag, hip_pitch)
    ankle = _segment(knee, out, SHANK, roll_mag, hip_pitch + knee_pitch)
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
        "left_shoulder": [0.5 - HALF_SHOULDER, sh_y, 0.0, 1.0],
        "right_shoulder": [0.5 + HALF_SHOULDER, sh_y, 0.0, 1.0],
        # Arms hanging down, elbows and wrists included so the upper-body
        # retarget has something to solve.
        "left_elbow": [0.5 - HALF_SHOULDER - 0.01, sh_y + 0.11, 0.0, 1.0],
        "right_elbow": [0.5 + HALF_SHOULDER + 0.01, sh_y + 0.11, 0.0, 1.0],
        "left_wrist": [0.5 - HALF_SHOULDER - 0.02, sh_y + 0.22, 0.0, 1.0],
        "right_wrist": [0.5 + HALF_SHOULDER + 0.02, sh_y + 0.22, 0.0, 1.0],
        "nose": [0.5, sh_y - 0.12, 0.0, 1.0],
    }
    kps.update(_leg_landmarks("L", *left))
    kps.update(_leg_landmarks("R", *right))
    return kps


def warm(retargeter, frames=80):
    """Let the self-calibration converge on the standing reference lengths."""
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
    ph.update(0.1)                        # a foreshortened frame
    assert ph.value > 0.9 * before        # barely moves the reference
    assert ph.update(float("nan")) == ph.value


def test_calibration_recovers_segment_lengths() -> None:
    r = LowerBodyRetargeter()
    warm(r)
    assert abs(r.geom.torso - TORSO) < 1e-3
    assert abs(r.geom.thigh - THIGH) < 5e-3
    assert abs(r.geom.shank - SHANK) < 5e-3
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
        assert abs(leg.hip_roll - roll) < 0.03, (roll, hip, knee, leg.hip_roll)
        assert abs(leg.hip_pitch - hip) < 0.05, (roll, hip, knee, leg.hip_pitch)
        assert abs(leg.knee_pitch - knee) < 0.06, (roll, hip, knee, leg.knee_pitch)


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
    assert 0.0 < shallow.crouch_u <= deep.crouch_u
    from nao_retarget import MAX_CROUCH
    assert deep.crouch_u <= MAX_CROUCH


def test_lift_ignores_the_subject_moving_away_from_the_camera() -> None:
    """Scale invariance: the whole figure shrinking must not read as a lift."""
    r = LowerBodyRetargeter()
    warm(r)
    far = {}
    for name, v in figure().items():
        far[name] = [0.5 + (v[0] - 0.5) * 0.6, 0.5 + (v[1] - 0.5) * 0.6, v[2], v[3]]
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
