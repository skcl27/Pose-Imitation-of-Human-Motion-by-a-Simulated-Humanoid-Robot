"""Tests for the balance-aware lower-body controller (main/libraries/lower_body.py).

These cover the safety-critical invariants that can be checked off-simulation:

* standing still commands exactly the proven symmetric crouch,
* a foot is never unloaded before the CoM model says the weight is transferred,
* the lean direction is taken from the model rather than a hard-coded sign,
* everything ramps (no discontinuities) and always has a path back to the crouch,
* the tilt / confidence / both-feet-up guards all stand the robot down,
* every commanded angle stays inside NAO's mechanical limits.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import NaoCoMModel  # noqa: E402
from lower_body import (  # noqa: E402
    ANKLE_ROLL_LIMIT,
    LEG_JOINTS,
    MODE_DOUBLE,
    MODE_SINGLE,
    LowerBodyController,
    LowerBodyParams,
)
from nao_retarget import (  # noqa: E402
    LegTarget,
    LowerBodyObservation,
    crouch_posture,
)
from pose_control_utils import get_default_motor_configs  # noqa: E402

CONFIGS = get_default_motor_configs()
DT = 0.02


def rest_state():
    return {name: cfg.rest_angle for name, cfg in CONFIGS.items()}


def standing(crouch_u=0.0, conf=1.0):
    """An observation of a subject standing still with straight legs."""
    leg = LegTarget(0.0, 0.0, 0.0, 0.0, 0.0, lift=0.0, confidence=conf)
    return LowerBodyObservation(
        left=leg, right=LegTarget(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, conf),
        crouch_u=crouch_u, stance_side="", confidence=conf, valid=True,
    )


def abducted(angle=0.35, conf=1.0):
    """Both legs spread outward by ``angle`` -- a wider stance.

    NAO's roll signs are mirrored, so "outward" is +angle on the left and
    -angle on the right; the ankle cancels the hip to keep each sole flat.
    """
    return LowerBodyObservation(
        left=LegTarget(0.0, +angle, 0.0, 0.0, -angle, lift=0.0, confidence=conf),
        right=LegTarget(0.0, -angle, 0.0, 0.0, +angle, lift=0.0, confidence=conf),
        crouch_u=0.0, stance_side="", confidence=conf, valid=True,
        lift_source="feet",
    )


def leaning(angle=0.35, conf=1.0):
    """Both legs rolled the SAME way -- a lean, which moves the centre of mass."""
    leg = LegTarget(0.0, +angle, 0.0, 0.0, -angle, lift=0.0, confidence=conf)
    return LowerBodyObservation(
        left=leg, right=LegTarget(0.0, +angle, 0.0, 0.0, -angle, 0.0, conf),
        crouch_u=0.0, stance_side="", confidence=conf, valid=True,
        lift_source="feet",
    )


def split_stance(angle=0.5, conf=1.0):
    """One leg forward, one back -- antisymmetric pitch."""
    return LowerBodyObservation(
        left=LegTarget(-angle, 0.0, 0.0, +angle, 0.0, lift=0.0, confidence=conf),
        right=LegTarget(+angle, 0.0, 0.0, -angle, 0.0, lift=0.0, confidence=conf),
        crouch_u=0.0, stance_side="", confidence=conf, valid=True,
        lift_source="feet",
    )


def lifting(side="L", lift=0.9, conf=1.0):
    """An observation of a subject holding one knee up."""
    up = LegTarget(-1.0, 0.0, 1.3, -0.3, 0.0, lift=lift, confidence=conf)
    down = LegTarget(0.0, 0.0, 0.0, 0.0, 0.0, lift=0.0, confidence=conf)
    return LowerBodyObservation(
        left=up if side == "L" else down,
        right=up if side == "R" else down,
        crouch_u=0.0, stance_side=("R" if side == "L" else "L"),
        confidence=conf, valid=True,
    )


def run(ctl, obs, seconds, *, state=None, t0=0.0, **kw):
    """Drive the controller for ``seconds``, feeding its own output back as the
    measured state (a perfect-tracking approximation of the robot)."""
    state = rest_state() if state is None else state
    t = t0
    targets, meta = {}, {}
    for _ in range(max(1, int(seconds / DT))):
        targets, meta = ctl.step(t, obs, measured=state, **kw)
        state.update(targets)
        t += DT
    return targets, meta, state, t


# ------------------------------------------------------------------- standing
def test_standing_is_the_symmetric_crouch() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, standing(), 1.0)
    assert meta["mode"] == MODE_DOUBLE
    assert meta["single_support"] is False
    assert meta["balance_ok"] is True     # symmetric balance may still act
    for side in ("L", "R"):
        total = (targets[f"{side}HipPitch"] + targets[f"{side}KneePitch"]
                 + targets[f"{side}AnklePitch"])
        assert abs(total) < 1e-9          # torso vertical, soles flat
        assert abs(targets[f"{side}HipRoll"]) < 1e-9
    assert abs(targets["LHipPitch"] - targets["RHipPitch"]) < 1e-9


def test_no_observation_still_commands_a_safe_posture() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta = ctl.step(0.0, None, measured=rest_state())
    assert meta["mode"] == MODE_DOUBLE
    assert set(LEG_JOINTS) <= set(targets)
    base = crouch_posture(ctl.params.base_crouch_u)
    assert abs(targets["LKneePitch"] - base["LKneePitch"]) < 1e-9


def squatting(depth, conf=1.0):
    """Both legs in the crouch posture at ``depth`` radians."""
    leg = LegTarget(-depth, 0.0, 2.0 * depth, -depth, 0.0, lift=0.0, confidence=conf)
    return LowerBodyObservation(
        left=leg, right=LegTarget(-depth, 0.0, 2 * depth, -depth, 0.0, 0.0, conf),
        crouch_u=0.0, stance_side="", confidence=conf, valid=True,
        lift_source="feet",
    )


def test_squat_follows_the_human_one_to_one_within_range() -> None:
    p = LowerBodyParams()
    for depth in (0.20, 0.35, 0.60):
        ctl = LowerBodyController(com_model=NaoCoMModel())
        targets, meta, _, _ = run(ctl, squatting(depth), 2.0)
        assert meta["crouch_u"] == pytest.approx(depth, abs=1e-6)
        assert targets["LHipPitch"] == pytest.approx(-depth, abs=0.01)
        assert targets["LKneePitch"] == pytest.approx(2 * depth, abs=0.01)

    # Below the floor the knees stay softly bent: fully locked knees leave the
    # balance loop nothing to work with.
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, standing_meta, _, _ = run(ctl, squatting(0.0), 2.0)
    assert standing_meta["crouch_u"] == p.base_crouch_u

    # And an absurd request is capped.
    ctl2 = LowerBodyController(com_model=NaoCoMModel())
    _, deep, _, _ = run(ctl2, squatting(3.0), 2.0)
    assert deep["crouch_u"] == p.max_crouch_u


def test_a_waist_hinge_is_not_read_as_a_squat() -> None:
    """Flexing the hip with straight knees is a bend at the waist, which NAO has
    no torso joint to render -- it must not become a squat."""
    p = LowerBodyParams()
    leg = LegTarget(-0.5, 0.0, 0.0, 0.5, 0.0, lift=0.0, confidence=1.0)
    obs = LowerBodyObservation(
        left=leg, right=LegTarget(-0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0),
        confidence=1.0, valid=True, lift_source="feet",
    )
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, obs, 2.0)
    assert meta["crouch_u"] == p.base_crouch_u


def test_lifting_one_knee_does_not_make_the_robot_squat() -> None:
    """A raised leg is deeply flexed at hip and knee, so averaging both legs made
    a knee-lift also squat the robot. The straighter leg carries the weight and
    is the one that says how low the body is."""
    p = LowerBodyParams()
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, lifting("L"), 3.0)
    assert meta["crouch_u"] == p.base_crouch_u


# ------------------------------------------------------- the step sequence
def test_lift_requires_the_weight_shift_first() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    # One single step: the shift has barely started, so nothing may lift yet.
    _, meta = ctl.step(0.0, lifting("L"), measured=rest_state())
    _, meta = ctl.step(DT, lifting("L"), measured=rest_state())
    assert meta["shift"] < ctl.params.shift_ready
    assert meta["lift"] == 0.0
    assert meta["mode"] != MODE_SINGLE


def test_lift_completes_once_the_model_confirms_the_transfer() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, lifting("L"), 3.0)
    assert meta["mode"] == MODE_SINGLE
    assert meta["stance_side"] == "R"
    assert meta["single_support"] is True
    assert meta["balance_ok"] is False     # symmetric balance is invalid now
    assert meta["gate"] > 0.0
    assert meta["stance_margin"] > 0.0
    # The lifted leg is the one the human lifted, and it is clearly flexed.
    assert targets["LKneePitch"] > targets["RKneePitch"] + 0.3
    assert targets["LHipPitch"] < targets["RHipPitch"] - 0.2


def test_the_lean_direction_comes_from_the_com_model() -> None:
    """Loading a foot must raise that foot's stance margin -- whichever sign
    that turns out to be. A hard-coded lean is how balance loops tip faster."""
    model = NaoCoMModel()
    for side, other in (("L", "R"), ("R", "L")):
        ctl = LowerBodyController(com_model=model)
        _, meta, state, _ = run(ctl, lifting(other), 3.0)   # lift `other`, stand on `side`
        assert meta["stance_side"] == side
        assert model.stance_margin(state, side) > model.stance_margin(state, other)


def test_foot_returns_to_the_ground_when_the_human_lowers_it() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, state, t = run(ctl, lifting("L"), 3.0)
    assert meta["mode"] == MODE_SINGLE
    targets, meta, _, _ = run(ctl, standing(), 3.0, state=state, t0=t)
    assert meta["mode"] == MODE_DOUBLE
    assert meta["lift"] == 0.0 and meta["shift"] == 0.0
    assert abs(targets["LHipPitch"] - targets["RHipPitch"]) < 1e-9


def test_blends_are_rate_limited() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    prev = 0.0
    for i in range(40):
        _, meta = ctl.step(i * DT, lifting("L"), measured=rest_state())
        assert meta["shift"] - prev <= ctl.params.shift_rate * DT + 1e-9
        prev = meta["shift"]


# ---------------------------------------------------------------- the guards
def test_excess_tilt_stands_the_robot_down() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, state, t = run(ctl, lifting("L"), 3.0)
    assert meta["mode"] == MODE_SINGLE
    tilt = ctl.params.tilt_abort_rad + 0.1
    _, meta, _, _ = run(ctl, lifting("L"), 3.0, state=state, t0=t, torso_rp=(tilt, 0.0))
    assert meta["tilt_ok"] is False
    assert meta["mode"] == MODE_DOUBLE


def test_tilt_gate_also_freezes_the_antisymmetric_lean() -> None:
    """The "standing down" gate must not keep reapplying an asymmetric
    forward/back stride onto the legs -- that is exactly the kind of input
    that tips the robot over. Reapplying it anyway (only ``swing``/lift was
    gated, not the double-support deviation) is what let the robot get stuck
    leaning: the CoM balance correction never got to recover because the
    lean-inducing signal kept being reasserted every step."""
    p = LowerBodyParams()
    tilt = p.tilt_abort_rad + 0.1
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, split_stance(0.5), 1.0, torso_rp=(tilt, 0.0))
    assert meta["tilt_ok"] is False
    assert meta["mode"] == MODE_DOUBLE
    assert targets["LHipPitch"] == pytest.approx(targets["RHipPitch"], abs=1e-9)
    assert targets["LAnklePitch"] == pytest.approx(targets["RAnklePitch"], abs=1e-9)

    # ... but the same asymmetric stance is honoured once tilt recovers, so
    # this is a gate, not a dead code path.
    ctl2 = LowerBodyController(com_model=NaoCoMModel())
    ok_targets, ok_meta, _, _ = run(ctl2, split_stance(0.5), 1.0, torso_rp=(0.0, 0.0))
    assert ok_meta["tilt_ok"] is True
    assert abs(ok_targets["LHipPitch"] - ok_targets["RHipPitch"]) > 0.05


def test_low_confidence_stands_the_robot_down() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, lifting("L", conf=0.1), 3.0)
    assert meta["tracking"] is False
    assert meta["mode"] == MODE_DOUBLE


def test_both_feet_up_is_refused() -> None:
    """A jump, or bad tracking -- either way not a step."""
    up = LegTarget(-1.0, 0.0, 1.3, -0.3, 0.0, lift=0.9, confidence=1.0)
    obs = LowerBodyObservation(left=up, right=up, confidence=1.0, valid=True)
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, obs, 2.0)
    assert meta["swing_side"] == ""
    assert meta["mode"] == MODE_DOUBLE


def test_foot_sensors_scale_the_lift_rather_than_forbidding_it() -> None:
    """The FSRs are a confirmation, not a veto.

    They used to veto outright, which meant a sensor reading a constant 50/50 --
    uncalibrated, or a proto whose soles barely redistribute load -- forbade every
    step forever. That is exactly how a leg lift ends up doing nothing at all, and
    it is a silent failure. By the time this gate runs the CoM model has already
    agreed the weight is over the stance foot, and the tilt abort is the real
    safety net, so a disagreeing sensor reduces authority instead of removing it.
    """
    p = LowerBodyParams()
    # Load clearly on the WRONG foot: authority is cut, but not to zero.
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, unconfirmed, _, _ = run(ctl, lifting("L"), 3.0, fsr={"L": 25.0, "R": 2.0})
    assert unconfirmed["gate"] == pytest.approx(p.fsr_min_gain, abs=0.05)
    assert 0.0 < unconfirmed["lift"] < 0.6
    assert unconfirmed["fsr_share"] < 0.5

    # A static 50/50 reading is uninformative, not a refusal.
    ctl_flat = LowerBodyController(com_model=NaoCoMModel())
    _, flat, _, _ = run(ctl_flat, lifting("L"), 3.0, fsr={"L": 26.0, "R": 26.0})
    assert flat["lift"] > 0.0

    # Once the load really has transferred, the full lift is allowed.
    ctl2 = LowerBodyController(com_model=NaoCoMModel())
    _, confirmed, _, _ = run(ctl2, lifting("L"), 3.0, fsr={"L": 2.0, "R": 25.0})
    assert confirmed["gate"] > unconfirmed["gate"]
    assert confirmed["lift"] > unconfirmed["lift"]


def test_negligible_total_foot_load_is_ignored_not_trusted() -> None:
    """Airborne / uncalibrated sensors must not be read as a veto."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, lifting("L"), 3.0, fsr={"L": 0.0, "R": 0.0})
    assert meta["gate"] > 0.0


def test_without_a_com_model_the_lift_is_capped_not_cancelled() -> None:
    ctl = LowerBodyController(com_model=None)
    _, meta, _, _ = run(ctl, lifting("L"), 3.0)
    assert 0.0 < meta["lift"] <= ctl.params.ungated_lift_cap + 1e-6


def test_reset_clears_a_half_finished_transfer() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, lifting("L"), 3.0)
    assert meta["mode"] == MODE_SINGLE
    ctl.reset()
    _, meta = ctl.step(99.0, lifting("L"), measured=rest_state())
    assert meta["shift"] == 0.0 and meta["lift"] == 0.0


# -------------------------------------------------------------------- limits
def test_every_output_respects_nao_joint_limits() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    extreme = LegTarget(-3.0, 3.0, 5.0, -5.0, -3.0, lift=1.0, confidence=1.0)
    obs = LowerBodyObservation(
        left=extreme, right=LegTarget(0, 0, 0, 0, 0, 0.0, 1.0),
        crouch_u=5.0, stance_side="R", confidence=1.0, valid=True,
    )
    targets, _, _, _ = run(ctl, obs, 3.0, yaw_bias=9.0)
    assert set(LEG_JOINTS) <= set(targets)
    for name, value in targets.items():
        cfg = CONFIGS[name]
        assert cfg.min_angle - 1e-9 <= value <= cfg.max_angle + 1e-9, name
        assert math.isfinite(value)


def test_yaw_bias_is_capped_and_only_applied_while_standing() -> None:
    p = LowerBodyParams()
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, standing(), 1.0, yaw_bias=5.0)
    assert meta["mode"] == MODE_DOUBLE
    assert abs(targets["LHipYawPitch"] - p.max_yaw_bias) < 1e-6
    # Mid-step the shared hip yaw must stay out of it: it splays the legs.
    ctl2 = LowerBodyController(com_model=NaoCoMModel())
    targets2, meta2, _, _ = run(ctl2, lifting("L"), 3.0, yaw_bias=5.0)
    assert meta2["mode"] == MODE_SINGLE
    assert abs(targets2["LHipYawPitch"]) < 1e-9


# ---------------------------------------------------------------------------
# Symmetric / antisymmetric authority split
#
# Spreading your legs and leaning are both "asymmetric per-leg roll" to a naive
# reader, but one enlarges the support polygon and the other moves the centre of
# mass off it. Gating them equally is what turned a 40 deg human stance into a
# 12 deg robot one.
# ---------------------------------------------------------------------------
def test_a_wider_stance_passes_through_at_full_authority() -> None:
    p = LowerBodyParams()
    for angle in (0.10, 0.20, 0.35):
        ctl = LowerBodyController(com_model=NaoCoMModel())
        targets, meta, _, _ = run(ctl, abducted(angle), 2.0)
        assert meta["mode"] == MODE_DOUBLE
        want = min(angle, p.max_abduction)
        assert targets["LHipRoll"] == pytest.approx(+want, abs=0.02)
        assert targets["RHipRoll"] == pytest.approx(-want, abs=0.02)


def test_a_wider_stance_keeps_both_soles_flat() -> None:
    """The ankle must cancel the hip, or the robot stands on its inner edges."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, _, _, _ = run(ctl, abducted(0.35), 2.0)
    for side in ("L", "R"):
        assert targets[f"{side}HipRoll"] + targets[f"{side}AnkleRoll"] == \
            pytest.approx(0.0, abs=1e-6)


def test_abduction_is_capped_by_the_ankle_plus_a_tilt_budget() -> None:
    """NAO's HipRoll reaches 45 deg but AnkleRoll only 22.8, and past that the sole
    cannot be levelled. Stopping at 22.8 saturated below what people actually do,
    so a small bounded sole tilt is spent to buy the extra width."""
    p = LowerBodyParams()
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, _, _, _ = run(ctl, abducted(1.5), 2.0)      # absurd request
    assert targets["LHipRoll"] == pytest.approx(p.max_abduction, abs=0.02)
    assert p.max_abduction < CONFIGS["LHipRoll"].max_angle   # hip could go further
    assert p.max_abduction == pytest.approx(
        ANKLE_ROLL_LIMIT + p.sole_tilt_budget, abs=1e-9)
    # ... and it reaches past the ankle's own range, which is the whole point.
    assert p.max_abduction > ANKLE_ROLL_LIMIT
    # Recorded runs show subjects at ~25 deg (p99); the cap must clear that.
    assert p.max_abduction > math.radians(25)


def test_the_sole_tilt_budget_is_enforced_as_a_post_condition() -> None:
    """Whatever the symmetric, antisymmetric and balance terms add up to, no sole
    may end up further than the budget from flat -- standing on the edge of a foot
    is what tips the robot."""
    p = LowerBodyParams()
    for obs in (abducted(1.5), leaning(1.5), abducted(1.2), split_stance(1.5)):
        ctl = LowerBodyController(com_model=NaoCoMModel())
        targets, _, _, _ = run(ctl, obs, 2.0)
        for side in ("L", "R"):
            tilt = targets[f"{side}HipRoll"] + targets[f"{side}AnkleRoll"]
            assert abs(tilt) <= p.sole_tilt_budget + 1e-6, (side, tilt)


def test_stance_width_gives_way_before_sole_contact() -> None:
    """When the budget binds it is the HIP that backs off, not the ankle."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, _, _, _ = run(ctl, abducted(1.5), 2.0)
    # The ankle is at its hardware limit, doing all it can to level the sole.
    assert targets["LAnkleRoll"] == pytest.approx(
        CONFIGS["LAnkleRoll"].min_angle, abs=1e-6)


def test_a_holdable_lean_is_imitated_ONE_TO_ONE() -> None:
    """Balance compensates; it does not attenuate.

    This replaces a test that asserted the opposite -- that a lean was scaled to
    35% because same-sign roll moves the centre of mass. Attenuating was the wrong
    tool: a third of a lean still moves the CoM a third of the way out, so it
    bought fidelity loss without buying balance. The robot now strikes the pose and
    shifts its pelvis (and widens its stance) to make it holdable, which is what a
    person does when they lean.

    0.35 rad is inside what NAO can hold: a lean moves the CoM by
    leg_length * sin(theta), and at 0.35 rad that is 96 mm against an 88 mm
    polygon half-width plus the compensation's travel.
    """
    model = NaoCoMModel()
    ctl = LowerBodyController(com_model=model)
    targets, meta, _, _ = run(ctl, leaning(0.35), 5.0)
    lean = 0.5 * (targets["LHipRoll"] + targets["RHipRoll"])
    assert lean == pytest.approx(0.35, abs=0.03), lean      # 1:1, not 35%
    assert meta["lean_scale"] == 1.0                         # nothing was gated
    assert meta["com_margin"] > 0.0                          # and it is holdable


def test_a_lean_past_the_hardware_limit_is_attenuated_as_a_LAST_resort() -> None:
    """Past what the robot can physically hold, the pose has to give.

    A lean moves the CoM by leg_length * sin(theta) -- 158 mm at 0.6 rad, against
    an 88 mm polygon half-width. No compensation can hold that; the robot would
    have to step. So the pose is scaled back, but only after the compensation has
    been given com_grace_s to try.
    """
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, leaning(0.9), 5.0)
    assert meta["lean_scale"] < 1.0


def test_the_compensation_gets_a_grace_period_before_anything_is_scaled() -> None:
    """The fallback must not fire on the first tick -- the pelvis needs a few ticks
    to travel, and firing early is gating by another name."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    t = 0.0
    obs = leaning(0.9)
    for _ in range(5):                       # 0.1s, well inside com_grace_s
        t += 0.02
        _, meta = ctl.step(t, obs, measured={})
    assert meta["lean_scale"] == 1.0, "scaled back before the compensation could try"


def test_a_split_stance_is_still_gated() -> None:
    p = LowerBodyParams()
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, _, _, _ = run(ctl, split_stance(0.5), 2.0)
    spread = targets["LHipPitch"] - targets["RHipPitch"]
    assert spread == pytest.approx(-2 * p.asymmetric_gain * 0.5, abs=0.05)


def test_a_deeper_squat_is_symmetric_and_torso_stays_vertical() -> None:
    obs = LowerBodyObservation(
        left=LegTarget(-0.4, 0.0, 0.8, -0.4, 0.0, 0.0, 1.0),
        right=LegTarget(-0.4, 0.0, 0.8, -0.4, 0.0, 0.0, 1.0),
        crouch_u=0.30, stance_side="", confidence=1.0, valid=True,
        lift_source="feet",
    )
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, _, _, _ = run(ctl, obs, 2.0)
    assert targets["LHipPitch"] == pytest.approx(targets["RHipPitch"], abs=1e-9)
    for side in ("L", "R"):
        total = (targets[f"{side}HipPitch"] + targets[f"{side}KneePitch"]
                 + targets[f"{side}AnklePitch"])
        assert total == pytest.approx(0.0, abs=1e-9)


def test_only_one_visible_leg_is_read_conservatively() -> None:
    """With one leg out of view a symmetric pose and a lean are indistinguishable,
    so the whole deviation must be treated as the risky (antisymmetric) one."""
    p = LowerBodyParams()
    obs = LowerBodyObservation(
        left=LegTarget(0.0, 0.35, 0.0, 0.0, -0.35, 0.0, 1.0),
        right=None, crouch_u=0.0, confidence=0.6, valid=True, lift_source="feet",
    )
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, obs, 2.0)
    # The DEVIATION is what this test is about: read as antisymmetric (the
    # conservative reading) and applied to the visible leg only. Both hips also
    # carry the CoM compensation's same-sign roll on top, so the two are separated
    # by looking at the difference rather than at the absolute values.
    # (Whether it is then attenuated is a separate question, covered by the
    # compensation tests: a one-sided roll is half lean and half stance width, so
    # how holdable it is depends on the geometry, not on this decomposition.)
    # The L-R DIFFERENCE isolates the deviation from everything added same-sign on
    # top (the CoM compensation, and the lean scaling if it fired -- neither of
    # which changes a difference).
    deviation = targets["LHipRoll"] - targets["RHipRoll"]
    assert deviation == pytest.approx(p.asymmetric_gain * 0.35, abs=0.06), deviation


def test_a_lifted_leg_does_not_drag_the_stance_leg_with_it() -> None:
    """The swing foot is unloaded and free; the stance leg is carrying the robot,
    so the swing leg's pose must not be shared onto it."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, lifting("L"), 3.0)
    assert meta["mode"] == MODE_SINGLE
    assert targets["LKneePitch"] > 0.9              # swing leg clearly folded
    assert abs(targets["RKneePitch"]) < 0.35        # stance leg near the crouch


def test_a_full_weight_transfer_allows_the_full_requested_lift() -> None:
    """A completed transfer yields ~0.015 m of margin; if margin_full sat above
    that, the lift silently capped below the human's and read as "barely moves"."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, lifting("L", lift=1.0), 4.0)
    assert meta["gate"] == pytest.approx(1.0, abs=1e-6)
    assert meta["lift"] > 0.95


# ---------------------------------------------------------------------------
# The lean is verified against the model, not just gain-limited
# ---------------------------------------------------------------------------
def _leaning(roll, conf=1.0):
    """Both legs rolled the SAME way: an antisymmetric pose, i.e. a lean."""
    left = LegTarget(0.0, roll, 0.0, 0.0, -roll, lift=0.0, confidence=conf)
    right = LegTarget(0.0, roll, 0.0, 0.0, -roll, lift=0.0, confidence=conf)
    return LowerBodyObservation(
        left=left, right=right, crouch_u=0.0, stance_side="",
        confidence=conf, valid=True, lift_source="feet",
    )


def test_a_lean_the_robot_cannot_hold_is_scaled_back() -> None:
    """The gap this closes: the antisymmetric deviation was gated only by a fixed
    GAIN, with nothing consulting the robot's own model -- and this module's whole
    premise is that stability decisions are made from the robot's own state.

    At the shipped ``asymmetric_gain`` of 0.35 the gain alone happens to keep the
    lean inside the polygon (0.277 rad, ~39 mm of margin), so the guard is a safety
    net rather than an attenuator -- see the companion test below. It is exercised
    here by raising the gain, which is a supported configuration and exactly the
    change that would silently reintroduce a fall.
    """
    model = NaoCoMModel()
    loose = LowerBodyParams(asymmetric_gain=1.0)
    ctl = LowerBodyController(params=loose, com_model=model)
    targets, meta, _, _ = run(ctl, _leaning(2.0), 3.0)      # an extreme lean request
    assert meta["lean_scale"] < 1.0                          # it really was reduced

    # The contract: the lean is scaled back until the model is satisfied, or right
    # down to nothing. It cannot promise the margin outright, because NAO's roll
    # limits are asymmetric (LHipRoll reaches +45.3 deg, RHipRoll only +21.7), so
    # clamping an extreme request leaves a residual sole tilt that costs margin on
    # its own and is not a lean for this method to remove.
    # The contract is about the POSTURE being holdable, not about the lean reaching
    # any particular number: the guard stops scaling as soon as the model is
    # satisfied, so a more accurate support polygon simply means less scaling is
    # needed. What must hold is that the guard leaves the robot better supported
    # than the same request would unguarded.
    ungated = LowerBodyController(params=loose, com_model=None)
    bad, bad_meta, _, _ = run(ungated, _leaning(2.0), 3.0)
    assert bad_meta["lean_scale"] == 1.0                     # nothing checked it
    bad_lean = 0.5 * (bad["LHipRoll"] + bad["RHipRoll"])
    lean = 0.5 * (targets["LHipRoll"] + targets["RHipRoll"])
    assert bad_lean > 0.4                                    # a big unchecked lean
    assert abs(lean) < bad_lean                              # the guard reduced it
    assert model.support_margin(dict(targets)) > model.support_margin(dict(bad))


def test_a_lean_the_robot_CAN_hold_passes_through_untouched() -> None:
    """Inside the hardware limit, nothing is scaled and the pose arrives whole."""
    model = NaoCoMModel()
    for request in (0.10, 0.20, 0.30):
        ctl = LowerBodyController(com_model=model)
        targets, meta, _, _ = run(ctl, _leaning(request), 5.0)
        assert meta["lean_scale"] == 1.0, request
        lean = 0.5 * (targets["LHipRoll"] + targets["RHipRoll"])
        assert lean == pytest.approx(request, abs=0.03), (request, lean)


def test_stance_widening_is_preferred_over_fighting_the_lean() -> None:
    """Why width is one of the compensation's parameters.

    A pelvis shift and a lean are the same degree of freedom, so shifting the
    pelvis to hold a lean subtracts directly from the lean being imitated.
    Widening the stance enlarges the support polygon instead and is
    mirror-symmetric, so it costs the imitation nothing. Measured holding a 0.60
    rad lean: shift -0.20 buys +0.025 m of margin but loses 0.20 of lean, while
    width +0.20 buys +0.031 m and loses none.
    """
    model = NaoCoMModel()
    ctl = LowerBodyController(com_model=model)
    _, meta, _, _ = run(ctl, _leaning(0.55), 5.0)
    # It reached for width, not only for a pelvis shift.
    assert meta["com_shift_width"] > 0.0, meta


def test_the_compensation_costs_nothing_on_a_comfortable_pose() -> None:
    """A pose that stands up on its own must not be taxed -- otherwise every
    movement pays for balance it does not need, and the tick cost goes up too."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, standing(), 3.0)
    assert abs(meta["com_shift_pitch"]) < 1e-3
    assert abs(meta["com_shift_roll"]) < 1e-3
    assert abs(meta["com_shift_width"]) < 1e-3


def test_the_compensation_keeps_both_soles_flat() -> None:
    """It translates the pelvis; it must never tilt a sole to do it. A tilted sole
    contacts along one edge and collapses the very polygon being defended."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    t = 0.0
    for _ in range(250):
        t += 0.02
        targets, _ = ctl.step(t, _leaning(0.5), measured={})
        for side in ("L", "R"):
            tilt = targets[f"{side}HipRoll"] + targets[f"{side}AnkleRoll"]
            assert abs(tilt) <= LowerBodyParams().sole_tilt_budget + 1e-6, (side, tilt)


def test_stance_width_is_never_scaled_by_the_lean_limit() -> None:
    """A wider stance is mirror-symmetric: CoM-neutral, and it ENLARGES the
    polygon. It must not be caught by a limit aimed at leaning."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    wide = LegTarget(0.0, +0.60, 0.0, 0.0, -0.60, lift=0.0, confidence=1.0)
    narrow = LegTarget(0.0, -0.60, 0.0, 0.0, +0.60, lift=0.0, confidence=1.0)
    obs = LowerBodyObservation(
        left=wide, right=narrow, crouch_u=0.0, stance_side="",
        confidence=1.0, valid=True, lift_source="feet",
    )
    targets, meta, _, _ = run(ctl, obs, 3.0)
    assert meta["lean_scale"] == 1.0
    # And the stance really is wide: the rolls are opposite-signed and large.
    assert targets["LHipRoll"] > 0.2
    assert targets["RHipRoll"] < -0.2


def test_the_lean_limit_does_not_fight_the_weight_transfer() -> None:
    """During a shift the lean is deliberate -- its purpose IS to move the CoM
    onto one foot, so the double-support margin is the wrong test. stance_margin
    in _lift_gate is the right one and is already applied."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, meta, _, _ = run(ctl, lifting("L"), 3.0)
    assert meta["shift"] > 0.5                    # a transfer is under way
    assert meta["lean_scale"] == 1.0              # and it was left alone


def test_the_lean_limit_degrades_to_the_old_behaviour_without_a_model() -> None:
    """No NumPy, no model, no check -- but it must not crash or freeze the legs."""
    ctl = LowerBodyController(com_model=None)
    targets, meta, _, _ = run(ctl, _leaning(2.0), 3.0)
    assert meta["lean_scale"] == 1.0
    assert set(LEG_JOINTS) <= set(targets)


def test_a_marginal_frame_does_not_abort_a_weight_transfer() -> None:
    """The retargeter halves its confidence when only one leg is in view, which
    puts a clear single-leg detection at ~0.49 against a 0.50 gate. Without
    hysteresis the layer flapped double -> load -> double, restarting the transfer
    every time -- visible as legs that twitch instead of stepping."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    p = ctl.params
    # Establish a transfer with two legs clearly visible.
    good = lifting("L", conf=1.0)
    t = 0.0
    for _ in range(60):
        t += 0.02
        _, meta = ctl.step(t, good)
    assert meta["shift"] > 0.5, meta

    # Now confidence dips to just under the ENGAGE bar but above the KEEP bar.
    marginal = lifting("L", conf=0.49)
    assert p.conf_keep <= 0.49 < p.conf_min
    for _ in range(30):
        t += 0.02
        _, meta = ctl.step(t, marginal)
    assert meta["tracking"] is True                # still engaged
    assert meta["shift"] > 0.5, meta               # the transfer survived

    # A genuine collapse in confidence still stands the robot down.
    bad = lifting("L", conf=0.20)
    for _ in range(200):
        t += 0.02
        _, meta = ctl.step(t, bad)
    assert meta["tracking"] is False
    assert meta["shift"] < 0.05


def test_engaging_still_needs_the_full_confidence_bar() -> None:
    """Hysteresis must lower the bar to CONTINUE, never to start."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    marginal = lifting("L", conf=0.45)
    t = 0.0
    for _ in range(80):
        t += 0.02
        _, meta = ctl.step(t, marginal)
    assert meta["tracking"] is False
    assert meta["shift"] < 1e-3


def test_a_split_stance_is_checked_too_not_just_a_lean() -> None:
    """The guard covers every antisymmetric channel, not only roll.

    A split stance -- one leg forward, one back -- moves the CoM fore/aft and
    rotates the feet, shrinking the polygon on the axis where NAO has least to
    give (30 mm behind the ankle against 100 mm in front). Checking only the roll
    lean left 15 of 204 double-support frames outside the support polygon on
    recorded data, every one of them with the roll guard already fully applied.
    """
    model = NaoCoMModel()
    loose = LowerBodyParams(asymmetric_gain=1.0)
    # Antisymmetric PITCH: left leg flexed forward, right extended back.
    fwd = LegTarget(-1.2, 0.0, 0.0, 0.0, 0.0, lift=0.0, confidence=1.0)
    back = LegTarget(+0.4, 0.0, 0.0, 0.0, 0.0, lift=0.0, confidence=1.0)
    obs = LowerBodyObservation(
        left=fwd, right=back, crouch_u=0.0, stance_side="",
        confidence=1.0, valid=True, lift_source="feet",
    )
    guarded = LowerBodyController(params=loose, com_model=model)
    targets, meta, _, _ = run(guarded, obs, 3.0)

    ungated = LowerBodyController(params=loose, com_model=None)
    bad, bad_meta, _, _ = run(ungated, obs, 3.0)
    assert bad_meta["lean_scale"] == 1.0

    # The guarded posture must be at least as well supported as the unguarded one,
    # and if the guard acted at all it must have improved matters.
    assert model.support_margin(dict(targets)) >= model.support_margin(dict(bad)) - 1e-9
    if meta["lean_scale"] < 1.0:
        assert model.support_margin(dict(targets)) > model.support_margin(dict(bad))


def test_the_symmetric_squat_is_never_scaled_by_the_asymmetry_guard() -> None:
    """A deep squat is mirror-symmetric: CoM-neutral, and the crouch posture keeps
    the ankle under the hip at any depth. It must reach the robot untouched."""
    ctl = LowerBodyController(com_model=NaoCoMModel())
    targets, meta, _, _ = run(ctl, squatting(0.60), 3.0)
    assert meta["lean_scale"] == 1.0
    assert meta["crouch_u"] == pytest.approx(0.60, abs=1e-6)
    assert targets["LHipPitch"] == pytest.approx(targets["RHipPitch"], abs=1e-9)
