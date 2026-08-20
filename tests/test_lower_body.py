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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import NaoCoMModel  # noqa: E402
from lower_body import (  # noqa: E402
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


def test_squat_follows_the_human_but_stays_bounded() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    _, shallow, _, _ = run(ctl, standing(crouch_u=0.05), 1.0)
    ctl2 = LowerBodyController(com_model=NaoCoMModel())
    _, deep, _, _ = run(ctl2, standing(crouch_u=10.0), 1.0)   # absurd request
    assert shallow["crouch_u"] == ctl.params.base_crouch_u    # below the floor
    assert deep["crouch_u"] == ctl2.params.max_crouch_u       # capped


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


def test_foot_sensors_veto_a_lift_when_the_load_is_not_transferred() -> None:
    ctl = LowerBodyController(com_model=NaoCoMModel())
    # The stance foot (R) is carrying almost nothing -> refuse to unload the other.
    _, meta, _, _ = run(ctl, lifting("L"), 3.0, fsr={"L": 25.0, "R": 2.0})
    assert meta["gate"] == 0.0
    assert meta["lift"] == 0.0
    # ... and it goes ahead once the sensors agree with the model.
    ctl2 = LowerBodyController(com_model=NaoCoMModel())
    _, meta2, _, _ = run(ctl2, lifting("L"), 3.0, fsr={"L": 2.0, "R": 25.0})
    assert meta2["gate"] > 0.0


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
