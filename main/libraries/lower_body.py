"""Balance-aware lower-body controller: human leg pose -> safe NAO leg motion.

``nao_retarget.LowerBodyRetargeter`` answers *what the human's legs are doing*.
This module answers the separate, robot-side question: **how much of that may the
robot actually execute right now without falling over?**

Why the two are separated
-------------------------
A camera cannot see whether NAO's centre of mass is over a foot -- that depends on
the robot's own configuration and mass distribution. So the retargeter stays a
pure kinematic observer, and every stability decision is made here from the
robot's own state (``balance.NaoCoMModel`` forward kinematics, the InertialUnit
tilt, and the foot force sensors when the model provides them).

The step sequence
-----------------
Lifting a foot on a free-standing biped is not one action but three, and the
previous code skipped the first two -- which is why raising a leg in front of the
camera produced nothing:

1. **LOAD**  - lean the body so the centre of mass moves over the *stance* foot.
   The lean direction is not hard-coded: we probe both signs against the CoM
   model and keep the one that actually raises the stance-foot margin (a
   hard-coded lean sign is the classic way to make a balance controller tip
   *faster*, and this makes that failure impossible).
2. **SINGLE** - only once the model reports positive stance margin (and the foot
   force sensors, when present, confirm the transfer) does the swing leg start to
   follow the human's leg, its authority scaled *continuously* by that margin.
3. **UNLOAD** - the human lowers the leg, or the margin/tilt safety closes the
   gate; both blends ramp back down to the symmetric crouch.

Everything is expressed as two rate-limited blends (``shift`` and ``lift``), so
there are no discrete jumps and no state to get stuck in: the controller can
always ramp back to the exact symmetric crouch that is the project's proven
no-fall baseline.

Symmetric / asymmetric split
----------------------------
The commanded posture is ``crouch_posture(u)`` -- the statically-balanced squat
whose ``Hip + Knee + Ankle == 0`` keeps the torso vertical and the soles flat --
plus the human's *deviation from* that posture, per leg, authority-weighted. The
symmetric part therefore always stays inside the proven-stable family, and only
the asymmetric detail is gated. Standing still yields a deviation of exactly
zero, so the controller degrades to the old baseline rather than to noise.

Pure Python + the (optional) NumPy CoM model, no Webots import, so all of the
sequencing and gating logic is unit-testable off-simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from nao_retarget import LegTarget, LowerBodyObservation, crouch_posture
from pose_control_utils import JointLimiter, get_default_motor_configs

# The 12 leg joints this controller owns while it is active.
LEG_JOINTS = (
    "LHipYawPitch", "RHipYawPitch",
    "LHipRoll", "RHipRoll",
    "LHipPitch", "RHipPitch",
    "LKneePitch", "RKneePitch",
    "LAnklePitch", "RAnklePitch",
    "LAnkleRoll", "RAnkleRoll",
)

MODE_DOUBLE = "double"
MODE_LOAD = "load"
MODE_SINGLE = "single"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class LowerBodyParams:
    """Tuning for :class:`LowerBodyController` (radians / seconds)."""

    # -- posture -----------------------------------------------------------
    base_crouch_u: float = 0.10     # knees never fully locked: leaves the
                                    # balance loop some authority to work with
    max_crouch_u: float = 0.35      # deep squats are where NAO goes marginal

    # -- how much asymmetric human detail we allow, per channel ------------
    max_hip_pitch_dev: float = 0.90
    max_hip_roll_dev: float = 0.35
    max_knee_dev: float = 1.10
    max_ankle_dev: float = 0.60
    # Authority given to per-leg deviations while BOTH feet are loaded. Small on
    # purpose: an asymmetric double-support pose moves the CoM but cannot be
    # verified against a single-foot support polygon.
    double_support_gain: float = 0.30

    # -- weight transfer ---------------------------------------------------
    shift_rad: float = 0.16         # lean amplitude that loads the stance foot
    shift_rate: float = 1.1         # 1/s ramp of the shift blend
    shift_ready: float = 0.75       # shift blend required before lifting starts
    lift_rate: float = 1.8          # 1/s ramp of the lift blend
    lift_start: float = 0.22        # human lift fraction that requests a step
    lift_stop: float = 0.10         # hysteresis: below this the foot comes down

    # -- safety gates ------------------------------------------------------
    margin_min: float = 0.003       # m; stance margin needed to begin lifting
    margin_full: float = 0.018      # m; margin at which full lift is allowed
    fsr_load_frac: float = 0.55     # stance foot's share of total foot load
    fsr_total_min: float = 1.0      # N; below this the FSR reading is ignored
    tilt_abort_rad: float = 0.28    # |IMU roll/pitch| beyond this -> stand down
    conf_min: float = 0.50          # min lower-body landmark confidence

    # -- standing turn -----------------------------------------------------
    # Shared-hip-yaw bias used for immediate visual feedback while a stepping
    # turn is unavailable. Deliberately tiny: NAO's HipYawPitch axis is canted
    # 45 deg, so it splays the legs as well as yawing the pelvis.
    max_yaw_bias: float = 0.12

    # -- misc --------------------------------------------------------------
    max_dt_s: float = 0.1
    # Lift authority when no CoM model is available (numpy missing). We do not
    # refuse to move -- that is what made the legs look dead -- but we cap the
    # motion hard and the tilt abort remains the safety net.
    ungated_lift_cap: float = 0.35
    # Probe amplitude used to discover the lean sign from the CoM model.
    probe_rad: float = 0.10
    probe_refresh_s: float = 0.25


@dataclass
class LowerBodyState:
    shift: float = 0.0        # [0, 1] weight-transfer blend
    lift: float = 0.0         # [0, 1] swing-leg authority
    stance: str = ""          # "L" / "R" / "" (double support)
    last_now: Optional[float] = None


class LowerBodyController:
    """Turns a :class:`LowerBodyObservation` into safe NAO leg targets.

    Usage (per simulation step, whether or not a fresh camera frame arrived)::

        targets, meta = controller.step(now_s, observation,
                                        torso_rp=(roll, pitch),
                                        fsr={"L": fz, "R": fz},
                                        measured=current_joint_angles)

    ``meta["balance_ok"]`` tells the caller whether the *symmetric* CoM balance
    correction from ``balance.BalanceController`` is still a valid thing to add
    on top (it is not, once we are deliberately leaning onto one foot).
    """

    def __init__(
        self,
        params: Optional[LowerBodyParams] = None,
        *,
        com_model: Optional[object] = None,
        limiter: Optional[JointLimiter] = None,
    ) -> None:
        self.params = params or LowerBodyParams()
        self.limiter = limiter or JointLimiter(get_default_motor_configs())
        self.com_model = com_model
        self.state = LowerBodyState()
        self._probe: Dict[str, Tuple[float, float]] = {}  # stance -> (dir, t)
        self._last_obs = LowerBodyObservation()

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Drop all blend state (call after an external whole-body motion).

        A motion clip leaves the robot in a completely different configuration,
        so a half-finished weight transfer from before the clip must not be
        resumed on top of it.
        """
        self.state = LowerBodyState()
        self._probe.clear()

    def set_observation(self, obs: Optional[LowerBodyObservation]) -> None:
        """Latch the newest camera observation (control runs at sim rate)."""
        if obs is not None:
            self._last_obs = obs

    def stand_down(self) -> None:
        """Forget the latched observation so the sequencer ramps to the crouch.

        Call this when tracking goes stale. The latch exists so control keeps
        running at simulation rate between camera frames -- but without an
        expiry the controller would go on acting on a snapshot of a human who
        has left, holding a one-legged stance indefinitely. The blends still ramp
        down rather than snapping, so the foot is set down, not dropped.
        """
        self._last_obs = LowerBodyObservation()

    def step(
        self,
        now_s: float,
        obs: Optional[LowerBodyObservation] = None,
        *,
        torso_rp: Tuple[float, float] = (0.0, 0.0),
        fsr: Optional[Dict[str, float]] = None,
        measured: Optional[Dict[str, float]] = None,
        yaw_bias: float = 0.0,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        """Advance the sequencer one control step and emit leg targets."""
        p = self.params
        st = self.state
        if obs is not None:
            self._last_obs = obs
        obs = self._last_obs

        dt = 0.0 if st.last_now is None else _clamp(now_s - st.last_now, 0.0, p.max_dt_s)
        st.last_now = now_s

        roll, pitch = torso_rp
        tilt_ok = abs(roll) < p.tilt_abort_rad and abs(pitch) < p.tilt_abort_rad
        usable = bool(obs.valid and obs.confidence >= p.conf_min and tilt_ok)

        swing = self._requested_swing(obs) if usable else ""
        if swing:
            st.stance = "R" if swing == "L" else "L"
        elif st.lift <= 1e-3 and st.shift <= 1e-3:
            st.stance = ""

        # --- weight transfer blend ---------------------------------------
        shift_target = 1.0 if swing else 0.0
        st.shift = _approach(st.shift, shift_target, p.shift_rate * dt)

        # --- lift blend, gated by the robot's own balance state -----------
        gate, margin = self._lift_gate(st.stance, measured, fsr)
        human_lift = 0.0
        if swing:
            leg = obs.leg(swing)
            human_lift = leg.lift if leg is not None else 0.0
        ready = st.shift >= p.shift_ready
        lift_target = (human_lift * gate) if (swing and ready) else 0.0
        st.lift = _approach(st.lift, lift_target, p.lift_rate * dt)

        mode = (
            MODE_SINGLE if st.lift > 0.05
            else (MODE_LOAD if st.shift > 0.05 else MODE_DOUBLE)
        )

        # Keep the lean sign fresh for whichever foot is currently the stance
        # foot (cheap: cached for ``probe_refresh_s``).
        if st.stance:
            self._shift_direction(st.stance, measured, now_s)

        targets = self._compose(obs, swing, yaw_bias if mode == MODE_DOUBLE else 0.0)
        clamped = {n: self.limiter.clamp_angle(n, v) for n, v in targets.items()}
        meta: Dict[str, object] = {
            "why": self._explain(obs, tilt_ok, swing, gate, human_lift),
            "lift_source": obs.lift_source,
            "mode": mode,
            "single_support": mode == MODE_SINGLE,
            "balance_ok": mode == MODE_DOUBLE and st.shift < 0.05,
            "swing_side": swing,
            "stance_side": st.stance,
            "shift": round(st.shift, 4),
            "lift": round(st.lift, 4),
            "gate": round(gate, 4),
            "stance_margin": round(margin, 5),
            "human_lift": round(human_lift, 4),
            "crouch_u": round(self._crouch_u(obs), 4),
            "tilt_ok": tilt_ok,
            "tracking": usable,
        }
        return clamped, meta

    # ------------------------------------------------------------- internals
    def _explain(self, obs: LowerBodyObservation, tilt_ok: bool, swing: str,
                 gate: float, human_lift: float) -> str:
        """One short phrase naming the current limiting factor.

        Worth its keep: "the legs are not moving" has half a dozen legitimate
        causes (nobody in frame, legs cropped, low confidence, the CoM not yet
        over the stance foot) and they are indistinguishable from a bug unless
        the controller says which one it is.
        """
        p = self.params
        st = self.state
        if not obs.valid:
            return "no lower-body landmarks in frame"
        if obs.confidence < p.conf_min:
            return f"lower-body confidence {obs.confidence:.2f} < {p.conf_min:.2f}"
        if not tilt_ok:
            return "torso tilted past the safety limit; standing down"
        if obs.lift_source == "none":
            return "knees and feet both out of frame; leg lift cannot be seen"
        if not swing:
            if human_lift <= 0.0 and st.lift <= 1e-3:
                return "tracking (no leg lift requested)"
            return "returning the foot to the ground"
        if st.shift < p.shift_ready:
            return f"transferring weight onto the {st.stance or '?'} foot"
        if gate <= 0.0:
            return "holding: centre of mass not yet over the stance foot"
        return f"stepping ({st.lift * 100:.0f}% of the requested lift)"

    def _crouch_u(self, obs: LowerBodyObservation) -> float:
        p = self.params
        u = obs.crouch_u if obs.valid else 0.0
        return _clamp(max(u, p.base_crouch_u), 0.0, p.max_crouch_u)

    def _requested_swing(self, obs: LowerBodyObservation) -> str:
        """Which foot (if any) the human is asking the robot to lift."""
        p = self.params
        lifts = {s: (obs.leg(s).lift if obs.leg(s) is not None else 0.0) for s in ("L", "R")}
        side = "L" if lifts["L"] >= lifts["R"] else "R"
        # Hysteresis: it takes a clear lift to start a step and a clear return
        # to the ground to end it, so a foot hovering near the threshold does
        # not chatter the weight transfer.
        already = self.state.lift > 1e-3 or self.state.shift > 1e-3
        threshold = p.lift_stop if already else p.lift_start
        if lifts[side] < threshold:
            return ""
        # Both feet "lifted" is not a step -- it is a jump, or bad tracking.
        other = "R" if side == "L" else "L"
        if lifts[other] >= threshold and abs(lifts[side] - lifts[other]) < 0.08:
            return ""
        return side

    def _lift_gate(
        self,
        stance: str,
        measured: Optional[Dict[str, float]],
        fsr: Optional[Dict[str, float]],
    ) -> Tuple[float, float]:
        """Return ``(gate, stance_margin_m)`` -- how much lift is safe right now.

        ``gate`` scales the swing leg's authority continuously from 0 (CoM not
        over the stance foot: do not unload it) to 1 (comfortably over it), so
        the step degrades smoothly instead of snapping on and off.
        """
        p = self.params
        if not stance:
            return 0.0, 0.0

        # Foot force sensors, when the robot has them, must confirm the transfer.
        if fsr:
            total = float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))
            if total > p.fsr_total_min:
                share = float(fsr.get(stance, 0.0)) / total
                if share < p.fsr_load_frac:
                    return 0.0, 0.0

        if self.com_model is None or measured is None:
            # No model to prove safety with: move, but only a little.
            return p.ungated_lift_cap, 0.0
        try:
            margin = float(self.com_model.stance_margin(measured, stance))
        except Exception:  # noqa: BLE001 - model without stance_margin / bad state
            return p.ungated_lift_cap, 0.0
        span = max(p.margin_full - p.margin_min, 1e-6)
        return _clamp((margin - p.margin_min) / span, 0.0, 1.0), margin

    def _shift_direction(self, stance: str, measured: Optional[Dict[str, float]],
                         now_s: float) -> float:
        """Sign of the same-sign roll lean that loads ``stance``.

        Discovered from the CoM model instead of hard-coded (see the module
        docstring). Cached briefly because it only changes with the posture.
        """
        p = self.params
        cached = self._probe.get(stance)
        if cached is not None and (now_s - cached[1]) < p.probe_refresh_s:
            return cached[0]

        # Documented fallback if there is no model to ask: leaning the hips
        # toward +roll carries the pelvis away from that side, so loading the
        # left foot needs a negative same-sign roll.
        direction = -1.0 if stance == "L" else 1.0
        if self.com_model is not None and measured:
            best_margin = -math.inf
            for d in (1.0, -1.0):
                probe = dict(measured)
                for j in ("LHipRoll", "RHipRoll"):
                    probe[j] = probe.get(j, 0.0) + d * p.probe_rad
                for j in ("LAnkleRoll", "RAnkleRoll"):
                    probe[j] = probe.get(j, 0.0) - d * p.probe_rad
                try:
                    m = float(self.com_model.stance_margin(probe, stance))
                except Exception:  # noqa: BLE001
                    best_margin = -math.inf
                    break
                if m > best_margin:
                    best_margin, direction = m, d
        self._probe[stance] = (direction, now_s)
        return direction

    def _compose(
        self, obs: LowerBodyObservation, swing: str, yaw_bias: float
    ) -> Dict[str, float]:
        """Symmetric crouch + authority-weighted per-leg human deviation + lean."""
        p = self.params
        st = self.state
        u = self._crouch_u(obs)
        targets = dict(crouch_posture(u))

        for side in ("L", "R"):
            leg = obs.leg(side) if obs.valid else None
            if leg is None:
                continue
            weight = st.lift if side == swing else p.double_support_gain
            if weight <= 1e-4:
                continue
            for name, value in self._deviation(leg, side, u).items():
                targets[name] = targets.get(name, 0.0) + weight * value

        if st.shift > 1e-4 and st.stance:
            lean = p.shift_rad * st.shift * self._shift_dir_cached(st.stance)
            for name in ("LHipRoll", "RHipRoll"):
                targets[name] = targets.get(name, 0.0) + lean
            for name in ("LAnkleRoll", "RAnkleRoll"):
                targets[name] = targets.get(name, 0.0) - lean

        if abs(yaw_bias) > 1e-4:
            bias = _clamp(yaw_bias, -p.max_yaw_bias, p.max_yaw_bias)
            targets["LHipYawPitch"] = targets.get("LHipYawPitch", 0.0) + bias
            targets["RHipYawPitch"] = targets.get("RHipYawPitch", 0.0) + bias
        return targets

    def _shift_dir_cached(self, stance: str) -> float:
        cached = self._probe.get(stance)
        return cached[0] if cached is not None else (-1.0 if stance == "L" else 1.0)

    def _deviation(self, leg: LegTarget, side: str, u: float) -> Dict[str, float]:
        """The human leg pose minus the symmetric crouch, per-channel capped."""
        p = self.params
        base_hip, base_knee, base_ankle = -u, 2.0 * u, -u
        return {
            f"{side}HipPitch": _clamp(leg.hip_pitch - base_hip,
                                      -p.max_hip_pitch_dev, p.max_hip_pitch_dev),
            f"{side}HipRoll": _clamp(leg.hip_roll,
                                     -p.max_hip_roll_dev, p.max_hip_roll_dev),
            f"{side}KneePitch": _clamp(leg.knee_pitch - base_knee,
                                       -p.max_knee_dev, p.max_knee_dev),
            f"{side}AnklePitch": _clamp(leg.ankle_pitch - base_ankle,
                                        -p.max_ankle_dev, p.max_ankle_dev),
            f"{side}AnkleRoll": _clamp(leg.ankle_roll,
                                       -p.max_hip_roll_dev, p.max_hip_roll_dev),
        }


def _approach(current: float, target: float, max_delta: float) -> float:
    """Move ``current`` toward ``target`` by at most ``max_delta``."""
    if max_delta <= 0.0:
        return current
    return _clamp(current + _clamp(target - current, -max_delta, max_delta), 0.0, 1.0)


def default_lower_body_params() -> LowerBodyParams:
    return LowerBodyParams()
