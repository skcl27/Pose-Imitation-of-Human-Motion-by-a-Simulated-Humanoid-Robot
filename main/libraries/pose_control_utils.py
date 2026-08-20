"""
Utilities for pose imitation control of the Webots NAO humanoid.

This module is the heart of the Webots side of the pipeline. It converts the
*generic* joint angles produced by the Python retargeting stage into commands
that are correct for the **NAO (H25)** robot, then drives the motors smoothly
and keeps the robot standing.

Why a mapping layer is needed
-----------------------------
The Python retargeting module (`src/retargeting/mapper.py`) emits angles using a
neutral convention that does *not* match NAO's joint conventions:

* NAO ``LElbowRoll`` is **negative** (-1.5446 .. -0.0349 rad) and ``RElbowRoll``
  is **positive** (0.0349 .. 1.5446 rad). The pipeline sends the opposite signs,
  so without correction ``Motor.setPosition`` clamps both elbows straight and
  they never bend.
* NAO ``ShoulderPitch`` uses arm-down = +1.57 rad; the pipeline sends arm-down
  ≈ -1.57 rad (inverted).
* NAO has **no** ``TorsoPitch`` motor, so that channel is dropped.

``NaoPoseDriver`` applies a per-joint affine correction (``scale``/``offset``),
clamps to the real NAO mechanical limits (FR-5), exponentially smooths the
targets to prevent oscillation (FR-6), and holds a stable standing posture so
the robot does not fall during upper-body imitation (NFR-4).

Who commands the legs
---------------------
Only ONE layer may command the 12 leg joints at a time, or they fight each other
and the robot falls. The driver exposes exactly one entry point per layer and the
controller picks between them each step:

* :meth:`NaoPoseDriver.lower_body_tick` -- per-leg pose imitation with the
  weight-shift/lift sequencer (``lower_body.LowerBodyController``). The default:
  this is what makes squatting and raising a single leg work.
* :meth:`NaoPoseDriver.gait_tick` -- the in-place march engine
  (``gait.GaitEngine``), used when the human is walking but no pre-balanced
  Webots walk clip is available to translate with.
* :meth:`NaoPoseDriver.balance_tick` -- CoM balance only (legs otherwise static).
* :meth:`NaoPoseDriver.release_to_motion` -- hands the WHOLE body to a Webots
  ``Motion`` clip: per-joint commanding stops and the motor velocity caps are
  lifted, because a velocity-capped motor cannot follow a motion clip's keyframes
  and the "walk" degenerates into a stumble.

This module deliberately does **not** import the Webots ``controller`` package,
so the math (limits, mapping, smoothing) stays unit-testable off-simulation.
Motor/sensor objects are passed in from the controller process.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# NAO H25 joint limits (radians)
#
# Source: Aldebaran/SoftBank NAO H25 joint documentation and the Webots
# Nao.proto RotationalMotor min/maxPosition values. These are the *hardware*
# ranges; Webots clamps setPosition() to them, so commanding outside the range
# silently saturates the joint.
# ---------------------------------------------------------------------------
NAO_JOINT_LIMITS: Dict[str, MotorConfig] = {}


@dataclass
class MotorConfig:
    """Mechanical configuration for a single NAO motor."""
    name: str
    min_angle: float          # rad
    max_angle: float          # rad
    max_velocity: float       # rad/s (hardware ceiling)
    rest_angle: float = 0.0   # rad, neutral/standing default


def _deg(d: float) -> float:
    return math.radians(d)


def get_default_motor_configs() -> Dict[str, MotorConfig]:
    """Return mechanical configs for every NAO joint we care about.

    ``rest_angle`` encodes a stable standing posture: legs straight (0 rad),
    arms hanging slightly away from the torso so they do not self-collide.
    """
    cfgs = [
        # Head
        MotorConfig("HeadYaw",        _deg(-119.5), _deg(119.5), 8.27, 0.0),
        MotorConfig("HeadPitch",      _deg(-38.5),  _deg(29.5),  7.19, 0.0),

        # Left arm
        MotorConfig("LShoulderPitch", _deg(-119.5), _deg(119.5), 8.27, _deg(85)),
        MotorConfig("LShoulderRoll",  _deg(-18.0),  _deg(76.0),  7.19, _deg(10)),
        MotorConfig("LElbowYaw",      _deg(-119.5), _deg(119.5), 8.27, _deg(-70)),
        MotorConfig("LElbowRoll",     _deg(-88.5),  _deg(-2.0),  7.19, _deg(-30)),
        MotorConfig("LWristYaw",      _deg(-104.5), _deg(104.5), 24.6, 0.0),

        # Right arm (note the mirrored roll signs)
        MotorConfig("RShoulderPitch", _deg(-119.5), _deg(119.5), 8.27, _deg(85)),
        MotorConfig("RShoulderRoll",  _deg(-76.0),  _deg(18.0),  7.19, _deg(-10)),
        MotorConfig("RElbowYaw",      _deg(-119.5), _deg(119.5), 8.27, _deg(70)),
        MotorConfig("RElbowRoll",     _deg(2.0),    _deg(88.5),  7.19, _deg(30)),
        MotorConfig("RWristYaw",      _deg(-104.5), _deg(104.5), 24.6, 0.0),

        # Left leg
        MotorConfig("LHipYawPitch",   _deg(-65.6),  _deg(42.4),  4.16, 0.0),
        MotorConfig("LHipRoll",       _deg(-21.7),  _deg(45.3),  4.16, 0.0),
        MotorConfig("LHipPitch",      _deg(-88.0),  _deg(27.7),  6.40, 0.0),
        MotorConfig("LKneePitch",     _deg(-5.3),   _deg(121.0), 6.40, 0.0),
        MotorConfig("LAnklePitch",    _deg(-68.2),  _deg(52.9),  6.40, 0.0),
        MotorConfig("LAnkleRoll",     _deg(-22.8),  _deg(44.1),  4.16, 0.0),

        # Right leg
        MotorConfig("RHipYawPitch",   _deg(-65.6),  _deg(42.4),  4.16, 0.0),
        MotorConfig("RHipRoll",       _deg(-45.3),  _deg(21.7),  4.16, 0.0),
        MotorConfig("RHipPitch",      _deg(-88.0),  _deg(27.7),  6.40, 0.0),
        MotorConfig("RKneePitch",     _deg(-5.9),   _deg(121.5), 6.40, 0.0),
        MotorConfig("RAnklePitch",    _deg(-67.9),  _deg(53.4),  6.40, 0.0),
        MotorConfig("RAnkleRoll",     _deg(-44.1),  _deg(22.8),  4.16, 0.0),
    ]
    configs = {c.name: c for c in cfgs}
    NAO_JOINT_LIMITS.clear()
    NAO_JOINT_LIMITS.update(configs)
    return configs


# Build the module-level table on import.
get_default_motor_configs()


# ---------------------------------------------------------------------------
# Pipeline -> NAO joint mapping
# ---------------------------------------------------------------------------
@dataclass
class JointMap:
    """Affine correction from a pipeline joint angle to a NAO motor target.

    ``nao_target = scale * pipeline_angle + offset`` (then clamped to limits).
    """
    nao_name: str
    scale: float = 1.0
    offset: float = 0.0
    is_leg: bool = False      # gated behind drive_legs for balance safety


# The pipeline emits these keys (see src/retargeting/mapper.py):
#   LShoulderPitch, RShoulderPitch, LElbowRoll, RElbowRoll,
#   LHipPitch, RHipPitch, TorsoPitch
#
# Corrections:
#   * ShoulderPitch: pipeline arm-down ≈ -1.57, NAO arm-down = +1.57  -> scale -1
#   * ElbowRoll: pipeline left is positive / right negative; NAO is the
#     opposite sign for each side                                     -> scale -1
#   * Hips: same axis sense, gated behind drive_legs                  -> scale +1
#   * TorsoPitch: no NAO motor                                        -> omitted
PIPELINE_TO_NAO: Dict[str, JointMap] = {
    "LShoulderPitch": JointMap("LShoulderPitch", scale=-1.0),
    "RShoulderPitch": JointMap("RShoulderPitch", scale=-1.0),
    "LElbowRoll":     JointMap("LElbowRoll",     scale=-1.0),
    "RElbowRoll":     JointMap("RElbowRoll",     scale=-1.0),
    "LHipPitch":      JointMap("LHipPitch",      scale=1.0, is_leg=True),
    "RHipPitch":      JointMap("RHipPitch",      scale=1.0, is_leg=True),
}


# ---------------------------------------------------------------------------
# Joint limiting / smoothing helpers (pure math, unit-testable)
# ---------------------------------------------------------------------------
class JointLimiter:
    """Enforces joint angle limits."""

    def __init__(self, configs: Dict[str, MotorConfig]) -> None:
        self.configs = configs

    def clamp_angle(self, joint_name: str, angle: float) -> float:
        cfg = self.configs.get(joint_name)
        if cfg is None:
            return angle
        return max(cfg.min_angle, min(cfg.max_angle, angle))

    def is_within_limits(self, joint_name: str, angle: float) -> bool:
        cfg = self.configs.get(joint_name)
        if cfg is None:
            return True
        return cfg.min_angle <= angle <= cfg.max_angle


class ExponentialSmoother:
    """Per-joint exponential moving average to damp jitter (FR-6).

    ``alpha`` in (0, 1]; higher = more responsive, lower = smoother.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self._state: Dict[str, float] = {}

    def reset(self, joint_name: str, value: float) -> None:
        self._state[joint_name] = value

    def smooth(self, joint_name: str, target: float) -> float:
        prev = self._state.get(joint_name)
        value = target if prev is None else prev + (target - prev) * self.alpha
        self._state[joint_name] = value
        return value


class MotorHealthMonitor:
    """Tracks position-tracking error and flags stuck motors."""

    def __init__(self, max_position_error: float = 0.1, window: int = 100) -> None:
        self.max_position_error = max_position_error
        self.window = window
        self.position_errors: Dict[str, List[float]] = {}

    def record(self, joint_name: str, target: float, current: float) -> float:
        error = abs(target - current)
        errs = self.position_errors.setdefault(joint_name, [])
        errs.append(error)
        if len(errs) > self.window:
            errs.pop(0)
        return error

    def average_error(self, joint_name: str) -> float:
        errs = self.position_errors.get(joint_name, [])
        return sum(errs) / len(errs) if errs else 0.0

    def is_stuck(self, joint_name: str) -> bool:
        errs = self.position_errors.get(joint_name, [])[-10:]
        if not errs:
            return False
        return (sum(errs) / len(errs)) > self.max_position_error * 2


def map_pipeline_angles(
    incoming: Dict[str, float],
    *,
    drive_legs: bool = False,
    limiter: Optional[JointLimiter] = None,
) -> Dict[str, float]:
    """Convert pipeline joint angles to clamped NAO motor targets.

    Pure function (no Webots dependency) so the mapping is unit-testable.
    Unknown joints and (when ``drive_legs`` is False) leg joints are dropped.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    out: Dict[str, float] = {}
    for src, value in incoming.items():
        spec = PIPELINE_TO_NAO.get(src)
        if spec is None:
            continue
        if spec.is_leg and not drive_legs:
            continue
        target = spec.scale * float(value) + spec.offset
        out[spec.nao_name] = limiter.clamp_angle(spec.nao_name, target)
    return out


# ---------------------------------------------------------------------------
# Standing posture
# ---------------------------------------------------------------------------
# Joints that imitation actively drives. Everything else is held at its
# rest_angle for a stable, natural-looking standing pose.
DRIVEN_ARM_JOINTS = ("LShoulderPitch", "RShoulderPitch", "LElbowRoll", "RElbowRoll")
# Lower body: the symmetric, statically-balanced crouch axis (hip/knee/ankle
# pitch). This is the posture every leg layer decays back to; see
# ``nao_retarget.crouch_posture``.
DRIVEN_LEG_JOINTS = (
    "LHipPitch", "RHipPitch",
    "LKneePitch", "RKneePitch",
    "LAnklePitch", "RAnklePitch",
)
# Every joint the lower-body layers may command. Used for the gentler leg
# velocity cap: these carry the robot's weight, so a jolt here is a fall.
ALL_LEG_JOINTS = DRIVEN_LEG_JOINTS + (
    "LHipYawPitch", "RHipYawPitch",
    "LHipRoll", "RHipRoll",
    "LAnkleRoll", "RAnkleRoll",
)


def standing_posture() -> Dict[str, float]:
    """Return the neutral standing target for every NAO joint (radians).

    Legs are kept straight (0 rad) and stiff so the robot stays balanced
    during upper-body imitation (NFR-4).
    """
    return {name: cfg.rest_angle for name, cfg in NAO_JOINT_LIMITS.items()}


def _null_logger(_msg: str) -> None:  # pragma: no cover - default sink
    pass


# ---------------------------------------------------------------------------
# NaoPoseDriver — owns the Webots motors and applies pose frames
# ---------------------------------------------------------------------------
@dataclass
class DriverStats:
    frames_applied: int = 0
    joints_last_applied: int = 0
    stale: bool = False


class NaoPoseDriver:
    """Drives the NAO motors from pipeline pose frames.

    The driver is constructed with a live Webots ``Robot`` instance. It does
    the device lookups, applies the standing posture, then on each ``update``
    maps -> clamps -> smooths -> commands the motors. It also reads the
    position sensors (``<name>S``) for health monitoring.

    Parameters
    ----------
    robot:
        Webots ``Robot`` instance.
    drive_legs:
        If True, build the per-leg lower-body layer
        (``lower_body.LowerBodyController``): squat, leg abduction, and
        single-leg lift with a model-verified weight transfer. If False the legs
        are held in the neutral standing posture and only the upper body imitates.
    smoothing_alpha:
        EMA factor for joint targets in (0, 1].
    velocity_scale:
        Fraction of each joint's hardware max velocity used as the motion cap.
    leg_velocity_factor:
        Extra multiplier (0..1) applied on top of ``velocity_scale`` for the leg
        joints only, so a weight-bearing crouch/lean eases in slowly and does not
        jolt the centre of mass off the feet (NFR-4). While stepping or marching
        the (higher) ``gait_leg_velocity_factor`` is used instead, because there
        the leg has to actually keep up with the motion.
    stale_after_s:
        If no command arrives within this many seconds, the driver is marked
        stale (the robot simply holds its last commanded pose).
    """

    def __init__(
        self,
        robot,
        *,
        drive_legs: bool = False,
        drive_head: bool = True,
        swap_sides: bool = False,
        smoothing_alpha: float = 0.4,
        velocity_scale: float = 0.5,
        leg_velocity_factor: float = 0.5,
        stale_after_s: float = 0.5,
        enable_balance: bool = False,
        enable_walk: bool = False,
        walk_tier: str = "march",
        gait_smoothing_alpha: float = 0.7,
        gait_leg_velocity_factor: float = 0.85,
        gait_params: Optional[object] = None,
        lower_body_params: Optional[object] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.drive_legs = drive_legs
        self.drive_head = drive_head
        self.swap_sides = swap_sides
        self.velocity_scale = max(0.05, min(1.0, velocity_scale))
        self.leg_velocity_factor = max(0.05, min(1.0, leg_velocity_factor))
        self.gait_leg_velocity_factor = max(0.05, min(1.0, gait_leg_velocity_factor))
        self.stale_after_s = stale_after_s
        self.enable_walk = enable_walk
        self.walk_tier = walk_tier
        self.log = logger or _null_logger

        self.configs = get_default_motor_configs()
        self.limiter = JointLimiter(self.configs)
        self.smoother = ExponentialSmoother(smoothing_alpha)
        # Legs under the gait engine need their own, snappier smoother so the
        # walking waveform is not double-attenuated by the arm/head EMA (FR-6).
        self.gait_smoother = ExponentialSmoother(gait_smoothing_alpha)
        self.health = MotorHealthMonitor()

        self.motors: Dict[str, object] = {}
        self.sensors: Dict[str, object] = {}
        self.commanded: Dict[str, float] = {}
        self.measured: Dict[str, float] = {}
        # True while a Webots Motion clip owns the whole body (see
        # release_to_motion / reclaim_from_motion).
        self.suspended = False
        # The pose imitation wants this leg posture; the balance loop adds small
        # corrections on top of it each control step.
        self.base_targets: Dict[str, float] = {}
        self.stats = DriverStats()
        self._last_command_time: Optional[float] = None

        # Model-based CoM balance feedback (Option 2: FK + known link masses).
        # Imported lazily and guarded so the driver still runs if numpy/balance
        # is unavailable.
        self.balance = None
        if enable_balance:
            try:
                from balance import BalanceController, NaoCoMModel
                self.balance = BalanceController(NaoCoMModel())
                self.log("Balance feedback ON (model-based CoM, Fibonacci search)")
            except Exception as exc:  # noqa: BLE001
                self.log(f"Balance feedback OFF ({exc})")

        # Walk engine (gait command -> balance-stable leg motion). When enabled,
        # the gait path is the SOLE commander of the legs and the retargeter's
        # crouch is skipped (gait owns the lower body). Imported lazily because
        # ``gait`` imports this module.
        self.gait_engine = None
        self._gait_meta: Dict[str, object] = {"single_support": False, "amp_gain": 0.0}
        if enable_walk:
            try:
                from gait import GaitEngine
                com_model = self.balance.model if self.balance is not None else None
                self.gait_engine = GaitEngine(
                    params=gait_params, com_model=com_model, limiter=self.limiter
                )
                self.log(f"Walk engine ON (tier={walk_tier})")
            except Exception as exc:  # noqa: BLE001
                self.enable_walk = False
                self.log(f"Walk engine OFF ({exc})")

        # Per-leg pose imitation + weight-shift/lift sequencer. This is the layer
        # that makes a squat and a single-leg lift actually reach the robot; it
        # needs the CoM model to decide when unloading a foot is safe, and
        # degrades to a hard-capped lift without it (never to "no motion at all",
        # which is what made the legs look dead before).
        self.leg_retargeter = None
        self.lower_body = None
        if drive_legs:
            try:
                from lower_body import LowerBodyController
                from nao_retarget import LowerBodyRetargeter
                com_model = self.balance.model if self.balance is not None else None
                self.leg_retargeter = LowerBodyRetargeter()
                self.lower_body = LowerBodyController(
                    params=lower_body_params, com_model=com_model, limiter=self.limiter
                )
                self.log(
                    "Lower-body pose imitation ON"
                    f" (CoM-gated stepping: {com_model is not None})"
                )
            except Exception as exc:  # noqa: BLE001
                self.log(f"Lower-body pose imitation OFF ({exc})")
        self._lb_meta: Dict[str, object] = {"mode": "off", "balance_ok": True}

        self._setup_devices()
        self.apply_standing_posture()

    # -- device setup -------------------------------------------------------
    def _setup_devices(self) -> None:
        found, missing = 0, []
        for name in self.configs:
            motor = self.robot.getDevice(name)
            if motor is None:
                missing.append(name)
                continue
            self.motors[name] = motor
            found += 1
            sensor = self.robot.getDevice(name + "S")
            if sensor is not None:
                try:
                    sensor.enable(self.timestep)
                except Exception:  # noqa: BLE001 - some devices may not be sensors
                    pass
                else:
                    self.sensors[name] = sensor
        self.log(f"Motors found: {found}/{len(self.configs)}; sensors: {len(self.sensors)}")
        if missing:
            self.log(f"Motors not present on this model: {', '.join(missing)}")

    def _set_motor(self, name: str, angle: float, velocity: float) -> None:
        motor = self.motors.get(name)
        if motor is None or self.suspended:
            # Suspended = a Webots Motion clip owns the body; commanding motors
            # now would fight the clip's keyframes and break its balance.
            return
        angle = self.limiter.clamp_angle(name, angle)
        try:
            motor.setVelocity(max(0.01, velocity))
            motor.setPosition(angle)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to command {name}: {exc}")
            return
        self.commanded[name] = angle

    def _velocity_for(self, name: str) -> float:
        cfg = self.configs.get(name)
        ceiling = cfg.max_velocity if cfg else 4.0
        scale = self.velocity_scale
        # Legs carry the robot's weight: move them gently so a crouch/sway eases
        # in rather than jolting the centre of mass off the feet (NFR-4).
        if name in ALL_LEG_JOINTS:
            scale *= self.leg_velocity_factor
        return ceiling * scale

    # -- posture ------------------------------------------------------------
    def apply_standing_posture(self) -> None:
        """Move every joint to its neutral standing target and seed smoothing."""
        posture = standing_posture()
        for name, angle in posture.items():
            self.smoother.reset(name, angle)
            self._set_motor(name, angle, self._velocity_for(name) * 0.6)
        self.log("Applied standing posture")

    # -- per-frame update ---------------------------------------------------
    def _apply_targets(self, targets: Dict[str, float], now_s: Optional[float]) -> int:
        """Smooth, command and bookkeep a set of NAO joint targets."""
        if self.suspended:
            # A motion clip owns the body; still record the frame time so
            # staleness detection keeps working across the clip.
            if now_s is not None:
                self._last_command_time = now_s
            return 0
        applied = 0
        for name, target in targets.items():
            if name not in self.motors:
                continue
            self.base_targets[name] = target
            smoothed = self.smoother.smooth(name, target)
            self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1

        if now_s is not None:
            self._last_command_time = now_s
        self.stats.frames_applied += 1
        self.stats.joints_last_applied = applied
        self.stats.stale = False
        return applied

    def balance_tick(self, torso_rp: tuple = (0.0, 0.0)) -> int:
        """Run one CoM balance cycle: re-command the legs as base + correction.

        Called every control step (not just on new pose frames) so balance is
        maintained continuously. ``torso_rp`` is the InertialUnit (roll, pitch)
        in rad. Returns the number of joints nudged. No-op if balance is off.
        """
        if self.balance is None or self.suspended:
            return 0
        # Best estimate of the current pose: measured where available, else the
        # last commanded angle.
        state = dict(self.commanded)
        state.update(self.measured)
        try:
            corr = self.balance.compute_correction(state, torso_rp)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Balance step failed, disabling ({exc})")
            self.balance = None
            return 0

        applied = 0
        for name, delta in corr.items():
            if name not in self.motors:
                continue
            base = self.base_targets.get(name, self.configs[name].rest_angle)
            smoothed = self.smoother.smooth(name, base + delta)
            self._set_motor(name, smoothed, self._velocity_for(name))
            applied += 1
        return applied

    # -- lower-body pose imitation -----------------------------------------
    def lower_body_tick(
        self,
        now_s: float,
        torso_rp: tuple = (0.0, 0.0),
        fsr: Optional[Dict[str, float]] = None,
        yaw_bias: float = 0.0,
    ) -> int:
        """Advance the per-leg pose imitation one control step.

        This is the DEFAULT leg commander: it turns the latest camera
        observation into a safe leg posture via
        ``lower_body.LowerBodyController`` (symmetric crouch + authority-weighted
        per-leg deviation + a model-verified weight shift when the human lifts a
        foot), then folds in the symmetric CoM balance correction *only* while the
        sequencer says both feet are still evenly loaded -- once we are
        deliberately leaning onto one foot, "centre the CoM between the feet" is
        the wrong objective and would cancel the transfer.

        Called every simulation step rather than per camera frame, so the legs
        keep being controlled at sim rate even when pose packets stall. Returns
        the number of joints commanded (0 when the layer is off/suspended).
        """
        if self.lower_body is None or self.suspended:
            return 0
        state = dict(self.commanded)
        state.update(self.measured)
        try:
            targets, meta = self.lower_body.step(
                now_s, torso_rp=torso_rp, fsr=fsr, measured=state, yaw_bias=yaw_bias
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Lower-body step failed, disabling ({exc})")
            self.lower_body = None
            return 0
        self._lb_meta = meta

        if self.balance is not None and meta.get("balance_ok", False):
            try:
                corr = self.balance.compute_correction(state, torso_rp)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Balance step failed, disabling ({exc})")
                self.balance = None
                corr = {}
            for name, delta in corr.items():
                targets[name] = self.limiter.clamp_angle(
                    name, targets.get(name, 0.0) + delta
                )
            # The balance loop's roll terms are not sole-tilt-neutral, so the
            # lower body's tilt guarantee has to be re-applied once they are in.
            self.lower_body.apply_sole_tilt_limit(targets)

        applied = 0
        for name, value in targets.items():
            if name not in self.motors:
                continue
            self.base_targets[name] = value
            # The gait smoother's snappier alpha is right here too: the legs must
            # follow a step, not lag it into a shuffle.
            smoothed = self.gait_smoother.smooth(name, value)
            self._set_motor(name, smoothed, self._gait_velocity_for(name))
            applied += 1
        return applied

    def set_lower_body_observation(self, obs: Optional[object]) -> None:
        """Latch a fresh lower-body observation (no-op when the layer is off)."""
        if self.lower_body is not None and obs is not None:
            self.lower_body.set_observation(obs)

    def lower_body_stand_down(self) -> None:
        """Tell the lower body to forget the last observation (tracking lost)."""
        if self.lower_body is not None:
            self.lower_body.stand_down()

    @property
    def lower_body_meta(self) -> Dict[str, object]:
        """Latest lower-body telemetry (mode, shift, lift, stance margin, ...)."""
        return dict(self._lb_meta)

    # -- whole-body Webots Motion clips ------------------------------------
    def release_to_motion(self) -> None:
        """Hand the whole body to a Webots ``Motion`` clip.

        Two things have to happen, and missing either one is why "play a walk
        clip" usually looks broken:

        1. Stop commanding motors, so our targets do not fight the clip's
           keyframes.
        2. **Lift the velocity caps.** ``Motion`` playback works by calling
           ``setPosition`` on every joint each step; a motor still limited to
           ~25% of its maximum velocity simply cannot reach those keyframes, so
           the pre-balanced gait arrives late at every foot placement and the
           robot topples. Motors keep whatever velocity was last set, so the caps
           must be raised explicitly here.
        """
        if self.suspended:
            return
        for name, motor in self.motors.items():
            cfg = self.configs.get(name)
            try:
                motor.setVelocity(cfg.max_velocity if cfg else 6.0)
            except Exception:  # noqa: BLE001
                pass
        self.suspended = True

    def reclaim_from_motion(self) -> None:
        """Take the body back from a motion clip without a jolt.

        The smoothers still hold pre-clip values, and the robot is now somewhere
        else entirely, so they are reseeded from the position sensors before
        per-joint commanding resumes.
        """
        if not self.suspended:
            return
        self.suspended = False
        self.reseed_from_measured()
        for name in self.motors:
            self._set_motor(name, self.measured.get(name, self.commanded.get(name, 0.0)),
                            self._velocity_for(name))

    # -- gait / walking -----------------------------------------------------
    def set_gait_command(self, gait: Optional[Dict[str, object]]) -> None:
        """Hand the latest gait command (from the Python cue extractor) to the
        walk engine. No-op when walking is disabled."""
        if self.gait_engine is not None:
            self.gait_engine.set_command(gait)

    def _gait_velocity_for(self, name: str) -> float:
        """Leg velocity while walking: a higher factor than the gentle crouch so
        the gait waveform actually moves (it would otherwise collapse to a
        shuffle under the crouch's slow leg velocity)."""
        cfg = self.configs.get(name)
        ceiling = cfg.max_velocity if cfg else 4.0
        return ceiling * self.velocity_scale * self.gait_leg_velocity_factor

    def gait_tick(self, now_s: float, torso_rp: tuple = (0.0, 0.0),
                  fsr: Optional[Dict[str, float]] = None) -> int:
        """Advance the walk engine one step and command the legs.

        When walking is enabled this REPLACES ``balance_tick`` for the lower
        body and is the sole commander of the 12 leg joints: it asks the engine
        for the gait leg posture, folds in the symmetric CoM balance correction
        while in double support (Tier A, where that correction is valid), and
        commands the legs with the snappier gait smoother and raised leg
        velocity. In single support (Tier B) the gait owns the roll axis and the
        symmetric correction is skipped (the engine's IMU-tilt abort is the
        safety net). Returns the number of joints commanded; 0 if walk is off.
        """
        if self.gait_engine is None or self.suspended:
            return 0
        state = dict(self.commanded)
        state.update(self.measured)
        try:
            targets, meta = self.gait_engine.step(
                now_s, tier=self.walk_tier, torso_rp=torso_rp, fsr=fsr, measured=state
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Gait step failed, disabling walk ({exc})")
            self.gait_engine = None
            return 0
        self._gait_meta = meta

        if self.balance is not None and not meta.get("single_support", False):
            try:
                corr = self.balance.compute_correction(state, torso_rp)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Balance step failed, disabling ({exc})")
                self.balance = None
                corr = {}
            for name, delta in corr.items():
                targets[name] = self.limiter.clamp_angle(
                    name, targets.get(name, 0.0) + delta
                )

        applied = 0
        for name, value in targets.items():
            if name not in self.motors:
                continue
            self.base_targets[name] = value
            smoothed = self.gait_smoother.smooth(name, value)
            self._set_motor(name, smoothed, self._gait_velocity_for(name))
            applied += 1
        return applied

    @property
    def gait_meta(self) -> Dict[str, object]:
        """Latest walk-engine telemetry (amp_gain, phase, cadence, single_support)."""
        return dict(self._gait_meta)

    def reseed_from_measured(self) -> None:
        """Reset the smoothers to the measured joint angles.

        Call this when handing control back from an external whole-body motion
        (e.g. a Webots walk clip) to per-joint imitation/gait, so targets ease
        from where the robot ACTUALLY is rather than from a stale smoother value
        — avoids a jolt at the motion->imitation transition.
        """
        for name in self.motors:
            val = self.measured.get(name)
            if val is None:
                continue
            self.smoother.reset(name, val)
            self.gait_smoother.reset(name, val)
            self.commanded[name] = val

    def update(self, incoming: Dict[str, float], now_s: Optional[float] = None) -> int:
        """Apply one frame of *pre-computed* pipeline joint angles (fallback).

        Returns the number of joints commanded.
        """
        targets = map_pipeline_angles(
            incoming, drive_legs=self.drive_legs, limiter=self.limiter
        )
        return self._apply_targets(targets, now_s)

    def update_from_keypoints(
        self, keypoints: Dict[str, object], now_s: Optional[float] = None
    ) -> int:
        """Apply one camera frame: arms/head now, legs via the lower-body layer.

        The arms and head are commanded here because they are safe to drive
        straight from the pose. The legs are NOT: their observation is only
        *latched* for whichever lower-body layer the controller is running this
        step, which then decides how much of it the robot can execute without
        losing balance. That split is what keeps a single commander on the legs.

        Returns the number of joints commanded. Imported lazily to avoid a
        circular import (``nao_retarget`` depends on this module).
        """
        from nao_retarget import retarget_upper_body

        targets = retarget_upper_body(
            keypoints,
            drive_head=self.drive_head,
            swap_sides=self.swap_sides,
            limiter=self.limiter,
        )
        if self.leg_retargeter is not None:
            try:
                obs = self.leg_retargeter.observe(keypoints)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Leg retargeting failed, disabling ({exc})")
                self.leg_retargeter = None
            else:
                self.set_lower_body_observation(obs)
        return self._apply_targets(targets, now_s)

    def read_feedback(self) -> None:
        """Read position sensors and record tracking error for health checks."""
        for name, sensor in self.sensors.items():
            try:
                value = float(sensor.getValue())
            except Exception:  # noqa: BLE001
                continue
            if math.isnan(value):  # sensors read NaN until the first sim step
                continue
            self.measured[name] = value
            if name in self.commanded:
                self.health.record(name, self.commanded[name], value)

    def check_stale(self, now_s: float) -> bool:
        """Mark the driver stale if no command arrived recently."""
        if self._last_command_time is None:
            return False
        stale = (now_s - self._last_command_time) > self.stale_after_s
        self.stats.stale = stale
        return stale

    # Velocity used by :meth:`stop`. Deliberately NOT zero: a zero-velocity motor
    # cannot move at all, so if anything ever calls stop() while the controller
    # keeps running, the robot is bricked with no diagnostic. A small positive
    # value holds position just as well and stays recoverable.
    HOLD_VELOCITY = 0.2

    def stop(self) -> None:
        """Hold the current position (graceful shutdown)."""
        for name, motor in self.motors.items():
            try:
                motor.setVelocity(self.HOLD_VELOCITY)
                if name in self.measured:
                    motor.setPosition(self.measured[name])
            except Exception:  # noqa: BLE001
                pass

    def stuck_motors(self) -> List[str]:
        return [n for n in self.motors if self.health.is_stuck(n)]

    @property
    def logged_joints(self) -> List[str]:
        """Joints worth logging for fidelity metrics (driven joints only)."""
        joints = [
            "LShoulderPitch", "RShoulderPitch",
            "LShoulderRoll", "RShoulderRoll",
            "LElbowRoll", "RElbowRoll",
        ]
        if self.drive_head:
            joints += ["HeadYaw", "HeadPitch"]
        if self.drive_legs or self.enable_walk:
            for side in ("L", "R"):
                joints += [
                    f"{side}HipYawPitch", f"{side}HipRoll", f"{side}HipPitch",
                    f"{side}KneePitch", f"{side}AnklePitch", f"{side}AnkleRoll",
                ]
        return [j for j in joints if j in self.motors]


# ---------------------------------------------------------------------------
# Trajectory logging (FR-7 / US-3)
# ---------------------------------------------------------------------------
class JointTrajectoryLogger:
    """Append-only CSV log of commanded vs. achieved joint angles.

    The Python pipeline logs the *commanded* angles upstream; only the Webots
    side can observe the robot's *achieved* angles (from the position sensors).
    Logging both here lets the evaluation step compute per-joint MAE between
    target and achieved motion (PRD US-3, NFR-3) and end-to-end timing.

    The logger is defensive by design: any I/O error disables logging rather
    than disturbing the real-time control loop.
    """

    def __init__(
        self,
        directory: str,
        joints: Iterable[str],
        *,
        filename: Optional[str] = None,
        flush_every: int = 50,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        import csv
        import os

        self.joints = list(joints)
        self.flush_every = max(1, flush_every)
        self.log = logger or _null_logger
        self._rows_since_flush = 0
        self._file = None
        self._writer = None

        try:
            os.makedirs(directory, exist_ok=True)
            if filename is None:
                filename = f"webots_joint_trajectory_{int(time_now())}.csv"
            path = os.path.join(directory, filename)
            self._file = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            header = ["wall_time_s", "sim_time_s", "frame_index"]
            for j in self.joints:
                header += [f"{j}_cmd_rad", f"{j}_meas_rad"]
            self._writer.writerow(header)
            self.path = path
            self.log(f"Trajectory log: {path}")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Trajectory logging disabled ({exc})")
            self._file = None
            self._writer = None
            self.path = None

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def record(
        self,
        sim_time_s: float,
        frame_index: int,
        commanded: Dict[str, float],
        measured: Dict[str, float],
    ) -> None:
        if self._writer is None:
            return
        try:
            row: List[object] = [round(time_now(), 6), round(sim_time_s, 6), frame_index]
            for j in self.joints:
                cmd = commanded.get(j)
                meas = measured.get(j)
                row.append("" if cmd is None else round(cmd, 6))
                row.append("" if meas is None else round(meas, 6))
            self._writer.writerow(row)
            self._rows_since_flush += 1
            if self._rows_since_flush >= self.flush_every:
                self._file.flush()
                self._rows_since_flush = 0
        except Exception as exc:  # noqa: BLE001
            self.log(f"Trajectory logging stopped ({exc})")
            self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:  # noqa: BLE001
                pass
        self._file = None
        self._writer = None


def time_now() -> float:
    """Wall-clock seconds. Wrapped so it is trivial to stub in tests."""
    import time as _time

    return _time.time()
