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

Symmetric / antisymmetric split
-------------------------------
The commanded posture is ``crouch_posture(u)`` -- the statically-balanced squat
whose ``Hip + Knee + Ankle == 0`` keeps the torso vertical and the soles flat --
plus the human's *deviation from* that posture, per leg, authority-weighted.

The weight is not one number, because two very different things live in those
deviations. Splitting each channel into its mirror-symmetric and antisymmetric
parts separates them exactly:

* **Mirror-symmetric** -- both legs abducting outward by the same amount (a wider
  stance), or both flexing equally (a squat). By symmetry these move the centre of
  mass *not at all*, and a wider stance makes the support polygon **bigger**. They
  are therefore safer than standing, and get full authority. Gating them was a
  plain mistake: it turned a 40 deg human stance into a 12 deg robot one.
* **Antisymmetric** -- both legs rolled the same way (a lean), or one leg forward
  and one back (a split stance). These do move the centre of mass over the feet,
  so they stay gated.

While a foot is genuinely off the ground the split is dropped: the swing leg is
unloaded and therefore free to take its own pose at whatever authority the safety
gate allows, and the stance leg is left near the balanced crouch because it is
carrying the whole robot.

Standing still yields a deviation of exactly zero either way, so the controller
degrades to the proven baseline rather than to noise.

What actually limits a wide stance
----------------------------------
Not the hip. NAO's ``HipRoll`` reaches 45.3 deg, but ``AnkleRoll`` only reaches
22.8 deg -- and the ankle is what levels the sole against the hip's abduction.
Past 22.8 deg the sole can no longer be kept flat.

Refusing to go past that point turned out to be too strict: recorded runs show
subjects spreading to ~25 deg routinely and 34 deg at the extreme, so the robot
saturated just below the human and it read as "it spreads, but not as much as
me". So a small, explicit ``sole_tilt_budget`` is spent instead: the hip may
abduct that much further than the ankle can level, leaving each sole a few
degrees off flat and the robot standing on the inner part of each foot, which is
a good trade for the extra width. The budget is then enforced as a *post-
condition* on the commanded angles (:meth:`_limit_sole_tilt`), so it holds no
matter how the symmetric, antisymmetric and balance terms happen to add up -- and
when something has to give, it is stance width, never sole contact.

Pure Python + the (optional) NumPy CoM model, no Webots import, so all of the
sequencing and gating logic is unit-testable off-simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from nao_retarget import LegTarget, LowerBodyObservation, crouch_posture
from pose_control_utils import JointLimiter, get_default_motor_configs

_CONFIGS = get_default_motor_configs()
# How far the ankle can roll to level a sole against the hip's abduction. Smaller
# than the hip's own range, which is what makes it -- not the hip -- the binding
# constraint on stance width.
ANKLE_ROLL_LIMIT = min(abs(_CONFIGS["LAnkleRoll"].min_angle),
                       _CONFIGS["RAnkleRoll"].max_angle)

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
    # Deepest squat commanded. See nao_retarget.MAX_CROUCH: the crouch posture is
    # statically balanced at any depth, so this is a range limit, not a safety one.
    max_crouch_u: float = 0.70

    # -- how much human detail we allow through, per channel ---------------
    max_hip_pitch_dev: float = 1.25
    max_hip_roll_dev: float = 0.7906  # rad ~ 45.3 deg, NAO HipRoll hardware limit
    max_knee_dev: float = 1.45
    max_ankle_dev: float = 0.60
    # Mirror-symmetric poses (wider stance, deeper squat) are CoM-neutral and
    # widen the support polygon, so they pass at full authority.
    symmetric_gain: float = 1.0

    # -- CoM compensation (feed-forward) -----------------------------------
    # Balance is not allowed to gate the imitation. The human's leg pose is
    # commanded at full authority and the robot then SHIFTS ITS CENTRE OF MASS to
    # make that pose holdable -- which is what a person does when they lean: they
    # move their hips, they do not refuse to lean.
    #
    # The shift is the sole-flat two-parameter family (hip +c / ankle -c on both
    # legs, per axis), so it translates the pelvis without tilting either sole and
    # without touching the joints that carry the imitation. Searched on a spiral
    # warm-started from the previous solution, so each tick is cheap (8 forward
    # kinematics passes, ~4.7 ms of a 20 ms step) and the answer converges over a
    # few ticks at simulation rate.
    com_shift_max_pitch: float = 0.30   # rad; fore/aft pelvis travel available
    com_shift_max_roll: float = 0.30    # rad; lateral pelvis travel available
    # Extra stance width the compensation may add. This is the best tool it has for
    # a LEAN, and the reason is worth stating: a pelvis shift and a lean are the
    # same degree of freedom, so shifting the pelvis to hold a lean directly
    # subtracts from the lean being imitated. Widening the stance instead ENLARGES
    # the support polygon and is mirror-symmetric, so it costs the imitation
    # nothing at all. Measured, holding a 0.60 rad lean:
    #
    #     nothing              -0.004 m
    #     pelvis shift -0.20   +0.025 m   but the lean delivered drops by 0.20
    #     stance width +0.20   +0.031 m   and the lean is untouched
    #
    # It is searched rather than computed because past about 0.20 rad it stops
    # helping: the ankle runs out of range to keep the widening sole flat.
    com_shift_max_width: float = 0.35   # rad
    # Step limits per tick, and the probe used to measure the gradient. The search
    # is a measured-gradient step, not a sampling search: the support margin is a
    # piecewise-linear function of the pelvis position, so three extra forward
    # kinematics passes give the exact local sensitivity of the margin to each
    # parameter, and one Newton-ish step closes most of the deficit.
    #
    # A golden-angle spiral was tried first and was the wrong tool: the objective
    # is non-smooth (the contact filter switches which sole corners count), so a
    # warm-started sampling search hill-climbed onto the wrong side of a kink and
    # then drifted, ending with a WORSE margin than it started with.
    com_shift_probe: float = 0.02       # rad; finite-difference step
    com_shift_max_step: float = 0.06    # rad; largest change per tick, per axis
    com_shift_slew: float = 0.5         # blend toward the computed step
    com_margin_target: float = 0.020    # m
    # How long the compensation is allowed to keep trying before the pose itself is
    # scaled back. Without this the fallback fired on the very first tick -- the
    # compensation needs a few ticks to travel -- so the motion was gated after all,
    # which is the behaviour this whole design exists to remove.
    com_grace_s: float = 0.5
    # How far out of flat each sole is allowed to end up. A sole is flat when
    # AnkleRoll == -HipRoll, and the ankle's range is smaller than the hip's, so a
    # wide stance necessarily tilts the soles a little. Spending a small, bounded
    # amount of tilt buys real stance width: at 0.15 rad the outer edge of a
    # 76 mm-wide foot lifts about 11 mm, so the robot stands on the inner part of
    # each sole while the two soles are far further apart than before.
    sole_tilt_budget: float = 0.15
    # Largest both-legs-outward abduction we ask for: what the ankle can level,
    # plus that tilt budget. Recorded runs show subjects reaching ~25 deg (p99)
    # and 34 deg (peak), so the ankle limit alone (22.8 deg) saturated just below
    # what people actually do -- which read as "it spreads, but not like me".
    max_abduction: float = ANKLE_ROLL_LIMIT + 0.15
    # Antisymmetric poses -- a lean, a split stance -- move the centre of mass, and
    # they used to be attenuated to 35% for exactly that reason. They are no longer:
    # the CoM compensation above answers them by moving the pelvis instead, so the
    # imitation runs at full authority and only gets scaled back as a LAST resort,
    # when the compensation has run out of travel (_limit_asymmetry). Gating the
    # motion was the wrong tool -- it made the robot follow a lean at a third of
    # size while still being just as unbalanced, because a third of a lean still
    # moves the CoM a third of the way out.
    asymmetric_gain: float = 1.0

    # -- weight transfer ---------------------------------------------------
    shift_rad: float = 0.16         # lean amplitude that loads the stance foot
    shift_rate: float = 1.1         # 1/s ramp of the shift blend
    shift_ready: float = 0.75       # shift blend required before lifting starts
    lift_rate: float = 1.8          # 1/s ramp of the lift blend
    lift_start: float = 0.15        # human lift fraction that requests a step
    lift_stop: float = 0.07         # hysteresis: below this the foot comes down

    # -- safety gates ------------------------------------------------------
    # Double-support margin the commanded LEAN must leave (see _limit_lean). The
    # antisymmetric deviation was previously gated only by a fixed gain, so the
    # layer could ask for up to 0.437 rad of lean while the support polygon runs
    # out at about 0.58 -- a predicted margin of 3 mm, with nothing checking it
    # against the robot's own model. That is precisely the decision this module's
    # docstring says belongs here rather than in the retargeter.
    lean_margin_min: float = 0.020  # m
    margin_min: float = 0.002       # m; stance margin needed to begin lifting
    # Margin at which the full requested lift is allowed. A completed weight
    # transfer yields ~0.015 m, so anything larger silently caps the lift below
    # the human's -- which read as "the leg only moves a little".
    margin_full: float = 0.012
    fsr_load_frac: float = 0.55     # stance share at which the FSRs fully confirm
    # Authority retained when the foot sensors do NOT confirm the transfer. The
    # FSRs are a *confirmation*, never a veto: a sensor that reads a constant
    # 50/50 (uncalibrated, or a proto whose soles barely redistribute) would
    # otherwise forbid every step forever, which is exactly how a leg lift ends
    # up doing nothing at all. The CoM model has already agreed by this point and
    # the tilt abort is the real safety net.
    fsr_min_gain: float = 0.40
    fsr_total_min: float = 1.0      # N; below this the FSR reading is ignored
    tilt_abort_rad: float = 0.28    # |IMU roll/pitch| beyond this -> stand down
    conf_min: float = 0.50          # min lower-body confidence to ENGAGE
    # ...and the lower bar to STAY engaged. Hysteresis, for the same reason
    # lift_start/lift_stop have it, and it is not optional here: the retargeter
    # halves its confidence when only ONE leg is in view (a single-leg read cannot
    # tell a lift from the other foot leaving the frame), which lands a clear
    # single-leg detection at almost exactly 0.49 -- just under a 0.50 gate. A
    # recorded run flapped double -> load -> double -> load -> single, aborting each
    # weight transfer part-way through, because the gate opened and shut on that
    # boundary. Starting a transfer needs two legs; finishing one does not.
    conf_keep: float = 0.40

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
    last_now: float | None = None


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
        params: LowerBodyParams | None = None,
        *,
        com_model: object | None = None,
        limiter: JointLimiter | None = None,
    ) -> None:
        self.params = params or LowerBodyParams()
        self.limiter = limiter or JointLimiter(get_default_motor_configs())
        self.com_model = com_model
        self.state = LowerBodyState()
        self._probe: dict[str, tuple[float, float]] = {}  # stance -> (dir, t)
        # Warm start for the feed-forward CoM shift (see com_shift_* params).
        self._shift_ff: dict[str, float] = {"pitch": 0.0, "roll": 0.0, "width": 0.0}
        self._shift_margin: float = 0.0
        # When the compensation first started failing to reach its target. Reset
        # every time it succeeds, so the grace period is about a SUSTAINED failure.
        self._shift_short_since: float | None = None
        self._last_obs = LowerBodyObservation()
        self._fsr_share: float | None = None

    # ------------------------------------------------------------------ API
    def reset(self) -> None:
        """Drop all blend state (call after an external whole-body motion).

        Includes the CoM shift: a clip leaves the robot somewhere else entirely, so
        the pelvis offset that suited the old pose is meaningless against the new.

        A motion clip leaves the robot in a completely different configuration,
        so a half-finished weight transfer from before the clip must not be
        resumed on top of it.
        """
        self.state = LowerBodyState()
        self._probe.clear()
        self._shift_ff = {"pitch": 0.0, "roll": 0.0, "width": 0.0}
        self._shift_margin = 0.0
        self._shift_short_since = None

    def set_observation(self, obs: LowerBodyObservation | None) -> None:
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
        obs: LowerBodyObservation | None = None,
        *,
        torso_rp: tuple[float, float] = (0.0, 0.0),
        fsr: dict[str, float] | None = None,
        measured: dict[str, float] | None = None,
        yaw_bias: float = 0.0,
    ) -> tuple[dict[str, float], dict[str, object]]:
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
        # Hysteresis on the confidence gate: a sequence already under way holds on
        # at the lower bar, so a marginal frame cannot abort a weight transfer
        # half-completed. Tilt has no such grace -- that gate is a safety abort.
        engaged = st.lift > 1e-3 or st.shift > 1e-3
        conf_bar = p.conf_keep if engaged else p.conf_min
        usable = bool(obs.valid and obs.confidence >= conf_bar and tilt_ok)

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

        targets = self._compose(
            obs, swing, yaw_bias if mode == MODE_DOUBLE else 0.0, usable
        )
        clamped = {n: self.limiter.clamp_angle(n, v) for n, v in targets.items()}
        # Verify the LEAN against the CoM model before it reaches a motor. Skipped
        # while a weight transfer is in progress: there the lean is deliberate and
        # its whole purpose is to move the CoM onto one foot, which the
        # double-support margin is the wrong test for (_lift_gate's stance_margin
        # is the right one, and it is already applied).
        # Sole tilt first: NAO's roll limits are asymmetric (LHipRoll reaches
        # +45.3 deg but RHipRoll only +21.7), so clamping a large symmetric request
        # breaks the tilt-neutrality the composition had.
        self._limit_sole_tilt(clamped)

        # Then SHIFT THE CENTRE OF MASS to make the commanded pose holdable. This
        # is the step that replaces gating: the pose is already at full authority
        # and stays that way while there is pelvis travel left to pay for it.
        margin = self._shift_com(clamped, measured, torso_rp)
        # Re-enforce sole contact after the shift. The shift itself is tilt-neutral
        # by construction, but NAO's roll limits are asymmetric (RHipRoll reaches
        # +21.7 deg where RAnkleRoll reaches -25.3), so CLAMPING it can break that
        # neutrality and leave a sole off flat -- which collapses the very polygon
        # the shift was computed to defend.
        self._limit_sole_tilt(clamped)

        # Only if the compensation has run out of travel is the pose itself scaled
        # back -- and only after it has been short for com_grace_s, because the
        # pelvis needs a few ticks to get there. Double support only: a deliberate
        # weight transfer is judged by stance_margin in _lift_gate, not by the
        # double-support polygon.
        if margin >= p.lean_margin_min:
            self._shift_short_since = None
        elif self._shift_short_since is None:
            self._shift_short_since = now_s
        starved = (self._shift_short_since is not None
                   and (now_s - self._shift_short_since) >= p.com_grace_s)
        lean_scale = 1.0
        if st.shift <= 1e-3 and starved:
            lean_scale = self._limit_asymmetry(clamped, measured)
            self._limit_sole_tilt(clamped)
        meta: dict[str, object] = {
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
            "fsr_share": (None if self._fsr_share is None
                          else round(self._fsr_share, 3)),
            "human_lift": round(human_lift, 4),
            "crouch_u": round(self._crouch_u(obs), 4),
            "crouch_cue": round(obs.crouch_u, 4),
            "crouch_solved": round(
                min(min(-lg.hip_pitch, 0.5 * lg.knee_pitch)
                    for lg in (obs.left, obs.right) if lg is not None)
                if (obs.valid and (obs.left is not None or obs.right is not None))
                else 0.0, 4),
            "tilt_ok": tilt_ok,
            "tracking": usable,
            "lean_scale": round(lean_scale, 3),
            "com_shift_pitch": round(self._shift_ff["pitch"], 4),
            "com_shift_roll": round(self._shift_ff["roll"], 4),
            "com_shift_width": round(self._shift_ff["width"], 4),
            "com_margin": round(self._shift_margin, 5),
            "confidence": round(obs.confidence, 3),
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
        engaged = st.lift > 1e-3 or st.shift > 1e-3
        bar = p.conf_keep if engaged else p.conf_min
        if obs.confidence < bar:
            return f"lower-body confidence {obs.confidence:.2f} < {bar:.2f}"
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
        if self._fsr_share is not None and self._fsr_share < p.fsr_load_frac:
            return (f"stepping at reduced authority: foot sensors report only "
                    f"{self._fsr_share * 100:.0f}% of the load on the "
                    f"{st.stance} foot")
        return f"stepping ({st.lift * 100:.0f}% of the requested lift)"

    def _crouch_u(self, obs: LowerBodyObservation) -> float:
        """Symmetric squat depth to command, in radians.

        A squat has exactly ONE degree of freedom: the crouch posture ties
        ``Hip = -d``, ``Knee = +2d``, ``Ankle = -d`` together so the torso stays
        vertical and the soles flat. So the depth is read off the leg solve as a
        single number and the three joints are rebuilt from it, rather than
        letting three independently-clamped channels drift out of that relation
        (and rather than adding the squat twice -- once here and again as a
        "symmetric pitch deviation", which is what it used to do).

        The depth is the SMALLEST reading available, across two axes and both
        legs, because every other reading can be inflated by something that is
        not a squat:

        * hip vs. knee -- bending at the WAIST flexes the hip relative to the
          torso while the knees stay straight, and NAO has no torso joint to
          render that with. Requiring the knee to agree turns a waist hinge into
          no squat, correctly.
        * left vs. right -- a raised leg is deeply flexed at both hip and knee,
          so averaging the two legs made lifting one knee also squat the robot.
          The straighter leg is the one bearing the weight, and it is the one
          that says how low the body actually is.
        """
        p = self.params
        legs = [lg for lg in (obs.left, obs.right) if lg is not None] if obs.valid else []

        # (a) The leg solve's own reading. Kept because it is the one that carries
        # the two invariants below, and because when the ankle IS visible it is a
        # direct geometric measurement rather than an inference.
        solved = 0.0
        if legs:
            solved = min(min(-lg.hip_pitch, 0.5 * lg.knee_pitch) for lg in legs)

        # (b) The retargeter's dedicated two-cue squat estimate. This used to be
        # thrown away, which made the squat impossible to see on the framing
        # people actually use: with the ankle out of shot ``_leg`` cannot solve the
        # shank, so it reports ``knee_pitch == 0``, and the ``0.5 * knee_pitch``
        # term above then drags (a) to zero however deeply the human bends. The
        # ankle is visible in under a fifth of recorded frames, and the commanded
        # depth accordingly sat on its floor in 96% of them.
        cue = obs.crouch_u if obs.valid else 0.0

        # The LARGER of the two, not the smaller. Both invariants survive it
        # because both sources independently report ~0 in those cases:
        #   * a waist hinge -- (a) sees knee_pitch 0; (b) requires the knees to
        #     agree via its own knee cue, which is also 0.
        #   * one knee lifted -- (a) takes the min across legs; (b) reads the
        #     STRAIGHTER knee, which is the weight-bearing one.
        # So max() adds reach without weakening either guarantee.
        depth = max(solved, cue)

        # base_crouch_u is a floor, not a target: fully locked knees leave the
        # balance loop nothing to work with (KneePitch bottoms out at -5.3 deg).
        return _clamp(max(depth, p.base_crouch_u), 0.0, p.max_crouch_u)

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
        measured: dict[str, float] | None,
        fsr: dict[str, float] | None,
    ) -> tuple[float, float]:
        """Return ``(gate, stance_margin_m)`` -- how much lift is safe right now.

        ``gate`` scales the swing leg's authority continuously from 0 (CoM not
        over the stance foot: do not unload it) to 1 (comfortably over it), so
        the step degrades smoothly instead of snapping on and off.
        """
        p = self.params
        self._fsr_share = None
        if not stance:
            return 0.0, 0.0

        # --- model gate: is the CoM over the stance foot? ------------------
        margin = 0.0
        if self.com_model is None or measured is None:
            gate = p.ungated_lift_cap   # nothing to prove safety with; move a little
        else:
            try:
                margin = float(self.com_model.stance_margin(measured, stance))
            except Exception:  # noqa: BLE001 - no stance_margin / bad state
                gate = p.ungated_lift_cap
            else:
                span = max(p.margin_full - p.margin_min, 1e-6)
                gate = _clamp((margin - p.margin_min) / span, 0.0, 1.0)

        # --- foot sensors: a confirmation, scaled -- never a veto ----------
        if fsr:
            total = float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))
            if total > p.fsr_total_min:
                share = float(fsr.get(stance, 0.0)) / total
                self._fsr_share = share
                span = max(p.fsr_load_frac - 0.5, 1e-6)
                confirm = _clamp((share - 0.5) / span, 0.0, 1.0)
                gate *= p.fsr_min_gain + (1.0 - p.fsr_min_gain) * confirm
        return gate, margin

    def _shift_direction(self, stance: str, measured: dict[str, float] | None,
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
        self, obs: LowerBodyObservation, swing: str, yaw_bias: float, usable: bool
    ) -> dict[str, float]:
        """Symmetric crouch + authority-weighted per-leg human deviation + lean.

        ``usable`` gates only the ANTIsymmetric part of the double-support
        deviation (see ``_apply_symmetric``): a lean/stride asymmetry is what
        tips the robot, so it must not keep being reapplied once tilt_ok is
        already False, the very case the "standing down" gate exists for. The
        symmetric part (wider stance) is CoM-neutral and stays at full
        authority regardless, per the module docstring.
        """
        p = self.params
        st = self.state
        u = self._crouch_u(obs)
        targets = dict(crouch_posture(u))

        left = obs.leg("L") if obs.valid else None
        right = obs.leg("R") if obs.valid else None
        dev_l = self._deviation(left, "L", u) if left is not None else None
        dev_r = self._deviation(right, "R", u) if right is not None else None

        stepping = bool(swing) and st.lift > 1e-4
        if stepping:
            # A foot that is off the ground is unloaded, so it is free to take the
            # human's own pose at whatever authority the safety gate allows. The
            # stance leg is carrying the robot, so it stays near the balanced
            # crouch -- sharing the swing leg's pose onto it would move the very
            # foot the CoM is standing on.
            stance = "R" if swing == "L" else "L"
            # ``_side`` is carried for readability only -- the deviation dict is
            # already keyed by joint name, so the side is not needed again here.
            for _side, dev, weight in (
                (swing, dev_l if swing == "L" else dev_r, st.lift),
                (stance, dev_l if stance == "L" else dev_r, p.asymmetric_gain),
            ):
                if dev is None or weight <= 1e-4:
                    continue
                for name, value in dev.items():
                    targets[name] = targets.get(name, 0.0) + weight * value
        else:
            self._apply_symmetric(targets, dev_l, dev_r, usable)

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

    # NAO's left/right sign conventions differ per axis: the ROLL channels are
    # mirrored (LHipRoll positive and RHipRoll negative both mean "outward"), the
    # PITCH channels are not. So a mirror-symmetric human pose shows up as
    # numerically opposite roll values and numerically equal pitch values.
    _MIRRORED_CHANNELS = ("HipRoll", "AnkleRoll")
    _ALIGNED_CHANNELS = ("HipPitch", "KneePitch", "AnklePitch")

    def _apply_symmetric(
        self,
        targets: dict[str, float],
        dev_l: dict[str, float] | None,
        dev_r: dict[str, float] | None,
        usable: bool,
    ) -> None:
        """Add the human's double-support deviation, split by symmetry.

        The mirror-symmetric half (wider stance, deeper squat) is CoM-neutral and
        enlarges the support polygon, so it passes at full authority regardless of
        ``usable``; the antisymmetric half (a lean, a split stance -- exactly the
        kind of asymmetry that tips the robot) moves the CoM and is gated to zero
        whenever ``usable`` is False, e.g. because tilt_ok already tripped. It must
        not keep reapplying a lean-inducing signal once the robot is already
        "standing down" for being too tilted -- that starves the CoM balance
        correction of ever catching up. See the module docstring.
        """
        p = self.params
        asym_gain = p.asymmetric_gain if usable else 0.0
        for channel, mirrored in (
            *((c, True) for c in self._MIRRORED_CHANNELS),
            *((c, False) for c in self._ALIGNED_CHANNELS),
        ):
            left = dev_l.get("L" + channel) if dev_l else None
            right = dev_r.get("R" + channel) if dev_r else None
            if left is None or right is None:
                # Only one leg in view: there is no way to tell a symmetric pose
                # from a lean, so read the whole thing as antisymmetric, which is
                # the conservative interpretation.
                for side, value in (("L", left), ("R", right)):
                    if value is not None:
                        targets[side + channel] = (
                            targets.get(side + channel, 0.0)
                            + asym_gain * value
                        )
                continue

            if mirrored:
                symmetric = 0.5 * (left - right)   # both legs outward by this much
                antisym = 0.5 * (left + right)     # both rolled the same way = lean
                symmetric = _clamp(symmetric, -p.max_abduction, p.max_abduction)
                l_sym, r_sym = symmetric, -symmetric
            else:
                # The symmetric part of a pitch channel IS the squat, and the
                # crouch posture already carries it (see _crouch_u). Adding it
                # again double-counted the squat and let a straight-legged human
                # cancel the base crouch entirely, locking the knees.
                antisym = 0.5 * (left - right)     # one forward, one back
                l_sym = r_sym = 0.0

            targets["L" + channel] = (
                targets.get("L" + channel, 0.0)
                + p.symmetric_gain * l_sym + asym_gain * antisym
            )
            targets["R" + channel] = (
                targets.get("R" + channel, 0.0)
                + p.symmetric_gain * r_sym
                + asym_gain * (antisym if mirrored else -antisym)
            )

    def _apply_shift(self, targets: dict[str, float], pitch: float, roll: float,
                     width: float = 0.0) -> dict[str, float]:
        """``targets`` with a sole-flat pelvis shift and stance widening added.

        Hip and ankle move by equal and opposite amounts on both legs, so the sum
        Hip+Knee+Ankle -- and therefore each sole's contact -- is untouched while
        the body translates over the feet. The knees are not touched at all, so
        nothing here competes with the joints carrying the squat.

        ``pitch`` and ``roll`` are same-sign on both legs: they translate the
        pelvis. ``width`` is mirror-symmetric: it moves the feet apart, enlarging
        the polygon without moving the centre of mass.
        """
        out = dict(targets)
        for side in ("L", "R"):
            out_sign = 1.0 if side == "L" else -1.0
            for joint, delta in (
                (f"{side}HipPitch", pitch),
                (f"{side}AnklePitch", -pitch),
                (f"{side}HipRoll", roll + out_sign * width),
                (f"{side}AnkleRoll", -roll - out_sign * width),
            ):
                out[joint] = targets.get(joint, 0.0) + delta
        return out

    def _shift_com(self, targets: dict[str, float],
                   measured: dict[str, float] | None,
                   torso_rp: tuple[float, float]) -> float:
        """Move the pelvis so the commanded pose is holdable. Feed-forward.

        This is the layer that replaces gating. The human's leg pose has already
        been composed at full authority; this asks "where must the centre of mass
        be for that pose to stand up?" and shifts the pelvis there, rather than
        refusing to strike the pose. It is the same thing a person does when they
        lean over: the hips go the other way.

        Feed-forward, because it evaluates the pose being COMMANDED rather than the
        one already measured -- so the compensation arrives with the motion instead
        of chasing it. ``BalanceController`` still runs as the feedback half,
        cleaning up what the model got wrong.

        Returns the achieved support margin. Modifies ``targets`` in place.
        """
        p = self.params
        if self.com_model is None:
            return 0.0
        state = dict(measured or {})

        def margin(pitch: float, roll: float, width: float) -> float:
            trial = dict(state)
            trial.update(self._apply_shift(targets, pitch, roll, width))
            try:
                mx, my = self.com_model.support_margins_tilted(trial, torso_rp)
            except AttributeError:
                # An older CoM model without the tilt-aware helper.
                mx, my = self.com_model.support_margins(trial)
            except Exception:  # noqa: BLE001
                return math.inf
            return min(mx, my)

        current = self._shift_ff
        axes = (("pitch", p.com_shift_max_pitch), ("roll", p.com_shift_max_roll),
                ("width", p.com_shift_max_width))

        def evaluate(shift: dict[str, float]) -> float:
            return margin(shift["pitch"], shift["roll"], shift["width"])

        base = evaluate(current)
        if base == math.inf:
            return 0.0

        # A pose that stands up on its own must cost nothing, or the compensation
        # becomes a tax on every movement -- and this is what keeps the typical
        # tick down to a single forward-kinematics pass.
        if base < p.com_margin_target:
            deficit = p.com_margin_target - base
            # Local sensitivity of the margin to each parameter, measured rather
            # than assumed: three probes, one per axis.
            gradients = []
            for name, _limit in axes:
                probe = dict(current)
                probe[name] += p.com_shift_probe
                gradients.append((name, (evaluate(probe) - base) / p.com_shift_probe))
            # Spend the step on the axis that buys the most margin per radian. One
            # axis at a time keeps it monotone: the objective is non-smooth, and
            # moving three parameters at once is how the earlier sampling search
            # walked onto the wrong side of a kink.
            name, gradient = max(gradients, key=lambda g: abs(g[1]))
            committed = False
            if abs(gradient) > 1e-6:
                limit = dict(axes)[name]
                step = _clamp(deficit / gradient,
                              -p.com_shift_max_step, p.com_shift_max_step)
                lo = 0.0 if name == "width" else -limit
                # Evaluate the value actually about to be COMMITTED, not the full
                # step: the objective is non-smooth, so a partial step can be worse
                # than both its endpoints. Checking the endpoint and committing a
                # fraction of it is how the previous version walked onto a kink and
                # then sat there with a margin worse than doing nothing.
                target_value = _clamp(current[name] + step, lo, limit)
                slewed = current[name] + p.com_shift_slew * (target_value - current[name])
                proposed = dict(current)
                proposed[name] = slewed
                if evaluate(proposed) > base:
                    current[name] = slewed
                    committed = True
            if not committed:
                # No parameter can improve matters from here. Give the offset back
                # rather than holding one that is not earning anything -- a stale
                # shift is a posture the imitation has to fight for nothing.
                for key, _limit in axes:
                    current[key] *= (1.0 - p.com_shift_slew)
        else:
            # Comfortable: let the compensation relax back toward neutral so it does
            # not accumulate a permanent offset the imitation has to fight.
            for name, _limit in axes:
                current[name] *= (1.0 - p.com_shift_slew * 0.25)

        shifted = self._apply_shift(targets, current["pitch"], current["roll"],
                                    current["width"])
        for name, value in shifted.items():
            targets[name] = self.limiter.clamp_angle(name, value)
        self._shift_margin = evaluate(current)
        return self._shift_margin

    # Scales tried when a commanded lean has to be reduced. Coarse on purpose: each
    # step costs a full forward-kinematics pass, and the blends upstream are already
    # rate-limited, so a fine search buys nothing but CPU.
    _LEAN_SCALES = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)

    # The two ways an antisymmetric leg pose moves the centre of mass, and how the
    # left/right pair combines for each. NAO mirrors its ROLL channels (LHipRoll
    # positive and RHipRoll negative both mean "outward") but not its pitch ones,
    # so the same-sign roll sum is the lean while the pitch DIFFERENCE is the split
    # stance. Both are scaled by one factor; their mirror-symmetric counterparts
    # (stance width, squat depth) are CoM-neutral or actively helpful and are never
    # touched -- a wider stance enlarges the polygon.
    _ASYM_CHANNELS = (
        ("HipRoll", True), ("AnkleRoll", True),
        ("HipPitch", False), ("KneePitch", False), ("AnklePitch", False),
    )

    def _limit_asymmetry(self, targets: dict[str, float],
                         measured: dict[str, float] | None) -> float:
        """Scale the antisymmetric leg pose until the model says it is holdable.

        Answers the question the module docstring reserves for this layer -- "how
        much of that may the robot actually execute right now?" -- for the one part
        of the pose that moves the centre of mass. Previously the antisymmetric
        deviation was gated by a fixed ``asymmetric_gain`` alone, with nothing
        consulting the robot's own model: the layer could ask for 0.437 rad of lean
        where the support polygon runs out around 0.58, and a split stance was
        unchecked entirely.

        Returns the scale applied (1.0 = untouched), for telemetry. A no-op when
        there is no CoM model to ask, in which case the fixed gain remains the only
        limit, exactly as before.
        """
        if self.com_model is None:
            return 1.0

        # Decompose once: how much antisymmetry is in each channel.
        anti: dict[str, float] = {}
        for channel, mirrored in self._ASYM_CHANNELS:
            left = targets.get("L" + channel, 0.0)
            right = targets.get("R" + channel, 0.0)
            anti[channel] = 0.5 * (left + right) if mirrored else 0.5 * (left - right)
        if all(abs(v) < 1e-4 for v in anti.values()):
            return 1.0

        def with_scale(scale: float) -> dict[str, float]:
            out = dict(targets)
            for channel, mirrored in self._ASYM_CHANNELS:
                drop = (1.0 - scale) * anti[channel]
                out["L" + channel] = targets.get("L" + channel, 0.0) - drop
                out["R" + channel] = targets.get("R" + channel, 0.0) - (
                    drop if mirrored else -drop
                )
            return out

        state = dict(measured or {})
        best_scale, best_margin = None, -math.inf
        for scale in self._LEAN_SCALES:
            trial = dict(state)
            trial.update(with_scale(scale))
            try:
                margin = float(self.com_model.support_margin(trial))
            except Exception:  # noqa: BLE001 - no support_margin on this model
                return 1.0
            if margin >= self.params.lean_margin_min:
                # The LARGEST scale that is holdable: keep as much of the human's
                # pose as the robot can actually stand in.
                if scale < 1.0:
                    for name, value in with_scale(scale).items():
                        targets[name] = self.limiter.clamp_angle(name, value)
                return scale
            if margin > best_margin:
                best_scale, best_margin = scale, margin
        # Nothing is holdable -- the pose is past what this robot can do standing
        # still, whatever it does with its pelvis. Keep the BEST-supported scale
        # rather than collapsing to zero: attenuating to nothing throws away the
        # imitation and does not even buy the best balance.
        if best_scale is not None and best_scale < 1.0:
            for name, value in with_scale(best_scale).items():
                targets[name] = self.limiter.clamp_angle(name, value)
        return best_scale if best_scale is not None else 1.0

    def apply_sole_tilt_limit(self, targets: dict[str, float]) -> dict[str, float]:
        """Keep each sole within ``sole_tilt_budget`` of flat, in place.

        A sole is flat when ``AnkleRoll == -HipRoll``. When the budget is exceeded
        the HIP gives way, not the ankle: standing on the edge of a foot is what
        tips the robot.

        What that costs is worth stating precisely, because it is not what it
        looks like. Sole tilt is ``HipRoll + AnkleRoll``, and both of the roll
        terms this layer generates cancel exactly in that sum -- symmetric stance
        width contributes ``+s`` and ``-s``, the weight-shift lean ``+l`` and
        ``-l``. So the tilt is produced *only* by the CoM balance correction,
        which applies the same sign to both legs; giving way at the hip therefore
        moves the LEAN, never the stance width. That is why balance.py bounds its
        two roll clamps to sum below this budget: it keeps this method a no-op on
        the balance path, rather than letting a saturated roll correction be
        quietly converted into a permanent whole-body lean (measured: a fixed
        0.300 rad lean in 98% of a recorded session).

        Call this **last**, immediately before commanding the motors. The layer
        applies it to its own output, but the caller then folds in the CoM balance
        correction, whose roll terms are not tilt-neutral (``balance.py`` searches
        hip and ankle roll independently) -- so the guarantee only actually holds
        at the final commanded values if it is re-applied there. Idempotent.
        """
        self._limit_sole_tilt(targets)
        return targets

    def _limit_sole_tilt(self, targets: dict[str, float]) -> None:
        """In-place implementation of :meth:`apply_sole_tilt_limit`."""
        budget = self.params.sole_tilt_budget
        for side in ("L", "R"):
            hip_name, ankle_name = f"{side}HipRoll", f"{side}AnkleRoll"
            hip = targets.get(hip_name)
            ankle = targets.get(ankle_name)
            if hip is None or ankle is None:
                continue
            tilt = hip + ankle
            if abs(tilt) <= budget:
                continue
            targets[hip_name] = self.limiter.clamp_angle(
                hip_name, hip - (tilt - math.copysign(budget, tilt))
            )

    def _shift_dir_cached(self, stance: str) -> float:
        cached = self._probe.get(stance)
        return cached[0] if cached is not None else (-1.0 if stance == "L" else 1.0)

    def _deviation(self, leg: LegTarget, side: str, u: float) -> dict[str, float]:
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
