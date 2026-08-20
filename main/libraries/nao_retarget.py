"""
Full-body retargeting: MediaPipe landmarks -> NAO joint angles.

The Python pipeline streams raw MediaPipe Pose landmarks (normalized image
coordinates: ``x`` right [0,1], ``y`` down [0,1], ``z`` depth, ``visibility``
[0,1]). This module converts them into NAO joint targets for the whole body.

Why retarget here instead of upstream
-------------------------------------
Driving the robot's *full* pose needs the actual limb geometry, not just a
handful of pre-baked angles. Keeping the kinematics next to the robot means the
controller owns everything NAO-specific (joint axes, signs, limits) and the
Python side stays a generic pose source.

Shoulder model
--------------
NAO's shoulder is 2-DOF: ``ShoulderPitch`` (raise the arm up/down in the
sagittal plane) and ``ShoulderRoll`` (abduct the arm sideways). From a frontal
camera the upper-arm direction projects onto the image plane as a vector with a
*vertical* part (up/down) and a *lateral* part (sideways). We decompose that
single observed direction into the two joints:

    lateral_unit = sideways component   ->  ShoulderRoll  = asin(lateral_unit)
    vertical_unit = up component        ->  ShoulderPitch = -asin(vertical_unit)

so arm-down -> pitch +90 deg, arm-up -> pitch -90 deg, arm-straight-out ->
roll +/-90 deg, and diagonals split cleanly between the two.

Leg model (:class:`LowerBodyRetargeter`)
----------------------------------------
The legs used to be driven only as a symmetric averaged crouch, which threw away
exactly the information a leg lift carries: raise one knee and the *average*
barely moves, so the robot looked frozen. The legs are now solved **per side and
in closed form** from the same frontal projection, using NAO's real leg chain
order (HipYawPitch -> HipRoll -> HipPitch -> KneePitch -> AnklePitch/Roll).

For a thigh at hip roll ``phi`` and hip pitch ``theta``, the thigh direction in
the torso frame is::

    R_x(phi) . R_y(theta) . (0, 0, -1)
      = ( -sin(theta),  sin(phi)cos(theta),  -cos(phi)cos(theta) )
         ^ forward       ^ lateral (left)      ^ vertical (up)

A frontal camera observes the lateral and vertical components directly (the
forward one is the foreshortened, unobservable axis), which makes the system
*exactly solvable*::

    d    = -up_obs            = cos(phi) cos(theta)
    phi  = atan2(lat_obs, d)                       # abduction, fully observable
    theta = +/- acos(d / cos(phi))                 # magnitude from foreshortening

The remaining sign of ``theta`` (thigh forward vs. backward) is the one thing a
single frontal view cannot see, so it is taken from the landmark depth ``z`` with
a deadband and a documented bias toward *forward* (human knee lifts are forward,
and NAO's HipPitch range is -88..+27.7 deg, i.e. mostly forward anyway).

The shank shares the hip roll and adds KneePitch about the same y axis, so the
identical solve on the knee->ankle segment yields ``theta_h + theta_k`` and hence
KneePitch; the sole is then levelled by ``AnklePitch = -(theta_h + theta_k)`` and
``AnkleRoll = -phi``. Everything falls out of one consistent model instead of
hand-tuned gains.

Self-calibration
----------------
Segment *reference* lengths (unforeshortened thigh / shank, upright hip height)
are learned from the stream by a peak-hold tracker normalized by torso length,
so the solve is scale-invariant and needs no per-user calibration step: walk in
front of the camera and the references settle within a second.

Everything is gated on landmark ``visibility`` so out-of-frame joints are simply
not commanded and the driver holds their last pose. The *safety* of a leg pose
(may the robot actually unload a foot right now?) is deliberately NOT decided
here -- that needs the robot's own CoM/force state and lives in
``lower_body.LowerBodyController``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from pose_control_utils import JointLimiter, get_default_motor_configs

# Visibility below which a landmark is considered unreliable and its dependent
# joints are skipped.
VIS_THRESHOLD = 0.5

# Tuning gains (kept gentle; joint limits clamp the rest).
ROLL_GAIN = 1.0
PITCH_GAIN = 1.0
HEAD_YAW_GAIN = 2.2
HEAD_PITCH_GAIN = 1.6
HEAD_PITCH_BASELINE = 0.9  # nose sits ~0.9 shoulder-widths above shoulder line

# ---------------------------------------------------------------------------
# Lower-body tuning
# ---------------------------------------------------------------------------
# How much of the reference leg length the foot must clear before we call it a
# lift, and how much clearance counts as a *full* lift (knee-high march).
LIFT_DEADBAND = 0.030
LIFT_FULL = 0.260
# Squat: hip height (above the ground line, in reference-leg-length units) has
# to drop by this fraction for a full crouch, and the knees must agree.
CROUCH_FULL_DROP = 0.28
KNEE_STRAIGHT_DEADZONE = 0.20  # rad of knee bend treated as "standing straight"
KNEE_BEND_RANGE = 1.30         # rad of human knee bend mapped to full crouch
MAX_CROUCH = 0.35              # rad; symmetric crouch amplitude u in [0, MAX_CROUCH]

# Depth (``z``) is the least reliable MediaPipe channel, so it is used only to
# pick the SIGN of the unobservable sagittal axis, and only past a deadband.
Z_SIGN_DEADBAND = 0.06   # in reference-segment-length units
# Below this |cos(hip_roll)| the sagittal angle is geometrically unobservable
# (the limb points nearly straight sideways), so we report pitch 0 rather than a
# noise-amplified value.
MIN_COS_ROLL = 0.30

Vec = Tuple[float, float, float]
Landmark = Tuple[float, float, float, float]  # x, y, z, visibility


# ---------------------------------------------------------------------------
# Small vector helpers (image coords: x right, y down, z depth)
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _asin(v: float) -> float:
    return math.asin(_clamp(v, -1.0, 1.0))


def _acos(v: float) -> float:
    return math.acos(_clamp(v, -1.0, 1.0))


def _sub(a: Landmark, b: Landmark) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Vec) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-9


def _angle_between(a: Vec, b: Vec) -> float:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.acos(_clamp(dot / (_norm(a) * _norm(b)), -1.0, 1.0))


def _hypot2(a: Landmark, b: Landmark) -> float:
    """In-image-plane distance between two landmarks (depth ignored)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------------------
# Landmark access
# ---------------------------------------------------------------------------
def _parse(keypoints: Dict[str, Sequence[float]]) -> Dict[str, Landmark]:
    """Normalize incoming landmark values to (x, y, z, visibility) tuples."""
    out: Dict[str, Landmark] = {}
    for name, v in keypoints.items():
        try:
            x = float(v[0])
            y = float(v[1])
            z = float(v[2]) if len(v) > 2 else 0.0
            vis = float(v[3]) if len(v) > 3 else 1.0
        except (TypeError, IndexError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        out[name] = (x, y, z, vis)
    return out


def _visible(kps: Dict[str, Landmark], *names: str, thr: float = VIS_THRESHOLD) -> bool:
    return all(n in kps and kps[n][3] >= thr for n in names)


def _mid_point(kps: Dict[str, Landmark], *names: str) -> Optional[Landmark]:
    """Midpoint of whichever of ``names`` are visible, or None if none are.

    Degrading to a single landmark (rather than requiring the pair) is what keeps
    the lower body alive when one hip or foot is briefly occluded.
    """
    pts = [kps[n] for n in names if _visible(kps, n)]
    if not pts:
        return None
    n = float(len(pts))
    return (
        sum(p[0] for p in pts) / n,
        sum(p[1] for p in pts) / n,
        sum(p[2] for p in pts) / n,
        min(p[3] for p in pts),
    )


# ---------------------------------------------------------------------------
# Per-segment retargeting (upper body)
# ---------------------------------------------------------------------------
def _arm(kps: Dict[str, Landmark], side: str, mid_x: float) -> Dict[str, float]:
    pre = "left_" if side == "L" else "right_"
    if not _visible(kps, pre + "shoulder", pre + "elbow"):
        return {}

    s = kps[pre + "shoulder"]
    e = kps[pre + "elbow"]
    dx = e[0] - s[0]
    dy = e[1] - s[1]
    length = math.hypot(dx, dy) + 1e-9

    vertical_up = -dy / length                       # +1 elbow above shoulder
    out_dir = 1.0 if (s[0] - mid_x) >= 0.0 else -1.0  # image side -> "outward"
    lateral_out = (dx * out_dir) / length             # +1 arm abducted outward

    pitch = -_asin(vertical_up) * PITCH_GAIN          # +down, -up
    roll_mag = _asin(lateral_out) * ROLL_GAIN         # >=0 outward, <0 across body

    out: Dict[str, float] = {}
    if side == "L":
        out["LShoulderPitch"] = pitch
        out["LShoulderRoll"] = +roll_mag             # NAO L: positive = outward
    else:
        out["RShoulderPitch"] = pitch
        out["RShoulderRoll"] = -roll_mag             # NAO R: negative = outward

    # Elbow flexion: angle between upper arm and forearm (0 = straight).
    if _visible(kps, pre + "wrist"):
        w = kps[pre + "wrist"]
        bend = _angle_between(_sub(e, s), _sub(w, e))
        if side == "L":
            out["LElbowRoll"] = -bend                # NAO L elbow bends negative
        else:
            out["RElbowRoll"] = +bend                # NAO R elbow bends positive
    return out


def _head(kps: Dict[str, Landmark]) -> Dict[str, float]:
    if not _visible(kps, "nose", "left_shoulder", "right_shoulder"):
        return {}
    nose = kps["nose"]
    ls = kps["left_shoulder"]
    rs = kps["right_shoulder"]
    mid_x = (ls[0] + rs[0]) / 2.0
    mid_y = (ls[1] + rs[1]) / 2.0
    shoulder_w = abs(ls[0] - rs[0]) + 1e-6

    # Yaw: nose horizontal offset from the shoulder midline.
    yaw = ((nose[0] - mid_x) / shoulder_w) * HEAD_YAW_GAIN
    # Pitch: nose vertical offset relative to its typical above-shoulder height.
    # Looking down brings the nose lower (toward the shoulders) -> positive pitch.
    pitch_raw = (nose[1] - mid_y) / shoulder_w        # negative when nose is high
    pitch = (pitch_raw + HEAD_PITCH_BASELINE) * HEAD_PITCH_GAIN
    return {"HeadYaw": yaw, "HeadPitch": pitch}


def _knee_bend(kps: Dict[str, Landmark], side: str) -> Optional[float]:
    """Human knee flexion (rad, 0 = straight) from hip-knee-ankle, or None."""
    pre = "left_" if side == "L" else "right_"
    if not _visible(kps, pre + "hip", pre + "knee", pre + "ankle"):
        return None
    thigh = _sub(kps[pre + "knee"], kps[pre + "hip"])
    shank = _sub(kps[pre + "ankle"], kps[pre + "knee"])
    return _angle_between(thigh, shank)


# ---------------------------------------------------------------------------
# Lower body: self-calibrating, per-leg closed-form solve
# ---------------------------------------------------------------------------
class PeakHold:
    """Running maximum of a signal: rises quickly, decays very slowly.

    Used to learn the subject's *unforeshortened* body proportions from the
    landmark stream. An instantaneous value is always <= the true length (a limb
    can only look shorter in projection, never longer), so the running peak
    converges on the real one. The slow decay lets the estimate follow a
    genuinely different subject or camera distance instead of latching forever.
    """

    __slots__ = ("value", "rise", "decay")

    def __init__(self, rise: float = 0.35, decay: float = 0.004,
                 initial: Optional[float] = None) -> None:
        self.value: Optional[float] = initial
        self.rise = rise
        self.decay = decay

    def update(self, sample: float) -> float:
        if not math.isfinite(sample) or sample <= 0.0:
            return self.value if self.value is not None else 0.0
        if self.value is None:
            self.value = sample
        elif sample > self.value:
            self.value += self.rise * (sample - self.value)
        else:
            self.value += self.decay * (sample - self.value)
        return self.value


@dataclass
class LegTarget:
    """One leg's retargeted NAO angles plus how far the human lifted that foot."""
    hip_pitch: float = 0.0
    hip_roll: float = 0.0
    knee_pitch: float = 0.0
    ankle_pitch: float = 0.0
    ankle_roll: float = 0.0
    lift: float = 0.0          # 0 = planted, 1 = knee-high lift
    confidence: float = 0.0    # 0..1 from landmark visibility

    def as_targets(self, side: str) -> Dict[str, float]:
        """Expand to NAO joint names for ``side`` in ("L", "R")."""
        return {
            f"{side}HipPitch": self.hip_pitch,
            f"{side}HipRoll": self.hip_roll,
            f"{side}KneePitch": self.knee_pitch,
            f"{side}AnklePitch": self.ankle_pitch,
            f"{side}AnkleRoll": self.ankle_roll,
        }


@dataclass
class LowerBodyObservation:
    """Everything the lower-body controller needs from one camera frame."""
    left: Optional[LegTarget] = None
    right: Optional[LegTarget] = None
    crouch_u: float = 0.0        # rad; symmetric squat amplitude (0 = upright)
    stance_side: str = ""        # "L" / "R" / "" (both feet down)
    confidence: float = 0.0      # 0..1 overall lower-body confidence
    valid: bool = False

    def leg(self, side: str) -> Optional[LegTarget]:
        return self.left if side == "L" else self.right


@dataclass
class BodyGeometry:
    """Self-calibrating estimate of the subject's proportions (all normalized
    by torso length, so the estimate is invariant to camera distance)."""
    torso: float = 0.0
    thigh_ratio: PeakHold = field(default_factory=PeakHold)
    shank_ratio: PeakHold = field(default_factory=PeakHold)
    hip_height_ratio: PeakHold = field(default_factory=PeakHold)
    _torso_ema: Optional[float] = None

    def update_torso(self, kps: Dict[str, Landmark]) -> float:
        ls, rs = kps["left_shoulder"], kps["right_shoulder"]
        mid_sh = ((ls[0] + rs[0]) * 0.5, (ls[1] + rs[1]) * 0.5, 0.0, 1.0)
        mid_hip = _mid_point(kps, "left_hip", "right_hip")
        if mid_hip is None:
            return self.torso
        # Shoulder span is a useful floor: it keeps the scale sane when the
        # subject leans and the torso projects short.
        span = max(_hypot2(ls, rs), 1e-4)
        raw = max(_hypot2(mid_sh, mid_hip), 0.6 * span, 1e-4)
        self._torso_ema = raw if self._torso_ema is None else (
            self._torso_ema + 0.15 * (raw - self._torso_ema)
        )
        self.torso = self._torso_ema
        return self.torso

    @property
    def thigh(self) -> float:
        return max((self.thigh_ratio.value or 0.0) * self.torso, 1e-4)

    @property
    def shank(self) -> float:
        return max((self.shank_ratio.value or 0.0) * self.torso, 1e-4)

    @property
    def leg_length(self) -> float:
        return self.thigh + self.shank

    @property
    def calibrated(self) -> bool:
        return (
            self.torso > 1e-4
            and self.thigh_ratio.value is not None
            and self.shank_ratio.value is not None
        )


def _solve_segment(
    lat_obs: float, up_obs: float, roll: Optional[float], forward_hint: float
) -> Tuple[float, float]:
    """Solve one limb segment's (roll, pitch) from its frontal projection.

    ``lat_obs`` / ``up_obs`` are the segment's lateral (outward-positive) and
    upward components divided by its *reference* length, i.e. the observable two
    of the three unit-vector components of

        R_x(roll) . R_y(pitch) . (0, 0, -1)
          = (-sin(pitch), sin(roll)cos(pitch), -cos(roll)cos(pitch)).

    Pass ``roll=None`` to solve it (thigh: the hip has a roll DOF) or a known
    value to reuse it (shank: the knee has none). ``forward_hint`` > 0 means the
    depth channel says the segment points away from the camera, i.e. *backward*.

    Returns ``(roll, pitch)`` in radians, with pitch 0 = segment hanging straight
    down and negative = pointing forward (NAO's HipPitch/KneePitch sense).
    """
    d = _clamp(-up_obs, -1.0, 1.0)             # cos(roll) * cos(pitch)
    if roll is None:
        roll = math.atan2(lat_obs, d)
    cos_roll = math.cos(roll)
    if abs(cos_roll) < MIN_COS_ROLL:
        # Limb is nearly horizontal-sideways: the sagittal angle is not
        # observable from a frontal view, so claim nothing rather than amplify
        # noise into a large bogus pitch.
        return roll, 0.0
    pitch_mag = _acos(_clamp(d / cos_roll, -1.0, 1.0))
    # Depth resolves forward vs. backward; bias to forward (see module docstring).
    backward = forward_hint > Z_SIGN_DEADBAND
    return roll, (pitch_mag if backward else -pitch_mag)


class LowerBodyRetargeter:
    """Stateful per-leg retargeting of MediaPipe landmarks to NAO leg angles.

    Stateful because it self-calibrates the subject's segment lengths (see
    :class:`PeakHold`); apart from that it is a pure function of the landmark
    stream -- no Webots, no camera, no RNG -- so it is unit-testable
    off-simulation.
    """

    def __init__(self) -> None:
        self.geom = BodyGeometry()

    # -- public ------------------------------------------------------------
    def observe(self, keypoints: Dict[str, Sequence[float]]) -> LowerBodyObservation:
        """Retarget one frame of landmarks into a :class:`LowerBodyObservation`."""
        return self.observe_parsed(_parse(keypoints))

    def observe_parsed(self, kps: Dict[str, Landmark]) -> LowerBodyObservation:
        # Shoulders give the scale; ONE visible hip is enough to root the legs.
        # Requiring both used to drop the entire lower body whenever a hip was
        # briefly occluded -- i.e. exactly when the subject turned or stepped.
        if not _visible(kps, "left_shoulder", "right_shoulder"):
            return LowerBodyObservation()
        if _mid_point(kps, "left_hip", "right_hip") is None:
            return LowerBodyObservation()

        self.geom.update_torso(kps)
        self._calibrate_segments(kps)
        if not self.geom.calibrated:
            return LowerBodyObservation()

        ground_y = self._ground_line(kps)
        left = self._leg(kps, "L", ground_y)
        right = self._leg(kps, "R", ground_y)
        if left is None and right is None:
            return LowerBodyObservation()

        legs = [lg for lg in (left, right) if lg is not None]
        confidence = sum(lg.confidence for lg in legs) / len(legs)
        # Two legs seen is materially more trustworthy than one: a single-leg
        # read cannot tell a lift from the other foot leaving the frame.
        if len(legs) < 2:
            confidence *= 0.5

        stance = ""
        if left is not None and right is not None:
            if left.lift > right.lift + 0.08:
                stance = "R"
            elif right.lift > left.lift + 0.08:
                stance = "L"

        return LowerBodyObservation(
            left=left,
            right=right,
            crouch_u=self._crouch(kps, ground_y),
            stance_side=stance,
            confidence=_clamp(confidence, 0.0, 1.0),
            valid=True,
        )

    # -- internals ---------------------------------------------------------
    def _calibrate_segments(self, kps: Dict[str, Landmark]) -> None:
        """Learn reference thigh/shank lengths (torso-normalized peak-hold)."""
        torso = max(self.geom.torso, 1e-4)
        for side in ("L", "R"):
            pre = "left_" if side == "L" else "right_"
            if _visible(kps, pre + "hip", pre + "knee"):
                self.geom.thigh_ratio.update(
                    _hypot2(kps[pre + "knee"], kps[pre + "hip"]) / torso
                )
            if _visible(kps, pre + "knee", pre + "ankle"):
                self.geom.shank_ratio.update(
                    _hypot2(kps[pre + "ankle"], kps[pre + "knee"]) / torso
                )

    def _foot_y(self, kps: Dict[str, Landmark], side: str) -> Optional[float]:
        """Image y of a foot: ankle, refined with the heel when it is visible."""
        pre = "left_" if side == "L" else "right_"
        if not _visible(kps, pre + "ankle"):
            return None
        ys = [kps[pre + "ankle"][1]]
        if _visible(kps, pre + "heel"):
            ys.append(kps[pre + "heel"][1])
        return sum(ys) / len(ys)

    def _ground_line(self, kps: Dict[str, Landmark]) -> Optional[float]:
        """Image y of the ground: the LOWER of the two feet.

        This is the trick that makes lift detection calibration-free -- whichever
        foot is planted defines the floor, so the other foot's rise above it is
        the lift, with no need to know where the real floor is in the image.
        """
        ys = [y for y in (self._foot_y(kps, "L"), self._foot_y(kps, "R")) if y is not None]
        return max(ys) if ys else None

    def _leg(
        self, kps: Dict[str, Landmark], side: str, ground_y: Optional[float]
    ) -> Optional[LegTarget]:
        pre = "left_" if side == "L" else "right_"
        if not _visible(kps, pre + "hip", pre + "knee"):
            return None
        hip = kps[pre + "hip"]
        knee = kps[pre + "knee"]
        ankle = kps[pre + "ankle"] if _visible(kps, pre + "ankle") else None

        mid_sh_x = (kps["left_shoulder"][0] + kps["right_shoulder"][0]) / 2.0
        # "Outward" for this leg, derived from the data so the mapping is
        # independent of whether the camera image was mirrored. With only one hip
        # visible its own x is useless as a midline, so fall back to the
        # shoulders'.
        both_hips = _visible(kps, "left_hip", "right_hip")
        mid_hip = _mid_point(kps, "left_hip", "right_hip")
        ref_x = mid_sh_x
        if both_hips and mid_hip is not None and abs(hip[0] - mid_hip[0]) > 1e-4:
            ref_x = mid_hip[0]
        out_dir = 1.0 if (hip[0] - ref_x) >= 0.0 else -1.0

        thigh_len = self.geom.thigh
        lat = ((knee[0] - hip[0]) * out_dir) / thigh_len
        up = -(knee[1] - hip[1]) / thigh_len
        fwd_hint = (knee[2] - hip[2]) / thigh_len       # +ve => knee further away
        roll_mag, hip_pitch = _solve_segment(
            _clamp(lat, -1.0, 1.0), _clamp(up, -1.0, 1.0), None, fwd_hint
        )

        total = hip_pitch          # theta_h + theta_k, defaults to knee straight
        if ankle is not None:
            shank_len = self.geom.shank
            lat_s = ((ankle[0] - knee[0]) * out_dir) / shank_len
            up_s = -(ankle[1] - knee[1]) / shank_len
            fwd_s = (ankle[2] - knee[2]) / shank_len
            # The shank shares the hip roll (the knee has no roll DOF), so reuse
            # roll_mag and read out theta_h + theta_k directly.
            _, total_signed = _solve_segment(
                _clamp(lat_s, -1.0, 1.0), _clamp(up_s, -1.0, 1.0), roll_mag, fwd_s
            )
            # Knees do not hyperextend: of the two sign branches keep the one
            # that yields a non-negative KneePitch.
            total = total_signed if (total_signed - hip_pitch) >= 0.0 else -total_signed
        knee_pitch = max(0.0, total - hip_pitch)
        total = hip_pitch + knee_pitch

        # NAO roll signs: LHipRoll positive = left leg outward, RHipRoll negative
        # = right leg outward. AnkleRoll cancels it so the sole stays level.
        hip_roll = roll_mag if side == "L" else -roll_mag

        lift = 0.0
        foot_y = self._foot_y(kps, side)
        if ground_y is not None and foot_y is not None:
            raw = (ground_y - foot_y) / max(self.geom.leg_length, 1e-4)
            lift = _clamp((raw - LIFT_DEADBAND) / LIFT_FULL, 0.0, 1.0)

        names = [pre + "hip", pre + "knee"] + ([pre + "ankle"] if ankle else [])
        conf = sum(kps[n][3] for n in names) / len(names)

        return LegTarget(
            hip_pitch=hip_pitch,
            hip_roll=hip_roll,
            knee_pitch=knee_pitch,
            ankle_pitch=-total,       # level the sole (all three rotate about y)
            ankle_roll=-hip_roll,     # level the sole laterally
            lift=lift,
            confidence=_clamp(conf, 0.0, 1.0),
        )

    def _crouch(self, kps: Dict[str, Landmark], ground_y: Optional[float]) -> float:
        """Symmetric squat amplitude u (rad) for the balanced crouch posture.

        Two independent cues must agree before the robot squats: the hips
        actually dropped toward the ground line, AND the knees are actually bent.
        Requiring both rejects the common false positive of the subject simply
        stepping further from the camera.
        """
        bends = [b for b in (_knee_bend(kps, "L"), _knee_bend(kps, "R")) if b is not None]
        if not bends:
            return 0.0
        # The STRAIGHTER knee, not the average: while one leg is lifted its own
        # deep knee fold says nothing about how low the body is, and averaging it
        # in would make every march step look like a squat.
        knee_cue = _clamp(
            (min(bends) - KNEE_STRAIGHT_DEADZONE) / KNEE_BEND_RANGE, 0.0, 1.0
        )

        height_cue = 1.0
        mid_hip = _mid_point(kps, "left_hip", "right_hip")
        if ground_y is not None and mid_hip is not None:
            ratio = (ground_y - mid_hip[1]) / max(self.geom.torso, 1e-4)
            ref = self.geom.hip_height_ratio.update(ratio)
            if ref > 1e-4:
                height_cue = _clamp((1.0 - ratio / ref) / CROUCH_FULL_DROP, 0.0, 1.0)

        return min(knee_cue, height_cue) * MAX_CROUCH


# ---------------------------------------------------------------------------
# Legacy symmetric crouch (kept for the walk engine's idle posture)
# ---------------------------------------------------------------------------
def crouch_posture(u: float) -> Dict[str, float]:
    """Symmetric, statically-balanced crouch: hip -u, knee +2u, ankle -u.

    ``HipPitch + KneePitch + AnklePitch == 0`` keeps the torso vertical and the
    feet flat, and thigh ~= shank length keeps the hip over the ankle, so the
    centre of mass stays inside the foot polygon and NAO holds the squat
    statically. This is the posture every lower-body mode decays back to.
    """
    return {
        "LHipPitch": -u, "RHipPitch": -u,
        "LKneePitch": 2.0 * u, "RKneePitch": 2.0 * u,
        "LAnklePitch": -u, "RAnklePitch": -u,
        "LHipRoll": 0.0, "RHipRoll": 0.0,
        "LAnkleRoll": 0.0, "RAnkleRoll": 0.0,
        "LHipYawPitch": 0.0, "RHipYawPitch": 0.0,
    }


def _swap_sides(targets: Dict[str, float]) -> Dict[str, float]:
    """Swap L<->R joints for a mirror-image mapping."""
    swapped: Dict[str, float] = {}
    for name, value in targets.items():
        if name.startswith("L"):
            swapped["R" + name[1:]] = value
        elif name.startswith("R"):
            swapped["L" + name[1:]] = value
        else:
            swapped[name] = value
    return swapped


# ---------------------------------------------------------------------------
# Public entry point (upper body + head; legs go through LowerBodyRetargeter)
# ---------------------------------------------------------------------------
def retarget_upper_body(
    keypoints: Dict[str, Sequence[float]],
    *,
    drive_head: bool = True,
    swap_sides: bool = False,
    limiter: Optional[JointLimiter] = None,
) -> Dict[str, float]:
    """Map MediaPipe landmarks to clamped NAO arm/head targets (radians).

    Only joints whose source landmarks are visible are returned; everything
    else is omitted so the caller can hold the previous pose.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    kps = _parse(keypoints)

    targets: Dict[str, float] = {}
    if _visible(kps, "left_shoulder", "right_shoulder"):
        mid_x = (kps["left_shoulder"][0] + kps["right_shoulder"][0]) / 2.0
        targets.update(_arm(kps, "L", mid_x))
        targets.update(_arm(kps, "R", mid_x))
    if drive_head:
        targets.update(_head(kps))

    if swap_sides:
        targets = _swap_sides(targets)
    return {name: limiter.clamp_angle(name, value) for name, value in targets.items()}


def retarget_full_body(
    keypoints: Dict[str, Sequence[float]],
    *,
    drive_legs: bool = False,
    drive_head: bool = True,
    swap_sides: bool = False,
    limiter: Optional[JointLimiter] = None,
    retargeter: Optional[LowerBodyRetargeter] = None,
) -> Dict[str, float]:
    """Arms + head, plus (when ``drive_legs``) the raw per-leg leg solve.

    NOTE: the leg angles returned here are the *human's* pose, with no balance
    safety applied. Production control goes through
    ``lower_body.LowerBodyController``, which gates and blends them against the
    robot's own CoM/force state. This entry point exists for the joint-angle
    fallback path and for tests.
    """
    limiter = limiter or JointLimiter(get_default_motor_configs())
    targets = retarget_upper_body(
        keypoints, drive_head=drive_head, swap_sides=False, limiter=limiter
    )

    if drive_legs:
        obs = (retargeter or LowerBodyRetargeter()).observe(keypoints)
        if obs.valid:
            for side in ("L", "R"):
                leg = obs.leg(side)
                if leg is not None:
                    targets.update(leg.as_targets(side))

    if swap_sides:
        targets = _swap_sides(targets)
    return {name: limiter.clamp_angle(name, value) for name, value in targets.items()}


def retargetable_joints(drive_legs: bool = False, drive_head: bool = True) -> List[str]:
    """The set of NAO joints this module can drive (for logging headers)."""
    joints = [
        "LShoulderPitch", "RShoulderPitch",
        "LShoulderRoll", "RShoulderRoll",
        "LElbowRoll", "RElbowRoll",
    ]
    if drive_head:
        joints += ["HeadYaw", "HeadPitch"]
    if drive_legs:
        for side in ("L", "R"):
            joints += [
                f"{side}HipPitch", f"{side}HipRoll", f"{side}KneePitch",
                f"{side}AnklePitch", f"{side}AnkleRoll",
            ]
    return joints
