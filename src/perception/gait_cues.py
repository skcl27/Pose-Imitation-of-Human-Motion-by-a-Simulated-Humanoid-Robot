"""Monocular gait-cue extraction: human keypoints -> a compact gait command.

The robot cannot copy human *leg joint angles* to walk (monocular depth is
unreliable and direct leg-copy topples a free-standing NAO — see
``main/libraries/nao_retarget`` and the project history). Instead we distil the
human's lower-body motion into a small, depth-free **gait command** that an
on-robot walk engine (``main/libraries/gait.py``) turns into balance-stable
steps. The human supplies *intent* (am I marching? how fast? which leg is up?
stop), not joint targets.

Why these cues are monocular-robust
-----------------------------------
Every cue is a **sign or relative magnitude of normalized, hip-rooted 2D
landmark motion** — never an absolute angle and never depth ``z``:

* Primary signal ``s(t)`` = (left-knee height − right-knee height), normalized by
  shoulder width. It is *anti-phase between the two knees*, so it is large only
  when the legs alternate (marching) and is structurally immune to arm swing
  (arms are not in it). Its zero-crossings give the step cadence; its sign gives
  which knee is currently raised; its amplitude gives how vigorously the human
  marches.
* Normalization by a *smoothed* body width makes the cues scale-invariant, so
  they survive the subject walking toward/away from the camera.
* **Torso yaw** (``body_yaw_rad``) comes from the shoulder/hip line's *rotation
  in the horizontal plane*: the line's lateral extent shrinks as the subject
  turns (foreshortening) while the two endpoints separate in depth. Taking
  ``atan2(depth spread, lateral spread)`` recovers the yaw angle itself, not a
  vague "turn intent", and its sign is the one thing depth is reliable for
  (which shoulder is nearer, not how near). This is what lets the robot rotate
  its whole body instead of just its head: NAO has no torso-yaw joint, so the
  controller servos its measured heading onto this angle by stepping round.

  The lateral term is used **signed**, which is what makes the estimate cover
  the whole circle rather than a quarter of it. Turn past 90 degrees and your
  shoulders swap sides in the image, so the lateral term changes sign -- and that
  sign is exactly the front/back discriminator a single camera is otherwise
  missing. Using ``abs()`` folded the two halves together and bounded the estimate
  to +/-90 degrees, so the robot could never be asked to turn round.

  The lateral term is ``left.x - right.x``, not the other way about. MediaPipe
  labels a person facing the camera with their *left* shoulder on the image's
  right, so ``right.x - left.x`` is negative in the facing-forward case -- which
  would report someone looking straight at the camera as being turned 180 degrees
  away. Measured on recorded runs: negative on 67% of frames with both shoulders
  clearly visible (median -0.085), and 99% of those frames have both ears visible,
  i.e. a face-on view. The hips agree, so it is a labelling convention rather than
  noise, and it holds whether or not the preview is mirrored (MediaPipe cannot
  tell a mirrored subject from a real one, so it labels by appearance either way)::

      facing the camera   lateral > 0, dz ~ 0   ->  yaw ~   0 deg
      turned 90 deg       lateral ~ 0, dz != 0  ->  yaw ~ +/-90 deg
      facing away         lateral < 0, dz ~ 0   ->  yaw ~ +/-180 deg

  Robustness matters more here than anywhere else in this module, because a
  noisy yaw does not merely wobble the robot -- it makes the controller demand
  turn clips in alternating directions, which starves forward walking and trips
  the locomotion failure backoff. Recorded runs showed |yaw| spiking to the
  +/-90 bound on 7-18% of frames while the subject stood square to the camera.
  Three guards fix that: the shoulder span is compared against a self-calibrated
  reference so a collapsed or occluded detection is rejected rather than turned
  into a large angle; the estimate is smoothed on the *shortest arc* (it is a
  circular quantity now, so a plain EMA would take the long way round through
  180 degrees); and a frame that disagrees sharply with the running estimate has
  its confidence halved instead of being trusted.

The extractor is stateful (it keeps a short timestamped history) but is a pure
function of the landmark stream — no camera, no Webots, no RNG — so it is
unit-testable off-simulation and deterministic for recorded-video replay
(PRD Acceptance #3).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

from src.type_defs import Keypoint, PoseFrame

# Lower-body landmarks the cue extractor needs visible to trust a gait reading.
_REQUIRED = ("left_hip", "right_hip", "left_knee", "right_knee")
_OPTIONAL = ("left_ankle", "right_ankle", "left_shoulder", "right_shoulder")

TWO_PI = 2.0 * math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class GaitCommand:
    """Compact, JSON-serializable gait command sent to the walk engine.

    ``state``       : "idle" or "march" (the engine treats anything it doesn't
                      recognize as "idle" -> decay to the stable crouch).
    ``cadence_hz``  : human step-cycle frequency (full L+R cycle), >= 0.
    ``phase``       : current gait phase in radians [0, 2*pi); 0 ~ left-knee up.
    ``swing_side``  : +1 left knee raised, -1 right knee raised, 0 neither.
    ``intensity``   : [0, 1] how vigorously the human marches (knee-lift amp).
    ``turn``        : [-1, 1] ``body_yaw_rad`` normalized by ``TURN_FULL_RAD``,
                      kept for controllers that only understand a turn *intent*.
    ``conf``        : [0, 1] fraction of required lower-body landmarks visible.
    ``body_yaw_rad``: signed torso yaw in radians; 0 = squarely facing the camera,
                      positive = rotated toward the subject's screen-left. This is
                      an *angle*, so the controller can close a heading loop on it.
    ``yaw_conf``    : [0, 1] confidence in ``body_yaw_rad``. Independent of
                      ``conf``: the yaw only needs the shoulders and hips, so it
                      stays usable when the legs leave the frame.
    """

    state: str
    cadence_hz: float
    phase: float
    swing_side: int
    intensity: float
    turn: float
    conf: float
    body_yaw_rad: float = 0.0
    yaw_conf: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "state": self.state,
            "cadence_hz": round(self.cadence_hz, 4),
            "phase": round(self.phase, 4),
            "swing_side": self.swing_side,
            "intensity": round(self.intensity, 4),
            "turn": round(self.turn, 4),
            "conf": round(self.conf, 3),
            "body_yaw_rad": round(self.body_yaw_rad, 4),
            "yaw_conf": round(self.yaw_conf, 3),
        }


IDLE = GaitCommand("idle", 0.0, 0.0, 0, 0.0, 0.0, 0.0)

# Torso yaw that maps to a full-scale (+/-1) legacy ``turn`` value.
TURN_FULL_RAD = math.radians(60.0)
# Depth is only trustworthy for the *sign* of a shoulder/hip depth difference, so
# it is scaled down before entering the yaw solve; the lateral (image-plane) term
# carries the magnitude. Under-reporting the angle is the safe direction: the
# robot then approaches the true heading rather than overshooting it.
YAW_Z_TRUST = 0.75
# Landmark pairs the yaw is averaged over, hips weighted lower (they are noisier
# and clothing-dependent).
_YAW_PAIRS = (("left_shoulder", "right_shoulder", 1.0), ("left_hip", "right_hip", 0.6))
# A body-fixed segment keeps a near-constant length however the subject turns
# (the depth term takes over from the lateral one), so an observed length well
# below the self-calibrated reference means the landmarks are unreliable -- not
# that the subject is at some large angle. Below ``_YAW_SPAN_FLOOR`` of the
# reference the reading is discarded; it ramps to full trust at ``_YAW_SPAN_GOOD``.
_YAW_SPAN_FLOOR = 0.45
_YAW_SPAN_GOOD = 0.70
# Circular EMA rate for the reported yaw. Slow on purpose: this drives a stepping
# servo whose granularity is a 60 degree clip, so chasing per-frame noise buys
# nothing and costs clip thrash.
_YAW_EMA_ALPHA = 0.18
# A single frame disagreeing with the running estimate by more than this is more
# likely noise than a real rotation at camera frame rates, so it is de-weighted.
_YAW_JUMP_RAD = 0.60


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi). Yaw is circular now, so this is load-bearing."""
    return (angle + math.pi) % TWO_PI - math.pi


class _PeakHold:
    """Running maximum: rises quickly, decays very slowly.

    Learns a body-fixed segment's true length from the stream. A projected
    segment can only ever look *shorter* than it is, so the running peak
    converges on the real length; the slow decay lets it follow a genuinely
    different subject or camera distance instead of latching forever.
    """

    __slots__ = ("value", "rise", "decay")

    def __init__(self, rise: float = 0.30, decay: float = 0.004) -> None:
        self.value: float = 0.0
        self.rise = rise
        self.decay = decay

    def update(self, sample: float) -> float:
        if not math.isfinite(sample) or sample <= 0.0:
            return self.value
        if self.value <= 0.0:
            self.value = sample
        elif sample > self.value:
            self.value += self.rise * (sample - self.value)
        else:
            self.value += self.decay * (sample - self.value)
        return self.value


class GaitCueExtractor:
    """Turn a stream of :class:`PoseFrame`s into a smoothed :class:`GaitCommand`.

    Parameters
    ----------
    window_s:
        Length of the timestamped history used for cadence/amplitude (s).
    amp_start / amp_stop:
        Normalized peak-to-peak of the knee-differential signal needed to START
        marching and below which we STOP. ``amp_start > amp_stop`` gives
        hysteresis; the gap (plus the conf gate) rejects jitter and the
        "waving arms while standing still" aliasing case (bias to STOP).
    conf_min:
        Minimum visible-landmark fraction; below it we report idle so the robot
        holds its crouch when the legs leave frame.
    cadence_max_hz:
        Hard clamp on reported cadence (defensive against noisy crossings).
    start_cycles:
        Number of consistent alternating half-steps (zero-crossings) required
        before we declare "march" (slow to start), while we drop to idle
        immediately when the amplitude collapses (instant to stop).
    """

    def __init__(
        self,
        *,
        window_s: float = 1.3,
        amp_start: float = 0.08,
        amp_stop: float = 0.05,
        conf_min: float = 0.6,
        cadence_max_hz: float = 2.5,
        start_cycles: int = 2,
    ) -> None:
        self.window_s = float(window_s)
        self.amp_start = float(amp_start)
        self.amp_stop = float(amp_stop)
        self.conf_min = float(conf_min)
        self.cadence_max_hz = float(cadence_max_hz)
        self.start_cycles = int(start_cycles)

        # (timestamp_s, normalized knee-differential signal s)
        self._hist: Deque[Tuple[float, float]] = deque()
        self._cross_times: Deque[float] = deque(maxlen=6)
        self._last_sign: int = 0
        self._scale_ema: Optional[float] = None  # smoothed body width
        self._yaw_ema: float = 0.0                # smoothed torso yaw (rad)
        self._yaw_seen: bool = False
        # Self-calibrated reference length of each yaw segment (shoulders, hips).
        self._yaw_span = [_PeakHold() for _ in _YAW_PAIRS]
        self._state: str = "idle"

    # -- public API ---------------------------------------------------------
    def update(self, pose: PoseFrame) -> GaitCommand:
        """Ingest one pose frame and return the current gait command."""
        kps = pose.keypoints
        # Torso yaw is computed FIRST and unconditionally: it needs only the
        # shoulders and hips, and the robot must be able to turn to face you
        # while standing perfectly still -- gating it behind the marching
        # confidence is exactly why body rotation used to move nothing but the
        # head.
        yaw, yaw_conf = self._update_yaw(kps)
        turn = _clamp(yaw / TURN_FULL_RAD, -1.0, 1.0)

        conf = self._confidence(kps)
        if conf < self.conf_min:
            # Legs not reliably visible: reset cadence state, hold idle.
            self._decay_to_idle()
            return GaitCommand("idle", 0.0, 0.0, 0, 0.0, turn, conf, yaw, yaw_conf)

        scale = self._body_scale(kps)
        s = self._knee_diff_signal(kps, scale)
        t = pose.timestamp_s
        self._push(t, s)
        self._update_crossings(t, s)

        amp = self._amplitude()
        cadence = self._cadence_hz()
        # Hysteresis state machine (bias to stop).
        if self._state == "march":
            if amp < self.amp_stop or cadence <= 0.0:
                self._state = "idle"
        else:
            if amp >= self.amp_start and len(self._cross_times) >= self.start_cycles:
                self._state = "march"

        if self._state != "march":
            return GaitCommand("idle", 0.0, 0.0, 0, 0.0, turn, conf, yaw, yaw_conf)

        phase = self._phase(s, cadence)
        swing = 1 if s > 0.01 else (-1 if s < -0.01 else 0)
        # Map amplitude to a [0,1] intensity (amp_start..~3x amp_start -> 0..1).
        intensity = _clamp((amp - self.amp_stop) / (3.0 * self.amp_start), 0.0, 1.0)
        cadence = _clamp(cadence, 0.0, self.cadence_max_hz)
        return GaitCommand(
            "march", cadence, phase, swing, intensity, turn, conf, yaw, yaw_conf
        )

    def reset(self) -> None:
        self._hist.clear()
        self._cross_times.clear()
        self._last_sign = 0
        self._scale_ema = None
        self._yaw_ema = 0.0
        self._yaw_seen = False
        self._yaw_span = [_PeakHold() for _ in _YAW_PAIRS]
        self._state = "idle"

    # -- internals ----------------------------------------------------------
    def _confidence(self, kps: Dict[str, Keypoint]) -> float:
        vis = [kps[n].visibility for n in _REQUIRED if n in kps]
        if len(vis) < len(_REQUIRED):
            return 0.0
        return sum(1.0 for v in vis if v >= 0.5) / len(_REQUIRED)

    def _body_scale(self, kps: Dict[str, Keypoint]) -> float:
        """Smoothed body width (shoulder span, hip span fallback) for normalization."""
        width = 0.0
        if "left_shoulder" in kps and "right_shoulder" in kps:
            width = abs(kps["left_shoulder"].x - kps["right_shoulder"].x)
        if width < 1e-3 and "left_hip" in kps and "right_hip" in kps:
            width = abs(kps["left_hip"].x - kps["right_hip"].x)
        width = max(width, 1e-3)
        # EMA so the normalizer doesn't wobble as the subject moves in depth.
        self._scale_ema = width if self._scale_ema is None else (
            self._scale_ema + 0.1 * (width - self._scale_ema)
        )
        return max(self._scale_ema, 1e-3)

    def _knee_diff_signal(self, kps: Dict[str, Keypoint], scale: float) -> float:
        """(left-knee height − right-knee height) / scale. Image y is DOWN, so a
        *raised* knee has a smaller y; height = hip_y − knee_y is larger when the
        knee is up. The differential is anti-phase between legs -> immune to arm
        swing, large only during alternating marching."""
        hip_y = 0.5 * (kps["left_hip"].y + kps["right_hip"].y)
        left_h = hip_y - kps["left_knee"].y
        right_h = hip_y - kps["right_knee"].y
        return (left_h - right_h) / scale

    def _update_yaw(self, kps: Dict[str, Keypoint]) -> Tuple[float, float]:
        """Smoothed torso yaw in radians (full circle) and its confidence.

        The shoulder (and hip) line is a body-fixed horizontal segment, so its
        image-plane extent and its depth spread are the two legs of a right
        triangle whose angle *is* the yaw::

            yaw = atan2(-depth spread, lateral extent)

        with the lateral term taken **signed** so the estimate covers the whole
        circle -- see the module docstring for the sign table and for why the
        robustness guards below exist.
        """
        best_conf = 0.0
        vec_x = 0.0     # accumulate as a VECTOR, not an angle: averaging angles
        vec_y = 0.0     # across the +/-180 seam gives nonsense.
        total_w = 0.0
        for index, (left_name, right_name, weight) in enumerate(_YAW_PAIRS):
            left = kps.get(left_name)
            right = kps.get(right_name)
            if left is None or right is None:
                continue
            vis = min(left.visibility, right.visibility)
            if vis < 0.5:
                continue
            # See the module docstring: MediaPipe puts the LEFT shoulder on the
            # image's right for a subject facing the camera, so this ordering is
            # what makes "facing forward" read as 0 rather than 180 degrees.
            lateral = left.x - right.x
            depth = (right.z - left.z) * YAW_Z_TRUST
            span = math.hypot(lateral, depth)
            if span < 1e-4:
                continue
            # A body-fixed segment has a near-constant length at any yaw, so the
            # observed length against its learned reference tells us whether to
            # believe this frame at all.
            reference = self._yaw_span[index].update(span)
            trust = _clamp(
                (span / max(reference, 1e-6) - _YAW_SPAN_FLOOR)
                / (_YAW_SPAN_GOOD - _YAW_SPAN_FLOOR), 0.0, 1.0
            )
            if trust <= 0.0:
                continue
            w = weight * trust
            vec_x += w * lateral / span
            vec_y += w * -depth / span
            total_w += w
            best_conf = max(best_conf, vis * trust)

        if total_w <= 0.0 or math.hypot(vec_x, vec_y) < 1e-9:
            return self._yaw_ema, 0.0

        raw = math.atan2(vec_y, vec_x)
        if not self._yaw_seen:
            self._yaw_ema = raw
            self._yaw_seen = True
            return self._yaw_ema, _clamp(best_conf, 0.0, 1.0)

        # Smooth along the SHORTEST arc: a plain EMA would travel the long way
        # round whenever the subject crosses the +/-180 seam.
        delta = _wrap_pi(raw - self._yaw_ema)
        if abs(delta) > _YAW_JUMP_RAD:
            # Too big a jump for one camera frame to be a real rotation. Follow it
            # (it may be genuine and we must not latch up) but say we are unsure.
            best_conf *= 0.5
        self._yaw_ema = _wrap_pi(self._yaw_ema + _YAW_EMA_ALPHA * delta)
        return self._yaw_ema, _clamp(best_conf, 0.0, 1.0)

    def _push(self, t: float, s: float) -> None:
        self._hist.append((t, s))
        cutoff = t - self.window_s
        while self._hist and self._hist[0][0] < cutoff:
            self._hist.popleft()

    def _update_crossings(self, t: float, s: float) -> None:
        # Sign with a small deadband so noise near zero doesn't fake crossings.
        sign = 1 if s > 0.02 else (-1 if s < -0.02 else 0)
        if sign != 0 and self._last_sign != 0 and sign != self._last_sign:
            self._cross_times.append(t)
        if sign != 0:
            self._last_sign = sign
        # Drop crossings older than the window.
        cutoff = t - self.window_s
        while self._cross_times and self._cross_times[0] < cutoff:
            self._cross_times.popleft()

    def _amplitude(self) -> float:
        if len(self._hist) < 3:
            return 0.0
        vals = [s for _, s in self._hist]
        return max(vals) - min(vals)  # peak-to-peak

    def _cadence_hz(self) -> float:
        """Full-cycle (L+R) frequency from half-step zero-crossing intervals."""
        if len(self._cross_times) < 2:
            return 0.0
        times = list(self._cross_times)
        intervals = [b - a for a, b in zip(times, times[1:], strict=False) if b > a]
        if not intervals:
            return 0.0
        half_period = sum(intervals) / len(intervals)  # time between crossings
        if half_period <= 1e-3:
            return 0.0
        # One full gait cycle = two half-steps (two crossings).
        return _clamp(1.0 / (2.0 * half_period), 0.0, self.cadence_max_hz)

    def _phase(self, s: float, cadence: float) -> float:
        """Gait phase in [0, 2*pi) from the signal and its derivative.

        For s = A*sin(phi), phi = atan2(s, s_dot/omega). We estimate s_dot from
        the last two samples and omega from the cadence; the on-robot engine only
        uses this to phase-lock its own integrated clock, so an approximate value
        is fine.
        """
        if cadence <= 0.0 or len(self._hist) < 2:
            return 0.0
        (t0, s0), (t1, s1) = self._hist[-2], self._hist[-1]
        dt = max(t1 - t0, 1e-3)
        s_dot = (s1 - s0) / dt
        omega = TWO_PI * cadence
        phi = math.atan2(s, s_dot / omega) if omega > 1e-6 else 0.0
        return phi % TWO_PI

    def _decay_to_idle(self) -> None:
        self._state = "idle"
        self._cross_times.clear()
        self._last_sign = 0
