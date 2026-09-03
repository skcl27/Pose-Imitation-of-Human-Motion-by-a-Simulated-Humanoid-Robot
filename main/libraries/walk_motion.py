"""Locate and select Webots NAO walk motions for *actual* locomotion.

The in-place gait engine (``gait.py``) keeps a free-standing NAO upright but does
not translate it across the floor — it marches in place. To make the robot
genuinely **move** (its world coordinates change on the static floor) we drive
Webots' own pre-balanced NAO motion clips (``Forwards``, ``Backwards``,
``TurnLeft``/``TurnRight``, ``SideStep``). Those clips are tuned by Cyberbotics to
keep NAO balanced *and* actually displace it, which is exactly what a Supervisor
base-teleport cannot do (that explodes the physics) and what an untuned online
gait struggles to do on a free-standing robot.

Turning is a yaw *servo*, not a gesture
---------------------------------------
NAO has no torso-yaw joint, so "the human turned round" cannot be imitated by a
joint angle -- the robot has to physically step round. Nor can it be run
open-loop ("human turned left, play one left-turn clip"), because clip and human
turn by different amounts and the error accumulates.

So :func:`plan_action` closes the loop on a *yaw error*: the controller measures
the robot's real heading with the InertialUnit, compares it with the human's
measured torso yaw, and asks for turn clips until the two agree. That makes the
rotation converge regardless of clip size, tracking noise or dropped frames, and
it makes turning work while *standing still* -- the old
an earlier open-loop selector only ever turned as a modifier of walking, which
is why rotating in front of the camera moved nothing but the head.

This module is the **pure, Webots-free** part: finding the motion files on disk
and choosing which one the current situation calls for. The thin Webots
``Motion`` playback wrapper lives in the controller (it needs ``from controller
import Motion``). Keeping the file discovery and selection here makes them
unit-testable on the dev machine, where Webots is not installed.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

# Logical walk action -> candidate Webots NAO motion filenames (first found wins).
# Names differ slightly across Webots releases, so each action lists fallbacks.
# ``forward`` prefers the SHORT clip: clips are played to completion (a clip
# boundary is a balanced double-support pose, which is the only safe place to
# stop), so the clip length is also the latency of "stop walking".
KNOWN_MOTIONS: dict[str, list[str]] = {
    "forward": ["Forwards.motion", "Forward.motion", "Forwards50.motion"],
    "backward": ["Backwards.motion", "Backward.motion"],
    # FINE turn clip: the smallest on disk, listed first on purpose. A clip is
    # played to completion, so the residual heading error after a turn is bounded
    # by half the clip -- which makes a SMALLER clip converge tighter. Preferring
    # the 60 deg clip left a permanent +/-30 deg floor on how accurately the robot
    # could face you; the 40 deg clip halves that to +/-20 deg.
    "turn_left": ["TurnLeft40.motion", "TurnLeft60.motion", "TurnLeft.motion"],
    "turn_right": ["TurnRight40.motion", "TurnRight60.motion", "TurnRight.motion"],
    # COARSE turn clip, for a genuine about-face. Without it, turning to face
    # behind you needs four or five fine clips in a row, each with its own
    # settle-and-restart, which in practice never completed. Webots ships
    # TurnLeft180 and it was simply never listed here.
    "turn_left_coarse": ["TurnLeft180.motion", "TurnLeft120.motion"],
    "turn_right_coarse": ["TurnRight180.motion", "TurnRight120.motion"],
    "side_left": ["SideStepLeft.motion"],
    "side_right": ["SideStepRight.motion"],
}

# Logical actions that rotate the robot, in the order plan_action prefers to
# consider them (it then picks the largest that will not overshoot).
TURN_ACTIONS = {
    "left": ("turn_left", "turn_left_coarse"),
    "right": ("turn_right", "turn_right_coarse"),
}

# Relative path of NAO's motion folder inside a Webots installation.
_NAO_MOTIONS_REL = os.path.join(
    "projects", "robots", "softbank", "nao", "motions"
)


def default_motion_search_dirs(extra: list[str] | None = None) -> list[str]:
    """Candidate directories that hold NAO ``.motion`` files.

    The most reliable entry is ``$WEBOTS_HOME`` (Webots sets it for controller
    processes); the rest cover common install locations on Linux/macOS so the
    discovery still works if the env var is missing. ``extra`` directories (e.g.
    a repo-local ``motions/`` fallback) are searched first.
    """
    dirs: list[str] = list(extra or [])

    home = os.environ.get("WEBOTS_HOME")
    if home:
        dirs.append(os.path.join(home, _NAO_MOTIONS_REL))

    common_roots = [
        "/usr/local/webots",
        "/usr/share/webots",
        "/opt/webots",
        "/snap/webots/current/usr/share/webots",
        "/Applications/Webots.app/Contents",
        os.path.expanduser("~/webots"),
        r"C:\Program Files\Webots",
        r"C:\Program Files (x86)\Webots",
    ]
    for root in common_roots:
        dirs.append(os.path.join(root, _NAO_MOTIONS_REL))

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def find_motion_files(search_dirs: list[str]) -> dict[str, str]:
    """Map each logical action to the first existing motion file found.

    Returns e.g. ``{"forward": "/.../Forwards.motion", "turn_left": "..."}``.
    Actions whose files are not found are simply omitted.
    """
    found: dict[str, str] = {}
    for action, candidates in KNOWN_MOTIONS.items():
        for fname in candidates:
            for d in search_dirs:
                path = os.path.join(d, fname)
                if os.path.isfile(path):
                    found[action] = path
                    break
            if action in found:
                break
    return found


# ---------------------------------------------------------------------------
# Yaw servo / locomotion planning
# ---------------------------------------------------------------------------
_TURN_RE = re.compile(r"Turn(Left|Right)(\d+)?", re.IGNORECASE)
# Turn clips whose filename carries no angle (``TurnLeft.motion``) are treated as
# this many degrees. Webots ships numbered clips, so this is only a fallback.
_UNNUMBERED_TURN_DEG = 40.0


def motion_nominal_yaw(path_or_name: str | None) -> float:
    """Signed nominal yaw (rad) a turn clip produces; 0 for non-turn clips.

    Read off the filename (``TurnLeft60.motion`` -> +60 deg), which is how
    Cyberbotics labels them. Positive = the robot turns to *its own left*.
    The value is only used to avoid firing a clip that would overshoot; the
    actual convergence comes from the measured-yaw feedback loop.
    """
    match = _TURN_RE.search(os.path.basename(path_or_name or ""))
    if match is None:
        return 0.0
    degrees = float(match.group(2)) if match.group(2) else _UNNUMBERED_TURN_DEG
    return math.radians(degrees) * (1.0 if match.group(1).lower() == "left" else -1.0)


@dataclass
class LocomotionParams:
    """Tuning for :func:`plan_action`."""
    # Yaw servo
    # |yaw error| that starts a turn. Matched to the smallest turn clip Webots
    # ships: at overshoot_frac 0.5 a 40 deg clip becomes viable at 20 deg
    # (0.349 rad), so an entry gate above that left a band of heading error where
    # the robot twisted its hip yaw but never actually stepped -- and the hip-yaw
    # bias is capped at 0.12 rad, so that band was where the pelvis sat pinned.
    turn_start_rad: float = 0.35
    turn_stop_rad: float = 0.18      # |yaw error| we consider "aligned" (~10 deg)
    # Refuse a clip that would overshoot: fire only when the error is at least
    # this fraction of the clip's nominal turn, so a 60 deg clip is not used to
    # correct a 15 deg error. At 0.5 the residual heading error is symmetric and
    # bounded by half a clip (+/- 30 deg with Webots' 60 deg turn clips) -- the
    # best any discrete-clip turn can do.
    overshoot_frac: float = 0.5
    # Forward walking
    walk_conf_min: float = 0.6
    walk_cadence_min_hz: float = 0.15


@dataclass(frozen=True)
class LocomotionPlan:
    """What the locomotion layer wants to do this control step."""
    action: str | None   # key into the discovered motion files, or None
    reason: str             # human-readable, for the controller log

    @property
    def is_turn(self) -> bool:
        return bool(self.action) and str(self.action).startswith(("turn_left", "turn_right"))


STAND = LocomotionPlan(None, "stand")


def plan_action(
    *,
    yaw_error_rad: float,
    gait: dict[str, object] | None = None,
    available: dict[str, str] | None = None,
    params: LocomotionParams | None = None,
    turning: bool = False,
    yaw_trustworthy: bool = True,
) -> LocomotionPlan:
    """Decide which pre-balanced motion clip (if any) to play right now.

    Parameters
    ----------
    yaw_error_rad:
        ``wrap_pi(desired_heading - measured_heading)``, i.e. how far the robot
        still has to rotate. **Positive = turn to the robot's own left.** The
        controller computes it from the InertialUnit and the human's torso yaw,
        so this stays a closed loop.
    gait:
        Latest gait command from the cue extractor (``state``, ``cadence_hz``,
        ``conf``); drives forward walking.
    available:
        Motion files discovered on disk (:func:`find_motion_files`). An action is
        never planned unless its clip exists.
    turning:
        True while a turn is already in progress, which switches the yaw gate
        from ``turn_start_rad`` to the tighter ``turn_stop_rad`` -- hysteresis, so
        the robot finishes the rotation instead of stalling one clip short.

    Aligning the heading takes priority over walking forward: walking off along
    the wrong heading is much harder to undo than a slightly late departure.
    """
    p = params or LocomotionParams()
    available = available or {}

    # A turn blocks forward walking while it runs, so it is only started on a
    # heading estimate that is steady (see YawServo.stable). An in-progress
    # rotation is allowed to finish either way -- stopping half-turned is worse
    # than finishing on a slightly stale heading.
    if math.isfinite(yaw_error_rad) and (yaw_trustworthy or turning):
        gate = p.turn_stop_rad if turning else p.turn_start_rad
        if abs(yaw_error_rad) >= gate:
            keys = TURN_ACTIONS["left" if yaw_error_rad > 0.0 else "right"]
            # Of the turn clips that exist for this direction, take the LARGEST
            # one that will not overshoot. Largest, because a big error corrected
            # by a fine clip needs many settle-play-settle cycles and in practice
            # never converged; not-overshooting, because a clip bigger than the
            # error leaves a larger error of the opposite sign than it started
            # with. Together these bound the residual by half the SMALLEST clip on
            # disk while still letting a full about-face happen in one action.
            best: tuple[float, str] | None = None
            for key in keys:
                clip = available.get(key)
                if clip is None:
                    continue
                nominal = abs(motion_nominal_yaw(clip))
                fits = nominal <= 1e-3 or abs(yaw_error_rad) >= p.overshoot_frac * nominal
                if fits and (best is None or nominal > best[0]):
                    best = (nominal, key)
            if best is not None:
                return LocomotionPlan(
                    best[1],
                    f"yaw error {math.degrees(yaw_error_rad):+.0f} deg, "
                    f"{math.degrees(best[0]):.0f} deg clip",
                )

    if gait and "forward" in available:
        marching = str(gait.get("state", "idle")) == "march"
        cadence = float(gait.get("cadence_hz", 0.0) or 0.0)
        conf = float(gait.get("conf", 0.0) or 0.0)
        if marching and cadence >= p.walk_cadence_min_hz and conf >= p.walk_conf_min:
            return LocomotionPlan("forward", f"marching at {cadence:.2f} Hz")

    return STAND


def motion_joints(path: str | None) -> list[str]:
    """Joint names a Webots ``.motion`` clip drives, from its header line.

    A ``.motion`` file is a CSV whose first line is
    ``#WEBOTS_MOTION,V1.0,<joint>,<joint>,...``, so the set of joints it commands
    is declared right there. Reading it lets the controller hand a clip exactly
    the joints it needs instead of the whole robot: Webots' walk clips turn out to
    list only the 12 leg joints, so a whole-body handover was freezing arm and
    head imitation for the length of every step for no reason at all.

    Returns ``[]`` if the file is missing or malformed, which the caller must read
    as "unknown, hand over everything" -- handing over too little would let our
    targets fight the clip's keyframes.
    """
    if not path:
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    except OSError:
        return []
    if not header.startswith("#WEBOTS_MOTION"):
        return []
    fields = [f.strip() for f in header.strip().split(",")]
    # Drop the marker and the version, keep the joint names.
    return [f for f in fields[2:] if f]


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi) -- shared by the yaw servo on both sides."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class YawServo:
    """Closed-loop heading tracker: robot IMU yaw follows the human's torso yaw.

    "The human turned round" is only meaningful as a *relative* rotation -- the
    subject's absolute yaw when they walked in front of the camera is arbitrary,
    and so is the robot's spawn heading. So the servo latches the pair once and
    then tracks the difference::

        desired_robot_yaw = robot_yaw_at_latch + sign * (human_yaw - human_yaw_at_latch)
        error             = wrap_pi(desired_robot_yaw - robot_yaw_now)

    ``error`` is what :func:`plan_action` consumes, and because it is measured
    against the robot's *actual* IMU heading every step the rotation converges
    even though the turn clips are coarse and the tracking is noisy.

    ``sign`` flips the mapping for a mirrored (selfie) camera feed: with
    ``input.flip_horizontal`` on, the on-screen figure turns the opposite way to
    the real subject, so the robot mirroring the screen is the consistent
    behaviour. Expose it as one knob rather than burying a sign in the geometry.
    """

    sign: float = 1.0
    conf_min: float = 0.5
    relatch_after_s: float = 2.0
    # The reference is latched from a short BURST of frames rather than a single
    # one. Latching on one frame makes the whole heading loop inherit that
    # frame's error forever: a recorded session shows the commanded hip-yaw bias
    # pinned at its cap in 94% of frames and pinned to the SAME side in 81%,
    # i.e. a fixed offset rather than tracking. Body yaw is the noisiest cue in
    # the pipeline (it is recovered from shoulder-line foreshortening), so the
    # one frame that happens to arrive first is a poor choice of origin.
    latch_window_s: float = 0.75
    latch_min_samples: int = 5
    # A turn is only worth starting if the heading estimate is steady enough to be
    # believed. Body yaw is recovered from shoulder-line foreshortening and is the
    # noisiest cue in the pipeline: measured on a session where the subject stood
    # in front of the camera moving their arms, the reported yaw swung over a 187
    # degree range, which put the servo error past the turn threshold in 77% of
    # frames. Since turning takes priority over walking, that starved forward
    # locomotion completely -- and because no clip ever ran, the robot never turned,
    # so the error never closed either.
    stability_window_s: float = 1.0
    stability_spread_rad: float = 0.5   # max peak-to-peak yaw to count as steady

    _human_ref: float | None = None
    _robot_ref: float | None = None
    _human: float = 0.0
    _last_seen: float | None = None
    _latch: list[tuple[float, float, float]] = field(default_factory=list)
    _recent: list[tuple[float, float]] = field(default_factory=list)

    @property
    def latched(self) -> bool:
        return self._human_ref is not None

    def _try_latch(self, now_s: float) -> None:
        """Latch the reference pair once the burst is long and dense enough.

        Uses the MEDIAN of each channel, not the mean: a yaw estimate that spikes
        to its +/-90 deg bound on a handful of frames is the documented failure
        mode of the cue, and a median is the estimator that ignores it.
        """
        if len(self._latch) < self.latch_min_samples:
            return
        if (now_s - self._latch[0][0]) < self.latch_window_s:
            return
        humans = sorted(sample[1] for sample in self._latch)
        robots = sorted(sample[2] for sample in self._latch)
        mid = len(humans) // 2
        self._human_ref = humans[mid]
        self._robot_ref = robots[mid]
        self._latch.clear()

    def update(self, *, human_yaw: float | None, conf: float,
               robot_yaw: float, now_s: float) -> None:
        """Ingest one measurement pair; (re)latch the reference when needed."""
        if human_yaw is None or not math.isfinite(human_yaw) or conf < self.conf_min:
            return
        gap = None if self._last_seen is None else (now_s - self._last_seen)
        if gap is not None and gap > self.relatch_after_s:
            # The subject was gone long enough that their old heading tells us
            # nothing: drop the reference and re-latch from a fresh burst rather
            # than chasing a stale error.
            self._human_ref = None
            self._robot_ref = None
            self._latch.clear()
        if self._human_ref is None:
            self._latch.append((now_s, human_yaw, robot_yaw))
            self._try_latch(now_s)
        self._human = human_yaw
        self._last_seen = now_s
        # Keep a short history purely to judge whether the estimate is steady.
        self._recent.append((now_s, human_yaw))
        while self._recent and (now_s - self._recent[0][0]) > self.stability_window_s:
            self._recent.pop(0)

    def error(self, robot_yaw: float) -> float:
        """Yaw the robot still has to turn; positive = to its own left."""
        if self._human_ref is None or self._robot_ref is None:
            return 0.0
        desired = self._robot_ref + self.sign * (self._human - self._human_ref)
        return wrap_pi(desired - robot_yaw)

    def stable(self) -> bool:
        """Is the heading estimate steady enough to act on?

        A turn is an expensive, slow, whole-body commitment that also blocks
        forward walking, so it should only be started on a heading the pipeline is
        actually sure of. Requires a full window of samples spanning less than
        ``stability_spread_rad`` peak-to-peak; an estimate that is swinging is
        reported as unusable rather than averaged into a confident-looking number.
        """
        if len(self._recent) < 3:
            return False
        if (self._recent[-1][0] - self._recent[0][0]) < self.stability_window_s * 0.8:
            return False
        values = [v for _, v in self._recent]
        return (max(values) - min(values)) <= self.stability_spread_rad

    def diagnostics(self, robot_yaw: float) -> dict[str, float]:
        """Everything needed to tell a tracking loop from a fixed offset.

        The distinction matters and cannot be made from the error alone: a servo
        that is converging and one that is stuck 20 deg out both report a
        non-zero error. Logging the reference pair alongside it makes the answer
        obvious on the status line.
        """
        return {
            "latched": 1.0 if self.latched else 0.0,
            "human_ref": 0.0 if self._human_ref is None else self._human_ref,
            "robot_ref": 0.0 if self._robot_ref is None else self._robot_ref,
            "human_now": self._human,
            "robot_now": robot_yaw,
            "error": self.error(robot_yaw),
            "stable": 1.0 if self.stable() else 0.0,
        }

    def reset(self) -> None:
        self._human_ref = None
        self._robot_ref = None
        self._human = 0.0
        self._last_seen = None
        self._latch.clear()
        self._recent.clear()
