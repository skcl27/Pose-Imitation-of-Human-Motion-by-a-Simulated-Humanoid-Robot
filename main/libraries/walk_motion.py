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
:func:`select_action` only ever turned as a modifier of walking, which is why
rotating in front of the camera moved nothing but the head.

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
from dataclasses import dataclass
from typing import Dict, List, Optional

# Logical walk action -> candidate Webots NAO motion filenames (first found wins).
# Names differ slightly across Webots releases, so each action lists fallbacks.
# ``forward`` prefers the SHORT clip: clips are played to completion (a clip
# boundary is a balanced double-support pose, which is the only safe place to
# stop), so the clip length is also the latency of "stop walking".
KNOWN_MOTIONS: Dict[str, List[str]] = {
    "forward": ["Forwards.motion", "Forward.motion", "Forwards50.motion"],
    "backward": ["Backwards.motion", "Backward.motion"],
    "turn_left": ["TurnLeft60.motion", "TurnLeft40.motion", "TurnLeft.motion"],
    "turn_right": ["TurnRight60.motion", "TurnRight40.motion", "TurnRight.motion"],
    "side_left": ["SideStepLeft.motion"],
    "side_right": ["SideStepRight.motion"],
}

# Relative path of NAO's motion folder inside a Webots installation.
_NAO_MOTIONS_REL = os.path.join(
    "projects", "robots", "softbank", "nao", "motions"
)


def default_motion_search_dirs(extra: Optional[List[str]] = None) -> List[str]:
    """Candidate directories that hold NAO ``.motion`` files.

    The most reliable entry is ``$WEBOTS_HOME`` (Webots sets it for controller
    processes); the rest cover common install locations on Linux/macOS so the
    discovery still works if the env var is missing. ``extra`` directories (e.g.
    a repo-local ``motions/`` fallback) are searched first.
    """
    dirs: List[str] = list(extra or [])

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


def find_motion_files(search_dirs: List[str]) -> Dict[str, str]:
    """Map each logical action to the first existing motion file found.

    Returns e.g. ``{"forward": "/.../Forwards.motion", "turn_left": "..."}``.
    Actions whose files are not found are simply omitted.
    """
    found: Dict[str, str] = {}
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


def select_action(
    gait: Optional[Dict[str, object]],
    *,
    conf_min: float = 0.6,
    turn_threshold: float = 0.4,
) -> Optional[str]:
    """Choose the walk action for a gait command, or None to stand still.

    The human's detected gait is mapped to locomotion intent:
      * not marching / unconfident / stopped  -> None (stand & resume imitation)
      * strong turn cue                        -> "turn_left" / "turn_right"
      * otherwise marching                     -> "forward" (the robot walks
                                                   forward, so its coordinates
                                                   actually change)

    ``turn`` sign convention: > 0 leans/rotates to the human's left in the
    mirrored selfie view -> turn_left. Flip ``turn_threshold`` sign handling in
    one place if a given camera setup reads reversed.
    """
    if not gait:
        return None
    if str(gait.get("state", "idle")) != "march":
        return None
    if float(gait.get("cadence_hz", 0.0) or 0.0) <= 0.0:
        return None
    if float(gait.get("conf", 0.0) or 0.0) < conf_min:
        return None

    turn = float(gait.get("turn", 0.0) or 0.0)
    if turn >= turn_threshold:
        return "turn_left"
    if turn <= -turn_threshold:
        return "turn_right"
    return "forward"


# ---------------------------------------------------------------------------
# Yaw servo / locomotion planning
# ---------------------------------------------------------------------------
_TURN_RE = re.compile(r"Turn(Left|Right)(\d+)?", re.IGNORECASE)
# Turn clips whose filename carries no angle (``TurnLeft.motion``) are treated as
# this many degrees. Webots ships numbered clips, so this is only a fallback.
_UNNUMBERED_TURN_DEG = 40.0


def motion_nominal_yaw(path_or_name: Optional[str]) -> float:
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
    turn_start_rad: float = 0.45     # |yaw error| that starts a turn (~26 deg)
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
    action: Optional[str]   # key into the discovered motion files, or None
    reason: str             # human-readable, for the controller log

    @property
    def is_turn(self) -> bool:
        return self.action in ("turn_left", "turn_right")


STAND = LocomotionPlan(None, "stand")


def plan_action(
    *,
    yaw_error_rad: float,
    gait: Optional[Dict[str, object]] = None,
    available: Optional[Dict[str, str]] = None,
    params: Optional[LocomotionParams] = None,
    turning: bool = False,
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

    if math.isfinite(yaw_error_rad):
        gate = p.turn_stop_rad if turning else p.turn_start_rad
        if abs(yaw_error_rad) >= gate:
            action = "turn_left" if yaw_error_rad > 0.0 else "turn_right"
            clip = available.get(action)
            if clip is not None:
                nominal = abs(motion_nominal_yaw(clip))
                if nominal <= 1e-3 or abs(yaw_error_rad) >= p.overshoot_frac * nominal:
                    return LocomotionPlan(
                        action, f"yaw error {math.degrees(yaw_error_rad):+.0f} deg"
                    )

    if gait and "forward" in available:
        marching = str(gait.get("state", "idle")) == "march"
        cadence = float(gait.get("cadence_hz", 0.0) or 0.0)
        conf = float(gait.get("conf", 0.0) or 0.0)
        if marching and cadence >= p.walk_cadence_min_hz and conf >= p.walk_conf_min:
            return LocomotionPlan("forward", f"marching at {cadence:.2f} Hz")

    return STAND


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

    _human_ref: Optional[float] = None
    _robot_ref: Optional[float] = None
    _human: float = 0.0
    _last_seen: Optional[float] = None

    @property
    def latched(self) -> bool:
        return self._human_ref is not None

    def update(self, *, human_yaw: Optional[float], conf: float,
               robot_yaw: float, now_s: float) -> None:
        """Ingest one measurement pair; (re)latch the reference when needed."""
        if human_yaw is None or not math.isfinite(human_yaw) or conf < self.conf_min:
            return
        gap = None if self._last_seen is None else (now_s - self._last_seen)
        if self._human_ref is None or (gap is not None and gap > self.relatch_after_s):
            # First sight, or the subject was gone long enough that their old
            # heading tells us nothing: re-zero instead of chasing a stale error.
            self._human_ref = human_yaw
            self._robot_ref = robot_yaw
        self._human = human_yaw
        self._last_seen = now_s

    def error(self, robot_yaw: float) -> float:
        """Yaw the robot still has to turn; positive = to its own left."""
        if self._human_ref is None or self._robot_ref is None:
            return 0.0
        desired = self._robot_ref + self.sign * (self._human - self._human_ref)
        return wrap_pi(desired - robot_yaw)

    def reset(self) -> None:
        self._human_ref = None
        self._robot_ref = None
        self._human = 0.0
        self._last_seen = None
