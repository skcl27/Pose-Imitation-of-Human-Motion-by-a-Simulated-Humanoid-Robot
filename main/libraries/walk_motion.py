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
#
# ``forward`` lists the SHORT clip, which is the safe fallback: a one-shot clip is
# played to completion because its boundaries are balanced double-support poses,
# so a short clip means a short "stop walking" latency. ``forward_continuous``
# lists the long clip separately, and :func:`select_walk_clip` promotes it over
# the short one when -- and only when -- a gait cycle is actually detected inside
# it, which is what makes it stoppable without being short. It is never planned
# under its own name.
KNOWN_MOTIONS: dict[str, list[str]] = {
    "forward": ["Forwards.motion", "Forward.motion"],
    "forward_continuous": ["Forwards50.motion"],
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


def select_walk_clip(files: dict[str, str]) -> tuple[dict[str, str], str]:
    """Resolve ``forward`` to the best walk clip available, and say why.

    Returns ``(files, note)`` with ``forward_continuous`` consumed: either
    promoted to ``forward`` or dropped. Nothing downstream ever sees it, so
    :func:`plan_action` and the whole locomotion layer keep working in terms of
    one ``forward`` action.

    A long clip beats a short one only if it can be CYCLED. Played one-shot the
    long clip is strictly worse: it commits the robot to 6.76 s and 0.46 m before
    it can be asked to stop. Cycled, it is strictly better: the same stride
    without the start/stop transient between repetitions, and a worst-case stop
    latency of one period plus the deceleration tail, which is shorter than the
    short clip's own length. So the choice is made by asking
    :func:`gait_cycle` -- if it cannot find a cycle (no NumPy, or a Webots release
    whose clip is not periodic), the short clip stays.
    """
    files = dict(files)
    long_clip = files.pop("forward_continuous", None)
    if long_clip is None:
        return files, "no continuous walk clip on disk"
    cycle = gait_cycle(long_clip)
    if cycle is None:
        return files, (f"{os.path.basename(long_clip)} has no detectable gait "
                       f"cycle; keeping the short clip")
    short = files.get("forward")
    files["forward"] = long_clip
    return files, (
        f"walking with {os.path.basename(long_clip)} as a continuous gait "
        f"({cycle.speed_mps:.3f} m/s sustained, stop within "
        f"{cycle.stop_latency_s:.2f}s)"
        + (f" instead of one-shot {os.path.basename(short)}" if short else "")
    )


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


def _keyframe_ms(stamp: str) -> float | None:
    """``00:02:600`` -> 2600.0 ms, or None if it is not a timestamp."""
    parts = stamp.split(":")
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]) * 60000.0 + float(parts[1]) * 1000.0 + float(parts[2])
    except ValueError:
        return None


def motion_poses(path: str | None) -> list[tuple[float, dict[str, float]]]:
    """Every keyframe of a clip as ``(seconds, {joint: rad})``.

    The clip file is the only description of what playback is about to do, so it
    is also the only place to answer questions like "when is this clip in double
    support?" -- which is what makes it possible to stop a long clip early
    instead of riding it to the end (see ``balance.safe_exit_times``).
    """
    joints = motion_joints(path)
    if not joints or path is None:
        return []
    out: list[tuple[float, dict[str, float]]] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2 + len(joints):
                    continue
                ms = _keyframe_ms(parts[0])
                if ms is None:
                    continue
                try:
                    values = [float(v) for v in parts[2:2 + len(joints)]]
                except ValueError:
                    continue
                out.append((ms / 1000.0, dict(zip(joints, values, strict=False))))
    except OSError:
        return []
    return out


def motion_first_pose(path: str | None) -> dict[str, float]:
    """The joint angles of a clip's FIRST keyframe, ``{joint: rad}``.

    A Webots ``.motion`` file is a header naming the joints it drives and then
    one line per keyframe::

        #WEBOTS_MOTION,V1.0,LHipYawPitch,LHipRoll,LHipPitch,...
        00:00:000,Pose1,0,0.027,-0.505,1.042,-0.537,-0.027,...

    This matters because playback commands that first keyframe on its very first
    step, with the velocity caps already lifted. Cyberbotics' walk and turn clips
    all begin in a deep, sole-flat crouch -- Forwards.motion opens at
    ``LHipPitch -0.505, LKneePitch +1.042, LAnklePitch -0.537`` (they sum to zero,
    so the torso stays vertical and the soles flat), which is a squat of about
    0.51 rad. The controller stands at 0.10. Handing over without getting there
    first asks the knees for 0.84 rad in one 20 ms step, and the robot squats out
    from under itself.

    So the controller reads this, ramps the legs to it, and only then plays the
    clip. Returns {} if the file is missing or unparseable.
    """
    poses = motion_poses(path)
    return dict(poses[0][1]) if poses else {}


def motion_pose_at(path: str | None, seconds: float) -> dict[str, float]:
    """The joint angles of the last keyframe at or before ``seconds``.

    Playback can be started part-way into a clip (see :func:`gait_cycle`), and
    when it is, the pose to ramp to first is the one at that offset -- not the
    clip's first keyframe. Falls back to the first keyframe for a negative or
    unreachable offset, and {} for an unreadable clip.
    """
    poses = motion_poses(path)
    if not poses:
        return {}
    chosen = poses[0][1]
    for time_s, angles in poses:
        if time_s > seconds + 1e-9:
            break
        chosen = angles
    return dict(chosen)


# ---------------------------------------------------------------------------
# Cyclic gait: turning a one-shot clip into a continuous walk
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaitCycle:
    """A periodic segment inside a motion clip, and how to enter and leave it.

    Cyberbotics' walk clips are one-shot: a squat, an acceleration, some strides,
    a deceleration, a stand. Played that way the transient dominates -- 49% of
    Forwards.motion's 2.60 s is start/stop, during which the settle actually pulls
    the torso 17 mm BACKWARD -- so the robot's average speed is a fraction of the
    0.18 m/s it reaches mid-stride, and asking it to walk further means replaying
    the whole transient again. Measured on the live robot: 0.036 m/s.

    But the middle of a long walk clip is a genuine limit cycle, and exactly so:
    in Forwards50.motion ``max|q(2.84 s) - q(1.80 s)| = 0.0000 rad`` across all 12
    leg joints, and the periodicity holds to that precision from t=1.80 s to
    t=5.32 s. That is not an approximation to work around -- it means the segment
    can be REWOUND with ``Motion.setTime()`` and the joints will not move at the
    seam, so one clip becomes a gait generator that runs for as long as the human
    keeps walking, at the speed the stride itself is worth (0.089 m/s here by
    forward kinematics) rather than the speed the transient averages down to.

    Four times define the schedule:

    ``enter_s``
        Where to start playback. The clip's opening squat translates the robot by
        under ``static_mm`` and the controller's own prepare-ramp does that squat
        better (rate-limited, under balance supervision), so it is skipped:
        0.72 s of Forwards50.motion's 6.76 s, worth 0.9 mm.
    ``loop_start_s`` / ``loop_end_s``
        The periodic window. On reaching ``loop_end_s`` playback rewinds by
        ``period_s`` and the stride repeats.
    ``exit_from_s`` -> ``exit_to_s``
        How to stop. NOT by freezing mid-stride -- the body keeps its momentum
        and walks itself over (see ``balance.safe_exit_times``) -- but by jumping
        once, at the one phase of the cycle where it is nearly free
        (``exit_cost_rad``: 0.0010 rad, i.e. 0.05 rad/s over one control step),
        into the clip's own deceleration, and letting Cyberbotics' own
        feet-together settle bring the robot to rest. Stop latency is therefore at
        most ``period_s + tail_s`` = 1.04 + 1.44 = 2.48 s, and 1.96 s on
        average -- shorter than the 2.60 s the short clip commits to for a single
        0.095 m step.

    Note what this does NOT claim. The stride is the same stride either clip
    walks: both take a 0.051 m half-step and both peak at 0.18 m/s. Cycling does
    not make the robot take faster steps; it removes the start/stop transient
    between them, which is where the time was going.
    """
    enter_s: float
    loop_start_s: float
    loop_end_s: float
    exit_from_s: float
    exit_to_s: float
    exit_cost_rad: float
    advance_m: float
    duration_s: float

    @property
    def period_s(self) -> float:
        return self.loop_end_s - self.loop_start_s

    @property
    def tail_s(self) -> float:
        return self.duration_s - self.exit_to_s

    @property
    def speed_mps(self) -> float:
        """Sustained forward speed while cycling, from the clip's own kinematics."""
        return self.advance_m / self.period_s if self.period_s > 0.0 else 0.0

    @property
    def stop_latency_s(self) -> float:
        """Worst case from "stop walking" to standing still."""
        return self.period_s + self.tail_s


# A periodic segment that does not translate the robot is not a gait cycle. Every
# clip has SOME periodicity (a turn rotates through repeated steps, a side-step
# shuffles), and looping those would spin or drift the robot indefinitely with no
# way to reason about where it ends up -- turning is closed-loop on the heading
# and needs discrete, countable clips. 0.02 m per cycle admits Forwards50's
# 0.092 m and rejects TurnLeft180's 0.001 m and SideStepLeft's -0.000 m.
GAIT_CYCLE_MIN_ADVANCE_M = 0.02


def gait_cycle(path: str | None,
               loop_tol: float = 1e-6,
               exit_tol: float = 0.02,
               static_mm: float = 2.0,
               min_period_s: float = 0.4,
               min_advance_m: float = GAIT_CYCLE_MIN_ADVANCE_M) -> GaitCycle | None:
    """Find the cyclic gait inside a walk clip, or None if it has none.

    Derived from the clip's own keyframes every time rather than tabulated, so it
    cannot drift from the files Webots actually ships, and so a clip that is not
    cyclic is *detected* as not cyclic instead of being looped on faith.

    ``loop_tol`` is deliberately near-exact (1 urad). A loop seam is a
    teleport in joint space executed in a single 20 ms step with the velocity
    caps lifted, so the only acceptable seam is one that commands no motion at
    all; a "close enough" seam of 0.1 rad would be a 5 rad/s jolt every stride.
    Forwards.motion's best available seam is 0.107 rad, which is exactly why it
    cannot be cycled and this returns None for it.

    Needs NumPy (for the forward-kinematic odometry). Returns None without it,
    and the caller falls back to one-shot playback.
    """
    poses = motion_poses(path)
    if len(poses) < 10:
        return None
    try:
        from balance import NaoCoMModel, clip_torso_speeds
    except Exception:  # noqa: BLE001 - no NumPy: no odometry, so no cycle
        return None

    times = [t for t, _ in poses]
    angles = [a for _, a in poses]
    names = list(angles[0])
    count = len(poses)
    step = times[1] - times[0]
    if step <= 0.0:
        return None

    def distance(i: int, k: int) -> float:
        return max((abs(angles[i][j] - angles[k][j]) for j in names), default=0.0)

    # The SMALLEST exact period, at its EARLIEST occurrence. Smallest because if
    # p is a period so is 2p, and a tighter loop means finer control over when to
    # leave; earliest because playback has to reach the window before it can
    # start cycling, and everything before it is transient.
    window: tuple[int, int] | None = None
    for period in range(max(1, int(round(min_period_s / step))), count // 2):
        for start in range(0, count - period):
            if distance(start, start + period) <= loop_tol:
                window = (start, period)
                break
        if window is not None:
            break
    if window is None:
        return None
    first, period = window

    # How far the cycle carries the robot, and where the cyclic region ends.
    model = NaoCoMModel()
    speeds = clip_torso_speeds(poses, model)
    advance = 0.0
    for index in range(first, first + period):
        advance += speeds[index] * step
    if advance < min_advance_m:
        return None
    last = max(i for i in range(count - period)
               if distance(i, i + period) <= loop_tol)
    # The first keyframe past the last full cycle. It is a legitimate jump TARGET
    # even though it is the boundary: the loop only ever rewinds at loop_end, so
    # landing beyond loop_end and stopping the rewinds means playing forward from
    # here -- straight into the clip's deceleration. It is also usually the
    # CHEAPEST target, because a clip that decelerates out of its own gait leaves
    # the cycle at a phase the cycle itself passes through.
    tail = last + period

    # Enter as late as the clip allows without skipping any translation: the
    # prepare-ramp will put the robot in that pose under its own rate limits.
    enter = 0
    for index in range(0, first + 1):
        if abs(sum(speeds[:index]) * step) * 1000.0 <= static_mm:
            enter = index
        else:
            break

    # Leave through the clip's own deceleration, at whichever phase of the cycle
    # is cheapest to jump from. Restricted to the FIRST window so the phase is
    # reached once per period no matter how long the robot has been walking.
    best: tuple[float, int, int] | None = None
    for a in range(first, first + period):
        for b in range(tail, count):
            cost = distance(a, b)
            if best is None or cost < best[0]:
                best = (cost, a, b)
    if best is None or best[0] > exit_tol:
        return None

    return GaitCycle(
        enter_s=float(times[enter]),
        loop_start_s=float(times[first]),
        loop_end_s=float(times[first + period]),
        exit_from_s=float(times[best[1]]),
        exit_to_s=float(times[best[2]]),
        exit_cost_rad=float(best[0]),
        advance_m=float(advance),
        duration_s=float(times[-1]),
    )


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
    # Must be STRICTLY greater than 0.5 or the turn cannot converge: firing at
    # |error| = f * nominal leaves |error - nominal| = |1 - 1/f| * |error|, which
    # is >= |error| for every f <= 0.5. At exactly 0.5 a clip turns the robot from
    # +e to -e forever, which is a limit cycle dressed as a controller. At 0.65
    # each turn cuts the error to at most 0.54 of what it was, so a 180 deg
    # request converges inside the deadband in three clips.
    overshoot_frac: float = 0.65
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


def _smallest_servable_turn(yaw_error_rad: float,
                            available: dict[str, str],
                            p: "LocomotionParams") -> float | None:
    """The smallest |yaw error| at which SOME turn clip for this direction fits.

    None when no clip exists for the direction at all (in which case the caller's
    own gate is left alone -- there is nothing to protect against, the loop below
    will simply find nothing).
    """
    keys = TURN_ACTIONS["left" if yaw_error_rad > 0.0 else "right"]
    thresholds = []
    for key in keys:
        clip = available.get(key)
        if clip is None:
            continue
        nominal = abs(motion_nominal_yaw(clip))
        # A clip whose nominal turn is unknown fits at any error (see `fits`).
        thresholds.append(0.0 if nominal <= 1e-3 else p.overshoot_frac * nominal)
    return min(thresholds) if thresholds else None


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
        # Never REQUEST a turn no clip can serve. The gate above says "this much
        # heading error is worth correcting"; ``fits`` below says "this clip is
        # big enough to be worth firing at that error". Those two were allowed to
        # disagree, and the gap between them was a trap: turn_start_rad 0.35 rad
        # admits the request at 20 deg, but with overshoot_frac 0.65 the smallest
        # clip on disk (TurnRight40, 0.698 rad) does not fit until 0.454 rad =
        # 26 deg. In that 6 deg band the controller announced a turn, spent 0.7 s
        # ramping the legs down into the clip's opening crouch -- and then found
        # `best` still None, planned nothing, and stood back up. 29 prepares in
        # the recorded session ramped and never played, 14 of them turn_right,
        # and 13.8% of walking frames sat in that band. It reads exactly like the
        # "tries to step and falls" the robot was reported for.
        #
        # So the entry gate is raised to whatever the smallest available clip can
        # actually be fired at. Consequence, stated plainly: the robot tolerates
        # up to 26 deg of heading error instead of 20 before it turns at all. That
        # is the honest cost of owning only a 40 deg clip, and it is much cheaper
        # than squatting and standing up again while never turning.
        servable = _smallest_servable_turn(yaw_error_rad, available, p)
        if servable is not None:
            gate = max(gate, servable)
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
