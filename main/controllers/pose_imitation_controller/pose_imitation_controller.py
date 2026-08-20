"""
Real-time Webots NAO pose imitation controller.

Receives human-pose frames via UDP from the Python pipeline (``src/``) and drives
the simulated NAO humanoid in real time: arms, head, legs, and genuine
locomotion (the robot's world coordinates change when you walk, and it turns to
face where you face).

Architecture: this file is the Webots glue and the **arbiter**. All the maths
lives in unit-tested, Webots-free libraries under ``main/libraries/``:

    UDP frame ──► arms + head        : nao_retarget.retarget_upper_body
                └► legs, one of:
                     1. locomotion   : walk_motion.plan_action  + Webots Motion
                                       clips  (real translation / turning)
                     2. march engine  : gait.GaitEngine
                                       (in-place, when no clips exist on disk)
                     3. pose imitation: lower_body.LowerBodyController
                                       (squat, single-leg lift, weight transfer)
                     4. stand         : balance.BalanceController only

Exactly ONE of those four commands the legs on any given step -- two of them at
once means the layers fight and the robot falls, which is the single most common
way a humanoid imitation controller breaks.

A fifth, always-on layer runs underneath the arbiter every tick regardless of
which of the four is active: a continuous tilt-risk EMA (_update_tilt_risk)
that makes _settled() progressively stricter about starting the next
locomotion clip after a wobble. It never touches an in-progress clip -- see
the MotionPlayer docstring for why -- only whether the *next* one is allowed
to start, so a rough patch degrades to march-in-place/pose-imitation (both
fall-safe by design) for a while instead of only reacting once a clip is
already failing.

Protocol (UDP, port 8765, JSON):
    {
      "timestamp_s": 1234567890.123,
      "frame_index": 45,
      "joint_angles_rad": {"LShoulderPitch": 0.5, ...},   # fallback
      "keypoints": {"left_shoulder": [x, y, z, visibility], ...},
      "gait": {"state": "march", "cadence_hz": 0.9, "body_yaw_rad": 0.4, ...}
    }
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
import sys
import time
from typing import Dict, List, Optional

try:
    from controller import Motion, Robot  # type: ignore
except ImportError:
    print("Error: Webots controller module not found. Run this only in Webots.")
    sys.exit(1)

# Make the shared library importable regardless of Webots' working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libraries"))
from pose_control_utils import JointTrajectoryLogger, NaoPoseDriver  # noqa: E402
from walk_motion import (  # noqa: E402
    LocomotionParams,
    YawServo,
    default_motion_search_dirs,
    find_motion_files,
    plan_action,
)

# ===========================================================================
# Configuration
# ===========================================================================
UDP_HOST = "127.0.0.1"
UDP_PORT = 8765
SOCKET_RCVBUF = 1 << 16

# --- What drives the legs ---------------------------------------------------
# "auto"   (recommended) the full stack: Webots' pre-balanced NAO walk/turn clips
#          for REAL locomotion, the in-place march engine when no clips are found
#          on disk, and per-leg pose imitation (squat / single-leg lift with a
#          model-verified weight transfer) the rest of the time.
# "pose"   per-leg pose imitation only -- no locomotion, no marching.
# "engine" march engine + pose imitation, never the motion clips (use this to
#          keep the robot on the spot).
# "off"    legs held in the standing posture; upper body only.
LEG_CONTROL = "auto"

DRIVE_HEAD = True         # head yaw/pitch follow the human head
SWAP_SIDES = False        # True = mirror-image mapping (robot's left <-> your right)
SMOOTHING_ALPHA = 0.4     # EMA factor for arm/head targets (0..1, higher = snappier)
VELOCITY_SCALE = 0.5      # fraction of each joint's hardware max velocity
LEG_VELOCITY_FACTOR = 0.5 # extra slow-down on leg joints when merely posturing
STALE_AFTER_S = 0.5       # hold pose if no command for this long

# --- Balance ---------------------------------------------------------------
# Model-based CoM feedback recovers the depth/balance information a 2D camera
# cannot give: forward kinematics + NAO link masses estimate the centre of mass
# each step, the InertialUnit supplies the gravity direction, and a
# Fibonacci-spiral search nudges ankles/hips to keep the CoM over the feet.
# Runs in a normal controller (no Supervisor). See main/libraries/balance.py.
ENABLE_BALANCE = True
INERTIAL_UNIT_NAME = "inertial unit"

# --- March engine ----------------------------------------------------------
# Tier A ("march") is a double-support weight-shift march: it never fully unloads
# a foot, so the symmetric balance loop stays valid and it cannot fall by design.
# Tier B ("step") is experimental single-support stepping. Real stepping is now
# better served by LEG_CONTROL="auto" (motion clips) or the pose-imitation
# sequencer, so leave this at "march".
WALK_TIER = "march"
GAIT_SMOOTHING_ALPHA = 0.7       # snappier than the arm EMA so gait/step survives
GAIT_LEG_VELOCITY_FACTOR = 0.85  # raised leg velocity while walking or stepping

# --- Locomotion (Webots .motion clips) -------------------------------------
# Turning is a closed loop on the InertialUnit heading, so it converges despite
# the clips being coarse: see walk_motion.YawServo / plan_action.
LOCOMOTION = LocomotionParams()
# Flip if the robot turns the wrong way for your camera setup. With the pipeline's
# default mirrored (selfie) preview the robot mirrors the on-screen figure, which
# is consistent with how the arms are mapped.
TURN_SIGN = 1.0
# Extra dirs searched (first) for NAO .motion files, in addition to $WEBOTS_HOME
# and the common install locations. A repo-local motions/ folder can go here.
MOTION_SEARCH_DIRS_EXTRA = [os.path.join(os.path.dirname(__file__), "motions")]

# --- Safety ----------------------------------------------------------------
# Emergency abort of a running motion clip. The gyro term is a lead compensator:
# a fall is visible in the tilt *rate* well before the tilt itself crosses a
# threshold, so predicting a quarter second ahead buys time to stop the clip and
# hand the body back to the balance loop.
TILT_ABORT_RAD = 0.40
TILT_RATE_LEAD_S = 0.25

# HARD WATCHDOG on motion playback. While a clip runs it owns the whole body, so
# per-joint commanding is suspended -- which means anything that stops the clip
# from ever reporting "over" freezes the ENTIRE robot, not just the legs. Webots'
# walk clips are a few seconds long, so any suspension beyond this is a bug, not
# a long clip: we take the body back and stop trusting clips.
MOTION_WATCHDOG_S = 8.0
# Consecutive locomotion attempts that end badly (watchdog trip, tilt abort, or
# the robot still tipped when the clip finishes) before clips are abandoned for
# this session. Falling over repeatedly is worse than never walking.
MOTION_MAX_FAILURES = 3
# A clip is only started from a settled, upright robot: starting one mid-wobble
# is how a walk turns into a fall. This ceiling is not fixed: it shrinks
# continuously (see _tilt_risk / _settled) toward MOTION_START_MIN_TILT_RAD as
# recent tilt trends upward, so the controller keeps declining to start another
# clip for a while after a wobble instead of only reacting once mid-clip.
MOTION_START_MAX_TILT_RAD = 0.15
MOTION_START_MIN_TILT_RAD = 0.05
# EMA time constant for the continuous tilt-risk signal, updated every control
# step (independent of which leg-control layer is active) from the same
# predicted-tilt formula the hard abort below already trusts.
TILT_RISK_TAU_S = 1.5

# The control loop must survive a bad step. An exception used to end run(),
# which then called driver.stop() and set every motor velocity to zero -- a
# permanently dead robot from one transient error. Now each step is contained:
# we log, force the body back under control, and carry on.
MAX_CONSECUTIVE_ERRORS = 20

# --- Sensors ---------------------------------------------------------------
GYRO_NAME = "gyro"
ACCELEROMETER_NAME = "accelerometer"
# Webots' Nao.proto exposes one 3-axis force sensor per foot ("LFsr"/"RFsr");
# the other names are fallbacks for older/other NAO protos. Any that resolve are
# summed per foot; the rest are ignored and the CoM model gates stepping alone.
FSR_DEVICES = {
    "L": ["LFsr", "LFootFSR", "LFoot/Fsr", "force_sensor_left"],
    "R": ["RFsr", "RFootFSR", "RFoot/Fsr", "force_sensor_right"],
}

# --- Logging ---------------------------------------------------------------
# Commanded vs. achieved joint angles to <project>/logs/ for offline
# imitation-fidelity metrics (PRD FR-7 / US-3).
ENABLE_TRAJECTORY_LOG = True
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs"))
STATUS_EVERY = 100  # frames

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PoseController")


# ===========================================================================
# Webots Motion playback
# ===========================================================================
class MotionPlayer:
    """Plays Webots' pre-balanced NAO ``.motion`` clips, one at a time.

    Why clips at all: an online gait good enough to translate a free-standing
    NAO across the floor is a research project in itself, while Cyberbotics ship
    walk/turn clips that are already balanced for this exact robot. Playing them
    is what makes the robot *genuinely* move -- its world coordinates change --
    rather than marching on the spot.

    Clips are played **to completion and then optionally replayed**, never looped
    and never cut short. A clip boundary is a balanced double-support pose, so it
    is the only place where handing control back is safe; that also makes clip
    length the granularity of "stop walking", which is why the short
    ``Forwards.motion`` is preferred over ``Forwards50.motion``. :meth:`abort`
    exists for the one case worth breaking that rule: an incipient fall.
    """

    def __init__(self, files: Dict[str, str],
                 log: Optional[object] = None) -> None:
        self._files = dict(files)
        self._cache: Dict[str, object] = {}
        self._log = log or (lambda *_a, **_k: None)
        self.action: Optional[str] = None
        self._motion: Optional[object] = None

    @property
    def available(self) -> Dict[str, str]:
        return dict(self._files)

    @property
    def active(self) -> bool:
        return self._motion is not None

    def _load(self, action: str) -> Optional[object]:
        if action in self._cache:
            return self._cache[action]
        path = self._files.get(action)
        if path is None:
            return None
        try:
            motion = Motion(path)
            if not motion.isValid():
                raise RuntimeError("clip rejected by Webots")
        except Exception as exc:  # noqa: BLE001
            self._log("Motion '%s' unusable (%s); dropping it", action, exc)
            self._files.pop(action, None)
            return None
        self._cache[action] = motion
        return motion

    def start(self, action: str) -> bool:
        """Begin ``action``; returns False if its clip is missing or invalid."""
        motion = self._load(action)
        if motion is None:
            return False
        try:
            motion.setLoop(False)
            # A clip that already ran must be rewound, or play() resumes at its
            # end and returns immediately. stop() + setTime(0) covers both the
            # "interrupted" and the "finished" case.
            motion.stop()
            try:
                motion.setTime(0)
            except Exception:  # noqa: BLE001 - older API without setTime
                pass
            if not motion.play():
                return False
        except Exception as exc:  # noqa: BLE001
            self._log("Could not play motion '%s': %s", action, exc)
            return False
        self.action = action
        self._motion = motion
        return True

    def poll(self) -> bool:
        """True while the clip is still running; clears itself when it is over."""
        if self._motion is None:
            return False
        try:
            over = bool(self._motion.isOver())
        except Exception:  # noqa: BLE001
            over = True
        if over:
            self._clear()
            return False
        return True

    def duration_s(self) -> Optional[float]:
        """Clip length in seconds, or None if Webots will not tell us."""
        if self._motion is None:
            return None
        try:
            ms = float(self._motion.getDuration())
        except Exception:  # noqa: BLE001
            return None
        return ms / 1000.0 if math.isfinite(ms) and ms > 0.0 else None

    def drop(self, action: Optional[str] = None) -> None:
        """Stop using ``action`` (or every clip) for the rest of the session."""
        target = action or self.action
        self.abort()
        if target is None:
            self._files.clear()
            self._cache.clear()
        else:
            self._files.pop(target, None)
            self._cache.pop(target, None)

    def abort(self) -> None:
        """Stop mid-clip. Only for a safety abort -- see the class docstring."""
        if self._motion is None:
            return
        try:
            self._motion.stop()
        except Exception:  # noqa: BLE001
            pass
        self._clear()

    def _clear(self) -> None:
        self.action = None
        self._motion = None


# ===========================================================================
# Controller
# ===========================================================================
class PoseImitationController:
    """Webots glue and lower-body arbiter around :class:`NaoPoseDriver`."""

    def __init__(self) -> None:
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        logger.info("Initializing NAO pose controller (timestep: %dms)", self.timestep)
        if self.timestep > 24:
            logger.warning(
                "WorldInfo.basicTimeStep is %d ms. NAO leg control and the walk "
                "clips need <= 20 ms; expect poor balance.", self.timestep
            )

        self.leg_control = LEG_CONTROL if LEG_CONTROL in (
            "auto", "pose", "engine", "off"
        ) else "auto"
        drive_legs = self.leg_control in ("auto", "pose", "engine")
        enable_walk = self.leg_control in ("auto", "engine")

        self.driver = NaoPoseDriver(
            self.robot,
            drive_legs=drive_legs,
            drive_head=DRIVE_HEAD,
            swap_sides=SWAP_SIDES,
            smoothing_alpha=SMOOTHING_ALPHA,
            velocity_scale=VELOCITY_SCALE,
            leg_velocity_factor=LEG_VELOCITY_FACTOR,
            stale_after_s=STALE_AFTER_S,
            enable_balance=ENABLE_BALANCE,
            enable_walk=enable_walk,
            walk_tier=WALK_TIER,
            gait_smoothing_alpha=GAIT_SMOOTHING_ALPHA,
            gait_leg_velocity_factor=GAIT_LEG_VELOCITY_FACTOR,
            logger=logger.info,
        )
        self._init_imu()
        self._init_walk_sensors()
        self._init_locomotion()
        self._init_socket()

        self.trajectory_log = None
        if ENABLE_TRAJECTORY_LOG:
            self.trajectory_log = JointTrajectoryLogger(
                LOG_DIR, self.driver.logged_joints, logger=logger.info
            )

        self.gait_cmd: Optional[Dict] = None
        self.leg_mode = "stand"
        self.frame_count = 0
        self._last_log_time = time.time()
        # True while a multi-clip rotation is still converging. It survives clip
        # boundaries on purpose: that is what lets plan_action use its tighter
        # gate to finish a turn instead of stalling one clip short.
        self._turning = False
        # Motion-playback watchdog state. Suspension hands the WHOLE body to a
        # clip, so it must always be bounded in time and in failure count.
        self._motion_started_at: Optional[float] = None
        self._motion_deadline: Optional[float] = None
        self._motion_failures = 0
        self._errors = 0
        # Continuous tilt-risk EMA (see _update_tilt_risk / _settled): runs every
        # tick regardless of which leg-control layer is active.
        self._tilt_risk = 0.0
        self._last_risk_update: Optional[float] = None
        self._report_startup()

    def _report_startup(self) -> None:
        """Print one block saying exactly what will and will not work.

        This exists because every "the robot does not move" report so far has had
        a cause that was visible at startup -- a missing device, a missing NumPy,
        no motion clips, a coarse timestep -- but was buried in the log. Saying
        it plainly up front turns a debugging session into a glance.
        """
        d = self.driver
        n_legs = sum(1 for n in d.motors if n.endswith(
            ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch", "AnklePitch", "AnkleRoll")))
        logger.info("=" * 68)
        logger.info("NAO pose imitation controller ready")
        logger.info("  timestep          : %d ms%s", self.timestep,
                    "" if self.timestep <= 20 else "   <-- TOO COARSE, use 20 ms")
        logger.info("  motors / sensors  : %d / %d  (%d leg joints)",
                    len(d.motors), len(d.sensors), n_legs)
        logger.info("  leg control       : %s", self.leg_control)
        logger.info("  arms + head       : ON")
        logger.info("  leg pose imitation: %s",
                    "ON (squat, single-leg lift)" if d.lower_body is not None
                    else "OFF  <-- legs will only hold a posture")
        logger.info("  CoM balance       : %s",
                    "ON" if d.balance is not None
                    else "OFF  <-- needs NumPy in Webots' Python")
        logger.info("  march engine      : %s",
                    "ON" if d.gait_engine is not None else "OFF")
        if self.leg_control == "auto":
            clips = sorted(self.motion.available)
            logger.info("  locomotion clips  : %s",
                        ", ".join(clips) if clips
                        else "NONE FOUND  <-- will march in place, not walk")
        logger.info("  heading feedback  : %s",
                    "ON (InertialUnit)" if self.imu is not None
                    else "OFF  <-- turning disabled")
        logger.info("  foot force sensors: %d",
                    len(self.fsr["L"]) + len(self.fsr["R"]))
        if d.lower_body is None or d.balance is None:
            logger.warning(
                "A layer is OFF above. If that was not intended, check that "
                "Webots' Python interpreter has NumPy: Tools > Preferences > "
                "Python command."
            )
        logger.info("Waiting for pose commands on %s:%d ...", UDP_HOST, UDP_PORT)
        logger.info("=" * 68)

    # ---------------------------------------------------------------- devices
    def _init_imu(self) -> None:
        """Enable the InertialUnit: gravity direction for balance AND the robot's
        true heading, which the turn servo closes its loop on."""
        self.imu = None
        imu = self.robot.getDevice(INERTIAL_UNIT_NAME)
        if imu is None:
            logger.warning(
                "InertialUnit '%s' not found; balance runs CoM-only and turning "
                "is disabled (no heading feedback).", INERTIAL_UNIT_NAME
            )
            return
        try:
            imu.enable(self.timestep)
            self.imu = imu
            logger.info("InertialUnit enabled (balance + heading feedback)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enable InertialUnit: %s", exc)

    def _imu_rpy(self) -> tuple:
        """(roll, pitch, yaw) of the torso in rad; (0, 0, 0) if unavailable."""
        if self.imu is None:
            return (0.0, 0.0, 0.0)
        try:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
        except Exception:  # noqa: BLE001
            return (0.0, 0.0, 0.0)
        if not all(math.isfinite(v) for v in (roll, pitch, yaw)):
            return (0.0, 0.0, 0.0)
        return (roll, pitch, yaw)

    def _init_walk_sensors(self) -> None:
        """Enable the gyro/accelerometer and any foot force sensors.

        All best-effort and NaN-guarded: the march tier needs none of them, and
        the stepping gate falls back to the CoM model alone when they are absent.
        """
        self.gyro = None
        self.fsr: Dict[str, List[object]] = {"L": [], "R": []}
        for name in (GYRO_NAME, ACCELEROMETER_NAME):
            dev = self.robot.getDevice(name)
            if dev is None:
                continue
            try:
                dev.enable(self.timestep)
            except Exception:  # noqa: BLE001
                continue
            if name == GYRO_NAME:
                self.gyro = dev
        for side, names in FSR_DEVICES.items():
            for name in names:
                dev = self.robot.getDevice(name)
                if dev is None:
                    continue
                try:
                    dev.enable(self.timestep)
                except Exception:  # noqa: BLE001
                    continue
                self.fsr[side].append(dev)
        n_fsr = len(self.fsr["L"]) + len(self.fsr["R"])
        logger.info("Sensors: gyro=%s, foot-force sensors=%d",
                    self.gyro is not None, n_fsr)

    def _read_fsr(self) -> Optional[Dict[str, float]]:
        """Per-foot load ``{"L": n, "R": n}`` from the FSRs, or None.

        NAO's foot sensors are 3-axis ("force-3d") TouchSensors, so the value
        comes from ``getValues()``, not ``getValue()`` -- reading them as scalars
        is why an earlier version silently got no load information and the
        stepping gate never saw a weight transfer. Both APIs are handled so this
        also works with 1-axis protos.
        """
        if not self.fsr["L"] and not self.fsr["R"]:
            return None
        out: Dict[str, float] = {}
        for side in ("L", "R"):
            total = 0.0
            for dev in self.fsr[side]:
                total += _sensor_magnitude(dev)
            out[side] = total
        return out

    def _tilt_rate(self) -> tuple:
        """(roll_rate, pitch_rate) in rad/s from the gyro; (0, 0) without one."""
        if self.gyro is None:
            return (0.0, 0.0)
        try:
            values = self.gyro.getValues()
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)
        if values is None or len(values) < 2:
            return (0.0, 0.0)
        rates = [float(v) if math.isfinite(float(v)) else 0.0 for v in values[:2]]
        return (rates[0], rates[1])

    def _init_locomotion(self) -> None:
        """Discover the walk/turn clips and build the yaw servo."""
        self.motion = MotionPlayer({}, log=logger.warning)
        self.yaw_servo = YawServo(sign=TURN_SIGN)
        if self.leg_control != "auto":
            logger.info("Locomotion clips disabled (LEG_CONTROL=%s)", self.leg_control)
            return
        dirs = default_motion_search_dirs(extra=MOTION_SEARCH_DIRS_EXTRA)
        files = find_motion_files(dirs)
        self.motion = MotionPlayer(files, log=logger.warning)
        if files:
            logger.info("Locomotion clips found: %s", ", ".join(sorted(files)))
        else:
            logger.warning(
                "No NAO .motion files found (searched %d dirs, e.g. %s). The robot "
                "will march in place instead of translating; set $WEBOTS_HOME or "
                "drop clips in %s to enable real locomotion.",
                len(dirs), dirs[0] if dirs else "-", MOTION_SEARCH_DIRS_EXTRA[0],
            )

    def _init_socket(self) -> None:
        logger.info("Opening UDP socket on %s:%d ...", UDP_HOST, UDP_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_RCVBUF)
        self.sock.bind((UDP_HOST, UDP_PORT))
        self.sock.setblocking(False)
        logger.info("UDP socket ready")

    # ------------------------------------------------------------------- comms
    def _drain_latest_command(self) -> Optional[Dict]:
        """Return the most recent pose command, discarding any backlog.

        UDP can queue several frames between simulation steps. We only care
        about the freshest pose, so we drain the buffer and keep the last one
        (keeps end-to-end latency low -- PRD NFR-1).
        """
        latest: Optional[Dict] = None
        while True:
            try:
                data, _ = self.sock.recvfrom(SOCKET_RCVBUF)
            except (BlockingIOError, OSError):
                break
            try:
                latest = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return latest

    # ---------------------------------------------------------------- arbiter
    def _update_yaw_servo(self, now: float, robot_yaw: float) -> None:
        gait = self.gait_cmd or {}
        yaw = gait.get("body_yaw_rad")
        if yaw is None:
            return
        try:
            human_yaw = float(yaw)
        except (TypeError, ValueError):
            return
        self.yaw_servo.update(
            human_yaw=human_yaw,
            conf=float(gait.get("yaw_conf", gait.get("conf", 0.0)) or 0.0),
            robot_yaw=robot_yaw,
            now_s=now,
        )

    def _predicted_tilt_rad(self, roll: float, pitch: float) -> float:
        """Worst-axis tilt magnitude, predicted a short time ahead by the gyro.

        Shared by the hard mid-clip abort (_falling) and the continuous
        pre-clip risk signal (_update_tilt_risk) so there is one formula, not
        two definitions of "how tipped over are we" drifting apart.
        """
        d_roll, d_pitch = self._tilt_rate()
        roll_pred = roll + TILT_RATE_LEAD_S * d_roll
        pitch_pred = pitch + TILT_RATE_LEAD_S * d_pitch
        return max(abs(roll_pred), abs(pitch_pred))

    def _falling(self, roll: float, pitch: float) -> bool:
        """Tilt (predicted a short time ahead by the gyro) past the abort limit."""
        return self._predicted_tilt_rad(roll, pitch) > TILT_ABORT_RAD

    def _update_tilt_risk(self, now: float, roll: float, pitch: float) -> None:
        """Advance the continuous tilt-risk EMA. Called once per tick, before
        the arbiter picks a leg-control layer, so it tracks balance risk
        regardless of which layer is currently driving the legs -- the
        "always on" check that _settled() leans on to keep the robot from
        launching a new locomotion clip too soon after a wobble.
        """
        mag = self._predicted_tilt_rad(roll, pitch)
        dt = 0.0 if self._last_risk_update is None else max(0.0, now - self._last_risk_update)
        self._last_risk_update = now
        alpha = 1.0 - math.exp(-dt / TILT_RISK_TAU_S) if dt > 0 else 1.0
        self._tilt_risk += alpha * (mag - self._tilt_risk)

    def _marching(self) -> bool:
        gait = self.gait_cmd or {}
        return (
            str(gait.get("state", "idle")) == "march"
            and float(gait.get("cadence_hz", 0.0) or 0.0) > 0.0
            and float(gait.get("conf", 0.0) or 0.0) >= LOCOMOTION.walk_conf_min
        )

    def _drive_legs(self, now: float, roll: float, pitch: float,
                    yaw: float) -> None:
        """Pick and run exactly one leg commander for this simulation step."""
        torso_rp = (roll, pitch)
        fsr = self._read_fsr()
        falling = self._falling(roll, pitch)

        # (1) A clip is playing: it owns the whole body until it ends, unless the
        #     robot is about to go over or the watchdog fires.
        if self.motion.active:
            action = self.motion.action
            if falling:
                logger.warning("Tilt abort (%.2f, %.2f rad): stopping motion '%s'",
                               roll, pitch, action)
                self._end_motion(action, ok=False, reason="tilt abort")
            elif self._motion_overran(now):
                # A clip that never reports "over" would keep the whole body
                # suspended forever, which reads as a totally dead robot.
                logger.error(
                    "Motion '%s' overran its watchdog (%.1fs); taking the body "
                    "back. This clip will not be used again.",
                    action, now - (self._motion_started_at or now),
                )
                self.motion.drop(action)
                self._end_motion(action, ok=False, reason="watchdog")
            elif self.motion.poll():
                self.leg_mode = f"motion:{action}"
                return
            else:
                # Finished normally -- but only counts as a success if the robot
                # is still upright, otherwise we are walking ourselves over.
                upright = abs(roll) < MOTION_START_MAX_TILT_RAD * 2.0 and \
                    abs(pitch) < MOTION_START_MAX_TILT_RAD * 2.0
                self._end_motion(action, ok=upright, reason="clip finished")

        if self.leg_control == "off":
            self.leg_mode = "stand"
            self.driver.balance_tick(torso_rp)
            return

        # (2) Real locomotion: walk/turn with a pre-balanced clip.
        if self.leg_control == "auto" and not falling:
            plan = plan_action(
                yaw_error_rad=self.yaw_servo.error(yaw),
                gait=self.gait_cmd,
                available=self.motion.available,
                params=LOCOMOTION,
                turning=self._turning,
            )
            if plan.action is None:
                # Nothing left to correct: the rotation (if any) has converged.
                self._turning = False
            elif not self._settled(roll, pitch):
                # Starting a clip mid-wobble is how a walk becomes a fall; wait.
                self.leg_mode = "pose"
            elif self.motion.start(plan.action):
                logger.info("Locomotion: %s (%s)", plan.action, plan.reason)
                self._turning = plan.is_turn
                self._begin_motion(now)
                self.driver.release_to_motion()
                self.leg_mode = f"motion:{plan.action}"
                return
        else:
            self._turning = False

        # (3) No clip available but the human is walking: march in place.
        #     Never while going over -- the pose layer below is the better
        #     recovery, because its tilt gate ramps the asymmetric part of the
        #     posture out and returns the legs to the balanced symmetric crouch,
        #     with the CoM correction folded back in as soon as both feet are
        #     evenly loaded again.
        if (
            self.leg_control in ("auto", "engine")
            and not falling
            and self.driver.enable_walk
            and self._marching()
            and "forward" not in self.motion.available
        ):
            self.leg_mode = f"march:{WALK_TIER}"
            self.driver.gait_tick(now, torso_rp, fsr=fsr)
            return

        # (4) Default: per-leg pose imitation (squat, single-leg lift).
        if self.driver.lower_body is not None:
            # Give the standing turn a little immediate feedback via the shared
            # hip yaw while the (coarse) stepping turn has not fired yet.
            self.leg_mode = "pose"
            self.driver.lower_body_tick(
                now, torso_rp, fsr=fsr, yaw_bias=self.yaw_servo.error(yaw)
            )
            return

        self.leg_mode = "stand"
        self.driver.balance_tick(torso_rp)

    def _settled(self, roll: float, pitch: float) -> bool:
        """Is the robot upright and calm enough to hand over to a clip?

        The allowed tilt window is not fixed: it shrinks continuously from
        MOTION_START_MAX_TILT_RAD toward MOTION_START_MIN_TILT_RAD as recent
        tilt risk (self._tilt_risk) climbs toward TILT_ABORT_RAD, so the
        controller keeps declining to start another clip for a while after a
        wobble instead of only reacting once mid-clip.
        """
        risk_frac = max(0.0, min(1.0, self._tilt_risk / TILT_ABORT_RAD))
        ceiling = MOTION_START_MAX_TILT_RAD - risk_frac * (
            MOTION_START_MAX_TILT_RAD - MOTION_START_MIN_TILT_RAD
        )
        if abs(roll) > ceiling or abs(pitch) > ceiling:
            return False
        d_roll, d_pitch = self._tilt_rate()
        return abs(d_roll) < 1.0 and abs(d_pitch) < 1.0

    def _begin_motion(self, now: float) -> None:
        """Arm the watchdog for a clip we are about to hand the body to."""
        self._motion_started_at = now
        # Prefer the clip's own length (plus slack for Webots' interpolation);
        # fall back to the hard cap when the API will not tell us.
        duration = self.motion.duration_s()
        budget = min(duration * 1.5 + 1.0, MOTION_WATCHDOG_S) if duration else \
            MOTION_WATCHDOG_S
        self._motion_deadline = now + budget

    def _motion_overran(self, now: float) -> bool:
        return self._motion_deadline is not None and now > self._motion_deadline

    def _end_motion(self, action: Optional[str], *, ok: bool, reason: str) -> None:
        """Take the body back from a clip and update the locomotion health count."""
        self.motion.abort()
        self._motion_started_at = None
        self._motion_deadline = None
        self._reclaim()
        if ok:
            self._motion_failures = 0
            return
        self._motion_failures += 1
        logger.warning("Locomotion attempt '%s' ended badly (%s): failure %d/%d",
                       action, reason, self._motion_failures, MOTION_MAX_FAILURES)
        if self._motion_failures >= MOTION_MAX_FAILURES:
            logger.error(
                "Disabling locomotion clips for this session after %d bad "
                "attempts. The robot will keep imitating your pose and will "
                "march in place instead of walking. Check WorldInfo "
                "contactProperties and basicTimeStep (see the controller README).",
                self._motion_failures,
            )
            self.motion.drop(None)

    def _reclaim(self) -> None:
        """Take the body back after a clip and reset the leg sequencers."""
        self.driver.reclaim_from_motion()
        if self.driver.lower_body is not None:
            self.driver.lower_body.reset()

    # ---------------------------------------------------------------- logging
    def _log_status(self) -> None:
        if self.frame_count % STATUS_EVERY != 0:
            return
        elapsed = time.time() - self._last_log_time
        fps = STATUS_EVERY / elapsed if elapsed > 0 else 0.0
        stats = self.driver.stats
        logger.info(
            "Frame %d | sim %.1f Hz | %s | legs=%s | %d joints applied",
            self.frame_count, fps,
            "STALE (holding)" if stats.stale else "tracking",
            self.leg_mode, stats.joints_last_applied,
        )
        if self.leg_mode == "pose":
            m = self.driver.lower_body_meta
            logger.info("  legs: %s", m.get("why", "?"))
            logger.info(
                "        mode=%s stance=%s shift=%.2f lift=%.2f gate=%.2f "
                "margin=%+.4fm crouch=%.2f lift-cue=%s",
                m.get("mode"), m.get("stance_side") or "-", float(m.get("shift", 0.0)),
                float(m.get("lift", 0.0)), float(m.get("gate", 0.0)),
                float(m.get("stance_margin", 0.0)), float(m.get("crouch_u", 0.0)),
                m.get("lift_source", "?"),
            )
        elif self.leg_mode.startswith("march"):
            m = self.driver.gait_meta
            logger.info(
                "  march amp=%.2f cadence=%.2fHz phase=%.2f single_support=%s",
                float(m.get("amp_gain", 0.0)), float(m.get("cadence", 0.0)),
                float(m.get("phase", 0.0)), m.get("single_support", False),
            )
        if self.motion.available:
            logger.info("  heading error %+.0f deg (servo %s)",
                        math.degrees(self.yaw_servo.error(self._imu_rpy()[2])),
                        "latched" if self.yaw_servo.latched else "waiting")
        for name in self.driver.stuck_motors():
            logger.warning(
                "Motor '%s' may be stuck (avg err %.3f rad)",
                name, self.driver.health.average_error(name),
            )
        self._last_log_time = time.time()

    # ------------------------------------------------------------------- loop
    def tick(self) -> None:
        """One control step: ingest, sense, arbitrate, log.

        Kept separate from :meth:`run` so the whole per-step path can be driven
        from a test harness without reimplementing it -- a duplicated loop body
        is a loop body that drifts out of sync with the real one.
        """
        now = self.robot.getTime()
        command = self._drain_latest_command()
        if command is not None:
            # Prefer full-body retargeting from raw landmarks; fall back to
            # pre-computed joint angles if only those were sent.
            keypoints = command.get("keypoints")
            if keypoints:
                self.driver.update_from_keypoints(keypoints, now_s=now)
            else:
                angles = command.get("joint_angles_rad", {})
                if angles:
                    self.driver.update(angles, now_s=now)
            self.gait_cmd = command.get("gait")
            self.driver.set_gait_command(self.gait_cmd)
        elif self.driver.check_stale(now):
            # Tracking lost: tell every layer to stand down. They ramp back to
            # the balanced crouch rather than freezing mid-step. The lower body
            # has to be told explicitly: it latches the last observation so it
            # can control at simulation rate between camera frames, and without
            # an expiry it would hold a one-legged stance long after the human
            # walked away.
            self.gait_cmd = None
            self.driver.set_gait_command(None)
            self.driver.lower_body_stand_down()

        self.driver.read_feedback()
        roll, pitch, yaw = self._imu_rpy()
        self._update_tilt_risk(now, roll, pitch)
        self._update_yaw_servo(now, yaw)
        self._drive_legs(now, roll, pitch, yaw)

        if self.trajectory_log is not None:
            self.trajectory_log.record(
                now, self.frame_count, self.driver.commanded, self.driver.measured
            )
        self._log_status()
        self.frame_count += 1

    def run(self) -> None:
        """Step the simulation, containing errors so one bad step cannot end it.

        A raised exception used to break out of this loop straight into
        :meth:`_cleanup`, which sets every motor velocity to zero -- a single
        transient error left a permanently dead robot with no obvious cause. Now
        a failed step is logged, the body is forced back under our control (in
        case the failure happened mid-handover to a motion clip), and the loop
        carries on. Only a sustained run of failures gives up, and even then the
        robot is left standing rather than limp.
        """
        logger.info("Starting control loop...")
        try:
            while self.robot.step(self.timestep) != -1:
                try:
                    self.tick()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._errors += 1
                    logger.exception(
                        "Control step %d failed (%d consecutive): %s",
                        self.frame_count, self._errors, exc,
                    )
                    self._recover_from_error()
                    if self._errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            "Giving up after %d consecutive failed steps.",
                            self._errors,
                        )
                        break
                    self.frame_count += 1
                else:
                    self._errors = 0
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down")
        finally:
            self._cleanup()

    def _recover_from_error(self) -> None:
        """Best-effort return to a known-good state after a failed step."""
        try:
            if self.motion.active or self.driver.suspended:
                self._end_motion(self.motion.action, ok=False, reason="step error")
        except Exception:  # noqa: BLE001 - recovery must never raise
            logger.exception("Recovery itself failed; forcing control back")
            try:
                self.driver.reclaim_from_motion()
            except Exception:  # noqa: BLE001
                pass

    def _cleanup(self) -> None:
        logger.info("Cleaning up...")
        try:
            self.motion.abort()
            self.driver.reclaim_from_motion()
            self.driver.stop()
        finally:
            if self.trajectory_log is not None:
                self.trajectory_log.close()
            if hasattr(self, "sock"):
                self.sock.close()
            logger.info("Controller stopped after %d frames", self.frame_count)


def _sensor_magnitude(device: object) -> float:
    """Per-foot load from a Webots TouchSensor, 3-axis or 1-axis, NaN-safe.

    NAO's ``LFsr``/``RFsr`` are ``TouchSensor`` nodes of type ``"force-3d"``, so
    they answer to ``getValues()`` and return ``[fx, fy, fz]``; ``getValue()``
    (singular) only supports the ``"bumper"``/``"force"`` types and Webots raises
    on it. That mismatch is why an earlier version silently got no load
    information at all and the step gate never saw a weight transfer.

    Of the three axes we take **fz**: the load bearing on a foot is the vertical
    force, and the shear components can be large during a walk and would inflate
    the reading exactly when the step gate needs it to be honest. Shorter vectors
    fall back to the norm, and 1-axis sensors to ``getValue()``, so other NAO
    protos still work.
    """
    getter = getattr(device, "getValues", None)
    if getter is not None:
        try:
            values = getter()
            if values is not None:
                fz = float(values[2])
                return abs(fz) if math.isfinite(fz) else 0.0
        except Exception:  # noqa: BLE001
            pass
    try:
        value = float(device.getValue())  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return 0.0
    return abs(value) if math.isfinite(value) else 0.0


def main() -> None:
    try:
        PoseImitationController().run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
