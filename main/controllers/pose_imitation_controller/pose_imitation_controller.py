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

try:
    from controller import Motion, Robot  # type: ignore
except ImportError:
    print("Error: Webots controller module not found. Run this only in Webots.")
    sys.exit(1)

try:
    # Supervisor is a Robot subclass, so everything else in this file is
    # unaffected. It is needed only to RELOAD the world after a fall; detection
    # needs no special privileges. Absent on older builds, hence the guard.
    from controller import Supervisor  # type: ignore
except ImportError:  # pragma: no cover - depends on the Webots build
    Supervisor = None  # type: ignore

# Make the shared library importable regardless of Webots' working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "libraries"))
from pose_control_utils import JointTrajectoryLogger, NaoPoseDriver  # noqa: E402
from walk_motion import (  # noqa: E402
    LocomotionParams,
    YawServo,
    default_motion_search_dirs,
    find_motion_files,
    motion_joints,
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
# Default is "pose": the legs IMITATE, continuously and in real time, and balance
# is handled by shifting the centre of mass to make the imitated pose holdable
# rather than by attenuating it (see lower_body._shift_com). Locomotion clips are
# a 2-3 second commitment during which the camera is ignored for the leg joints,
# which is the opposite of real-time imitation -- set "auto" to re-enable them when
# covering ground matters more than following the legs.
LEG_CONTROL = "pose"

DRIVE_HEAD = True         # head yaw/pitch follow the human head
SWAP_SIDES = False        # True = mirror-image mapping (robot's left <-> your right)
SMOOTHING_ALPHA = 0.4     # EMA factor for arm/head targets (0..1, higher = snappier)
VELOCITY_SCALE = 0.5      # fraction of each joint's hardware max velocity
LEG_VELOCITY_FACTOR = 0.5 # extra slow-down on leg joints when merely posturing
STALE_AFTER_S = 0.5       # hold pose if no command for this long

# --- Fall recovery ---------------------------------------------------------
# When the robot goes down it stays down: every layer correctly stands itself
# down, and the rest of a test session is then spent driving a robot lying on the
# floor. A recorded session lost 170 of its 178 seconds that way.
#
# The test is the one that needs no conventions: how far the head sits above the
# soles, measured along the world vertical. Forward kinematics gives both in the
# torso frame and the (calibrated) InertialUnit gives the vertical, so no
# Supervisor is needed to DETECT a fall -- only to recover from one.
#
# The threshold separates cleanly from every legitimate posture, because NAO's
# crouch keeps the torso vertical and so barely lowers the head:
#
#     standing                0.460 m       tipped 30 deg      0.398 m
#     base crouch  u=0.10     0.459 m       tipped 60 deg      0.228 m
#     DEEPEST squat u=0.70    0.412 m       on its side       -0.000 m
#                                           face down          0.035 m
#
# 0.25 m is ~57 deg of tilt: far past any recoverable posture, and 40% below the
# deepest squat the robot will ever be asked for. Negative values -- feet above
# head -- are covered by the same test.
AUTO_RELOAD_ON_FALL = True
# 0.30, not 0.25: a recorded fall came to rest with the head at 0.255 m (soles
# carrying 0.04 N, i.e. the robot on its arms) and sat there, unrecovered, for
# the rest of the episode. The deepest squat is 0.41 m, so 0.30 still clears every
# legitimate posture by 110 mm.
FALL_HEAD_HEIGHT_M = 0.30
FALL_CONFIRM_S = 1.0        # must hold this long: no reloading on a transient
FALL_RELOAD_COOLDOWN_S = 10.0
FALL_MAX_RELOADS = 20       # a broken setup must not reload forever
# Two further ways to be "down" that the head-height test misses, both recorded:
#   * STUCK: the torso tilted 0.37 rad for 40 s, both CoM shifters at their
#     clamps, the soles carrying 0.4 and 5 N -- the robot propped on an arm,
#     head still 0.35 m up. Nothing the balance loop can do from there.
#   * UNLOADED: soles carrying almost nothing for seconds on end -- the robot is
#     resting on something other than its feet. A hop or a rocking foot unloads
#     for a fraction of a second; single support keeps ~50 N on the stance foot.
STUCK_TILT_RAD = 0.35       # rad; past every stand-down gate, short of a topple
STUCK_CONFIRM_S = 3.0       # s; the balance loop gets this long to recover
FALL_UNLOADED_N = 8.0       # N; total sole load below this = not standing on the feet
FALL_UNLOADED_S = 2.0       # s

# --- IMU tilt zero ---------------------------------------------------------
# The InertialUnit's roll/pitch are used as "how tipped over is the robot", which
# assumes they read zero when it stands upright. On this NAO model they do not:
# measured on a standing robot at rest, roll reads +1.618 rad (93 deg) while the
# foot sensors carry its full 50 N of body weight and the gyro sits at
# 0.007 rad/s. The sensor frame is mounted rotated; the robot is fine.
#
# Every lower-body symptom on this project traced back to that one number:
#   * balance displaces the CoM by tilt_weight * height * roll = 291 mm of
#     phantom lateral error, against an 88 mm support half-width, so the loop
#     leans the robot onto one foot permanently (measured 47.6 N vs 2.8 N);
#   * lower_body's tilt_abort_rad (0.28) can never be satisfied, so leg imitation
#     stands down every frame and the legs stay bit-identical;
#   * _falling() is permanently true, so no walk clip or march ever starts.
#
# So the zero is LEARNED at startup instead of assumed. The world file always
# spawns the robot standing, and the foot sensors confirm it independently -- soles
# carrying roughly body weight mean "standing", whatever the IMU claims. Samples
# are only accepted while that holds, so a controller restarted on an
# already-fallen robot will not latch the fall as its new upright.
IMU_AUTO_ZERO = True
IMU_CALIBRATION_S = 1.0        # settle window at startup
IMU_CALIBRATION_MIN_SAMPLES = 20
# Soles must carry at least this much IN TOTAL for a sample to count as
# "standing". NAO weighs ~5.2 kg, so a loaded pair reads ~50 N.
IMU_CALIBRATION_MIN_LOAD_N = 20.0
# ...and EACH sole must carry at least this much. The total alone is not
# evidence of standing: a fallen robot resting on one foot logged 25.85 N on the
# right against 0.72 N on the left and passed the total check, latching a zero
# 17.4 deg out. A standing robot splits its weight (measured 25.26/25.26 N), so
# 10 N per foot passes every genuine stance and rejects a one-footed sprawl.
IMU_CALIBRATION_MIN_SOLE_LOAD_N = 10.0
# ...and the head must be about where a standing robot's head is, by forward
# kinematics with tilt taken as zero (so it does not depend on the estimate being
# calibrated). Standing reads ~0.459 m and the deepest squat 0.41; every bad
# calibration in the logs read 0.049-0.205 m.
IMU_CALIBRATION_MIN_HEAD_M = 0.40

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

# HARD WATCHDOG on motion playback. While a clip runs it owns the joints it
# declares -- for Webots' walk clips, the 12 leg joints -- and per-joint
# commanding is suspended for those, so anything that stops the clip from ever
# reporting "over" freezes the legs indefinitely. Webots' walk clips are a few
# seconds long, so any suspension beyond this is a bug, not a long clip: we take
# the joints back and stop trusting clips.
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

# Per-step controller state written alongside the joint angles. A log of joint
# angles alone says what the body did but not what the controller believed, and
# every diagnosis on this project has needed both halves: the permanent lean was
# only identifiable because the roll channels could be compared, and its CAUSE
# needed the support margin, which was not recorded at all.
DIAGNOSTIC_COLUMNS = (
    "imu_roll", "imu_pitch", "imu_yaw",
    "imu_roll_raw", "imu_pitch_raw", "imu_zero_roll", "imu_zero_pitch",
    "gyro_roll_rate", "gyro_pitch_rate",
    "tilt_risk", "predicted_tilt",
    "leg_mode", "stale",
    "lb_mode", "lb_shift", "lb_lift", "lb_gate", "lb_stance_margin",
    "lb_lean_scale", "lb_crouch_u", "lb_crouch_cue", "lb_conf",
    "lb_lift_source", "lb_rejected", "lb_why",
    # Where every radian of pelvis shift came from: the lower body's feed-forward
    # term (ff), the balance loop's feedback term (fb), and the static margin the
    # sum achieved. Without these the 2026-09-03 falls could only be attributed by
    # arithmetic on the joint columns (both shifters at their clamps read as
    # HipPitch +0.45 / AnklePitch -0.65), which took a day to notice.
    "lb_ff_pitch", "lb_ff_roll", "lb_ff_width", "lb_fb_pitch", "lb_fb_roll",
    "lb_com_margin",
    # Lateral centre of pressure from the foot sensors, as the left foot's share
    # of the total load: the one direct measurement of where the weight really is.
    "cop_share_l",
    "support_margin_x", "support_margin_y", "head_height", "reloads",
    "clip_planned", "clip_status", "clips_available", "yaw_stable",
    "yaw_error", "yaw_latched",
    "fsr_l", "fsr_r",
    "gait_state", "gait_cadence", "gait_conf", "body_yaw",
)

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

    def __init__(self, files: dict[str, str],
                 log: object | None = None) -> None:
        self._files = dict(files)
        self._cache: dict[str, object] = {}
        self._joints: dict[str, list[str]] = {}
        self._log = log or (lambda *_a, **_k: None)
        self.action: str | None = None
        self._motion: object | None = None

    @property
    def available(self) -> dict[str, str]:
        return dict(self._files)

    @property
    def active(self) -> bool:
        return self._motion is not None

    def _load(self, action: str) -> object | None:
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

    def duration_s(self) -> float | None:
        """Clip length in seconds, or None if Webots will not tell us."""
        if self._motion is None:
            return None
        try:
            ms = float(self._motion.getDuration())
        except Exception:  # noqa: BLE001
            return None
        return ms / 1000.0 if math.isfinite(ms) and ms > 0.0 else None

    def joints(self, action: str | None = None) -> list[str]:
        """Joints the clip for ``action`` drives, or [] if that is not knowable.

        Read from the clip's header (see ``walk_motion.motion_joints``) and cached,
        so the controller can suspend per-joint commanding for exactly those and
        leave the rest -- the arms and head -- imitating throughout the clip.
        """
        target = action or self.action
        if target is None:
            return []
        if target not in self._joints:
            self._joints[target] = motion_joints(self._files.get(target))
        return list(self._joints[target])

    def drop(self, action: str | None = None) -> None:
        """Stop using ``action`` (or every clip) for the rest of the session."""
        target = action or self.action
        self.abort()
        if target is None:
            self._files.clear()
            self._cache.clear()
            self._joints.clear()
        else:
            self._files.pop(target, None)
            self._cache.pop(target, None)
            self._joints.pop(target, None)

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
        # A Supervisor when the world allows it (the Nao node needs
        # `supervisor TRUE`), so a fall can be recovered from automatically. It
        # behaves as a plain Robot for everything else, and falls back to one when
        # the class is unavailable.
        self.robot = Supervisor() if (AUTO_RELOAD_ON_FALL and Supervisor is not None) \
            else Robot()
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
                LOG_DIR, self.driver.logged_joints,
                diagnostics=DIAGNOSTIC_COLUMNS, logger=logger.info,
            )

        self.gait_cmd: dict | None = None
        self.leg_mode = "stand"
        self.frame_count = 0
        self._last_log_time = time.time()
        # True while a multi-clip rotation is still converging. It survives clip
        # boundaries on purpose: that is what lets plan_action use its tighter
        # gate to finish a turn instead of stalling one clip short.
        self._turning = False
        # Motion-playback watchdog state. Suspension hands the WHOLE body to a
        # clip, so it must always be bounded in time and in failure count.
        self._motion_started_at: float | None = None
        self._motion_deadline: float | None = None
        self._motion_failures = 0
        self._errors = 0
        # Continuous tilt-risk EMA (see _update_tilt_risk / _settled): runs every
        # tick regardless of which leg-control layer is active.
        self._tilt_risk = 0.0
        self._last_risk_update: float | None = None
        # Learned IMU tilt zero (see IMU_AUTO_ZERO). None until calibrated; tilt is
        # reported as level in the meantime so nothing aborts on a reading we do
        # not yet understand.
        self._imu_zero: tuple | None = None
        self._imu_cal: list[tuple] = []
        self._imu_cal_started: float | None = None
        if not IMU_AUTO_ZERO:
            self._imu_zero = (0.0, 0.0)
        # Fall detection / recovery state.
        self._fall_since: float | None = None
        self._stuck_since: float | None = None
        self._unloaded_since: float | None = None
        self._reloads = 0
        self._last_reload: float | None = None
        self._head_height: float | None = None
        # What the locomotion layer wanted and what became of it. Recorded because
        # "the robot never walks" has several indistinguishable causes -- no clips
        # on disk, a clip Webots refuses to load, the settle gate never opening, or
        # the turn servo starving forward motion -- and none of them are visible
        # from the joint angles.
        self._clip_planned = ""
        self._clip_status = "idle"
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
        for reason in d.degraded:
            logger.error("DEGRADED: %s", reason)
        if d.degraded:
            logger.error(
                "The layer(s) above are NOT running. The usual cause is that "
                "Webots is launching this controller with an interpreter that "
                "has no NumPy -- check Tools > Preferences > Python command and "
                "point it at the project's environment. This message is repeated "
                "on the status line so it cannot scroll away."
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

    def _calibrate_imu(self, now: float, raw_roll: float, raw_pitch: float,
                       fsr: dict[str, float] | None) -> None:
        """Learn what "upright" reads on this robot's InertialUnit.

        Only accepts samples while the foot sensors say the robot is standing, so
        the zero cannot be latched from a fallen pose. Where no foot sensors
        resolve it falls back to trusting the world file's spawn -- which does
        place the robot upright -- and the startup log says which happened.
        """
        if self._imu_zero is not None:
            return
        if not all(math.isfinite(v) for v in (raw_roll, raw_pitch)):
            return
        if fsr:
            left = float(fsr.get("L", 0.0))
            right = float(fsr.get("R", 0.0))
            # TOTAL load is not evidence of standing: measured on a fallen robot
            # after a reset, one foot alone carried 25.85 N against 0.72 N on the
            # other -- 26.6 N of "standing" that sailed past a 20 N total check
            # and latched a zero 17.4 deg out, worth ~90 mm of phantom CoM shift.
            # Both soles must be loaded, which a robot lying on its side is not.
            if min(left, right) < IMU_CALIBRATION_MIN_SOLE_LOAD_N:
                return
            if (left + right) < IMU_CALIBRATION_MIN_LOAD_N:
                return
        # Independent of the foot sensors: forward kinematics with the tilt taken
        # as zero gives the height the head WOULD be at if upright. A standing
        # robot reads ~0.459 m; every bad calibration in the logs read 0.049 to
        # 0.205 m, so this separates them with an enormous margin and does not
        # depend on the very tilt estimate being calibrated.
        height = self.head_height(0.0, 0.0)
        if height is not None and height < IMU_CALIBRATION_MIN_HEAD_M:
            return
        if self._imu_cal_started is None:
            self._imu_cal_started = now
        self._imu_cal.append((raw_roll, raw_pitch))
        if (now - self._imu_cal_started) < IMU_CALIBRATION_S:
            return
        if len(self._imu_cal) < IMU_CALIBRATION_MIN_SAMPLES:
            return
        rolls = sorted(v[0] for v in self._imu_cal)
        pitches = sorted(v[1] for v in self._imu_cal)
        mid = len(rolls) // 2
        samples = len(self._imu_cal)
        self._imu_zero = (rolls[mid], pitches[mid])
        self._imu_cal.clear()
        magnitude = max(abs(self._imu_zero[0]), abs(self._imu_zero[1]))
        emit = logger.warning if magnitude > 0.05 else logger.info
        emit(
            "IMU tilt zero learned from %d standing samples: roll %+.3f, pitch "
            "%+.3f rad. Tilt is measured relative to this from now on.%s",
            samples, self._imu_zero[0], self._imu_zero[1],
            "" if magnitude <= 0.05 else
            f" That is {math.degrees(magnitude):.0f} deg, so this model's "
            f"InertialUnit is mounted rotated: without the correction every tilt "
            f"gate and the balance loop read a robot standing still as falling.",
        )

    # ------------------------------------------------------------ fall recovery
    def head_height(self, roll: float, pitch: float) -> float | None:
        """Height of the head above the soles along the WORLD vertical, in metres.

        Forward kinematics places the head and both soles in the torso frame; the
        calibrated InertialUnit supplies the vertical. Upright that is ~0.46 m and
        a deep squat only takes it to 0.41 m, because NAO's crouch keeps the torso
        vertical -- so it is a clean fall test rather than a proxy for one.

        Returns None when there is no CoM model to do the kinematics with (no
        NumPy in Webots' interpreter), in which case the caller falls back to tilt.
        """
        balance = self.driver.balance
        if balance is None:
            return None
        try:
            import numpy as np

            state = dict(self.driver.commanded)
            state.update(self.driver.measured)
            frames = balance.model.frames(state)
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            # Only the world-z row of the torso->world rotation is needed.
            rot_z = np.array([-sp * cr, sr, cr * cp])
            head_z = float(rot_z @ frames["HeadPitch"][:3, 3])
            soles = []
            for side in ("L", "R"):
                T = frames[f"{side}AnkleRoll"]
                sole = T[:3, :3] @ np.array([0.035, 0.0, -0.04519]) + T[:3, 3]
                soles.append(float(rot_z @ sole))
            return head_z - sum(soles) / len(soles)
        except Exception:  # noqa: BLE001 - a diagnostic must never break control
            return None

    def _fallen(self, now: float, roll: float, pitch: float,
                fsr: dict[str, float] | None = None) -> bool:
        """Has the robot been down for long enough to be worth recovering?

        Three independent tests, each with its own confirmation time so a stumble
        the balance loop catches is not treated as a fall:

        * the head is low (a topple; ``FALL_HEAD_HEIGHT_M`` / ``FALL_CONFIRM_S``),
        * the torso has been tilted past every stand-down gate for seconds
          (``STUCK_TILT_RAD`` / ``STUCK_CONFIRM_S``): the robot is propped on
          something, not standing, however high its head still is,
        * the soles have carried almost nothing for seconds
          (``FALL_UNLOADED_N`` / ``FALL_UNLOADED_S``).
        """
        if self._imu_zero is None:
            # Tilt is not yet meaningful, so neither is any test built on it.
            self._fall_since = self._stuck_since = self._unloaded_since = None
            return False
        height = self.head_height(roll, pitch)
        self._head_height = height
        tilt = max(abs(roll), abs(pitch))
        if height is not None:
            down = height < FALL_HEAD_HEIGHT_M
        else:
            # No kinematics available: tilt alone. Well past the abort limit, so it
            # cannot fire on a posture the balance loop might still save.
            down = tilt > 1.0

        confirmed = False
        if down:
            if self._fall_since is None:
                self._fall_since = now
                reason = (f"head only {height:.3f} m above the soles"
                          if height is not None else f"tilt {tilt:.2f} rad")
                logger.error("FALL DETECTED (%s). Confirming for %.1fs...",
                             reason, FALL_CONFIRM_S)
            confirmed |= (now - self._fall_since) >= FALL_CONFIRM_S
        else:
            self._fall_since = None

        if tilt > STUCK_TILT_RAD:
            if self._stuck_since is None:
                self._stuck_since = now
            elif (now - self._stuck_since) >= STUCK_CONFIRM_S:
                if not confirmed:
                    logger.error("STUCK: torso tilted %.2f rad for %.1fs; treating "
                                 "as a fall.", tilt, now - self._stuck_since)
                confirmed = True
        else:
            self._stuck_since = None

        total = None if not fsr else float(fsr.get("L", 0.0)) + float(fsr.get("R", 0.0))
        if total is not None and total < FALL_UNLOADED_N:
            if self._unloaded_since is None:
                self._unloaded_since = now
            elif (now - self._unloaded_since) >= FALL_UNLOADED_S:
                if not confirmed:
                    logger.error("OFF ITS FEET: soles carried %.1f N for %.1fs; "
                                 "treating as a fall.", total, now - self._unloaded_since)
                confirmed = True
        else:
            self._unloaded_since = None
        return confirmed

    def _recover_from_fall(self, now: float) -> bool:
        """Put the robot back on its feet by resetting the simulation.

        Returns True if a reset was issued. Rate-limited and capped: a setup that
        falls immediately every time must not reload in a loop, it must stop and
        say so, because an endless reload is harder to diagnose than a robot lying
        still.
        """
        if not AUTO_RELOAD_ON_FALL:
            return False
        # WALL time, not ``now``: ``now`` is robot.getTime(), which simulationReset()
        # rewinds to zero. Stamping the cooldown with it meant the next recovery had
        # to wait until the NEW episode's clock passed the OLD episode's reload time,
        # so the wait grew by 10 s every fall. The logs show it exactly -- resets at
        # sim 3.32, 16.82, 26.82, 36.82, each one 10 s after the previous stamp --
        # and a run that first fell at sim 46.60 needed sim > 56.6 to be picked up,
        # never reached it, and lay on the floor for the rest of the session. A rate
        # limit on a real-world action belongs on a clock that does not rewind.
        wall_now = time.time()
        if self._last_reload is not None and \
                (wall_now - self._last_reload) < FALL_RELOAD_COOLDOWN_S:
            return False
        if self._reloads >= FALL_MAX_RELOADS:
            if self._reloads == FALL_MAX_RELOADS:
                self._reloads += 1     # log this once, then stay quiet
                logger.error(
                    "Robot has fallen %d times; not reloading again. Something is "
                    "wrong that a reload will not fix -- run "
                    "scripts/analyze_run.py on this session's log.",
                    FALL_MAX_RELOADS,
                )
            return False

        self._reloads += 1
        self._last_reload = wall_now
        self._fall_since = self._stuck_since = self._unloaded_since = None
        logger.error("Reloading the simulation to stand the robot back up "
                     "(recovery %d/%d).", self._reloads, FALL_MAX_RELOADS)
        # Hand every layer back a clean slate first: whether or not Webots
        # restarts this controller, the state must not describe the fallen robot.
        self._reset_for_new_episode()
        for method in ("simulationReset", "worldReload"):
            call = getattr(self.robot, method, None)
            if call is None:
                continue
            try:
                call()
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s() failed (%s); trying the next option.",
                               method, exc)
        logger.error(
            "Cannot reset the simulation: this controller is not a Supervisor. Add "
            "`supervisor TRUE` to the Nao node in the world file, or press "
            "Ctrl+Shift+R in Webots to reload by hand."
        )
        return False

    def _reset_for_new_episode(self) -> None:
        """Drop all state that describes the old, fallen robot."""
        self.motion.abort()
        self.driver.reclaim_from_motion()
        # Return the COMMANDED pose to neutral. The reset puts the robot back
        # upright, but the driver's base_targets still held the collapsed pose it
        # fell in, so the first step of the new episode drove it straight back
        # into that shape -- which is what put the topple inside the IMU
        # calibration window and taught the balance loop a fallen "upright".
        self.driver.upper_body_stand_down()
        self.driver.lower_body_stand_down()
        if self.driver.lower_body is not None:
            self.driver.lower_body.reset()
        self.yaw_servo.reset()
        self._turning = False
        self._tilt_risk = 0.0
        self._last_risk_update = None
        # The tilt zero is KEPT. It measures how the InertialUnit is mounted on the
        # torso (Nao.proto: rolled +pi/2 about x), a property of the robot that a
        # reset cannot change. Re-learning it here is what produced 4-6 different
        # zeros per session, up to 2.6 deg apart (13 mm of phantom CoM error, with
        # a 39 mm fore/aft margin), each latched from a robot that had just been
        # stood back up and was still settling. The first zero of a session, taken
        # from the world file's clean spawn, is also the cleanest.
        self._imu_cal.clear()
        self._imu_cal_started = None

    def _corrected_tilt(self, raw_roll: float, raw_pitch: float) -> tuple:
        """(roll, pitch) relative to the learned upright; (0, 0) until calibrated."""
        if self._imu_zero is None:
            return (0.0, 0.0)
        return (raw_roll - self._imu_zero[0], raw_pitch - self._imu_zero[1])

    def _init_walk_sensors(self) -> None:
        """Enable the gyro/accelerometer and any foot force sensors.

        All best-effort and NaN-guarded: the march tier needs none of them, and
        the stepping gate falls back to the CoM model alone when they are absent.
        """
        self.gyro = None
        self.fsr: dict[str, list[object]] = {"L": [], "R": []}
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

    def _read_fsr(self) -> dict[str, float] | None:
        """Per-foot load ``{"L": n, "R": n}`` from the FSRs, or None.

        NAO's foot sensors are 3-axis ("force-3d") TouchSensors, so the value
        comes from ``getValues()``, not ``getValue()`` -- reading them as scalars
        is why an earlier version silently got no load information and the
        stepping gate never saw a weight transfer. Both APIs are handled so this
        also works with 1-axis protos.
        """
        if not self.fsr["L"] and not self.fsr["R"]:
            return None
        out: dict[str, float] = {}
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
    def _drain_latest_command(self) -> dict | None:
        """Return the most recent pose command, discarding any backlog.

        UDP can queue several frames between simulation steps. We only care
        about the freshest pose, so we drain the buffer and keep the last one
        (keeps end-to-end latency low -- PRD NFR-1).
        """
        latest: dict | None = None
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
                    yaw: float, fsr: dict[str, float] | None = None) -> None:
        """Pick and run exactly one leg commander for this simulation step.

        ``roll``/``pitch`` are relative to the LEARNED upright (see
        :meth:`_corrected_tilt`), never the raw InertialUnit reading.
        """
        torso_rp = (roll, pitch)
        tilt_rate = self._tilt_rate()
        if fsr is None:
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
            self.driver.balance_tick(torso_rp, tilt_rate=tilt_rate, now_s=now)
            return

        # (2) Real locomotion: walk/turn with a pre-balanced clip.
        #
        #     ``clip_declined`` records that this layer WANTED to act and could
        #     not. Branch 3 keys off it, which is the difference between "the
        #     robot marches in place while it waits to be steady enough to walk"
        #     and the old behaviour: it fell through to branch 4, which does
        #     nothing with the legs, so the robot stood motionless while the human
        #     marched at it and reported legs=pose with no error anywhere.
        clip_declined = False
        if self.leg_control == "auto" and not falling:
            plan = plan_action(
                yaw_error_rad=self.yaw_servo.error(yaw),
                gait=self.gait_cmd,
                available=self.motion.available,
                params=LOCOMOTION,
                turning=self._turning,
                yaw_trustworthy=self.yaw_servo.stable(),
            )
            self._clip_planned = plan.action or ""
            if plan.action is None:
                # Nothing left to correct: the rotation (if any) has converged.
                self._turning = False
                self._clip_status = "nothing planned"
            elif not self._settled(roll, pitch):
                # Starting a clip mid-wobble is how a walk becomes a fall; wait,
                # but let branch 3 keep the legs moving while we wait.
                clip_declined = True
                self._clip_status = "declined: not settled"
            elif self.motion.start(plan.action):
                logger.info("Locomotion: %s (%s)", plan.action, plan.reason)
                self._turning = plan.is_turn
                self._begin_motion(now)
                # Hand the clip only the joints it declares. Webots' walk clips
                # drive the 12 leg joints and nothing else, so the arms and head
                # keep following the human right through the step.
                self.driver.release_to_motion(self.motion.joints(plan.action))
                self.leg_mode = f"motion:{plan.action}"
                self._clip_status = "started"
                return
            else:
                # The clip was planned but Webots would not play it. Silence here
                # made a rejected clip indistinguishable from a clip nobody asked
                # for, which is a long debugging session for a one-line cause.
                logger.warning(
                    "Locomotion clip '%s' was planned (%s) but would not start; "
                    "falling back to the march engine this step.",
                    plan.action, plan.reason,
                )
                clip_declined = True
                self._clip_status = "start REFUSED by Webots"
        else:
            self._turning = False

        # (3) The clip layer is not walking us: march in place instead.
        #     Never while going over -- the pose layer below is the better
        #     recovery, because its tilt gate ramps the asymmetric part of the
        #     posture out and returns the legs to the balanced symmetric crouch,
        #     with the CoM correction folded back in as soon as both feet are
        #     evenly loaded again.
        #
        #     The gate is "did the clip layer decline?", NOT "is a forward clip
        #     absent from disk?". The old disk test meant that installing Webots
        #     -- which ships Forwards.motion -- permanently disabled the march
        #     engine, so the fallback existed only on machines that could not run
        #     the robot in the first place.
        if (
            self.leg_control in ("auto", "engine")
            and not falling
            and self.driver.enable_walk
            and self._marching()
            and (clip_declined or "forward" not in self.motion.available)
        ):
            self.leg_mode = f"march:{WALK_TIER}"
            self.driver.gait_tick(now, torso_rp, fsr=fsr, tilt_rate=tilt_rate)
            return

        # (4) Default: per-leg pose imitation (squat, single-leg lift).
        if self.driver.lower_body is not None:
            # Give the standing turn a little immediate feedback via the shared
            # hip yaw while the (coarse) stepping turn has not fired yet.
            self.leg_mode = "pose"
            self.driver.lower_body_tick(
                now, torso_rp, fsr=fsr, yaw_bias=self.yaw_servo.error(yaw),
                tilt_rate=tilt_rate,
            )
            return

        self.leg_mode = "stand"
        self.driver.balance_tick(torso_rp, tilt_rate=tilt_rate, now_s=now)

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

    def _end_motion(self, action: str | None, *, ok: bool, reason: str) -> None:
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
    def _diagnostics(self, roll: float, pitch: float, yaw: float) -> dict[str, object]:
        """One row of controller state for the trajectory log.

        Cheap by design -- everything here is already computed for this step,
        except the support margin, which is one forward-kinematics pass and is the
        single most useful number for telling "the robot is standing badly" from
        "the robot is standing fine and the pose is wrong".
        """
        d_roll, d_pitch = self._tilt_rate()
        m = self.driver.lower_body_meta
        gait = self.gait_cmd or {}
        fsr = self._read_fsr() or {}
        raw_roll, raw_pitch, _ = self._imu_rpy()
        zero = self._imu_zero
        out: dict[str, object] = {
            "imu_roll": roll, "imu_pitch": pitch, "imu_yaw": yaw,
            "imu_roll_raw": raw_roll, "imu_pitch_raw": raw_pitch,
            "imu_zero_roll": None if zero is None else zero[0],
            "imu_zero_pitch": None if zero is None else zero[1],
            "gyro_roll_rate": d_roll, "gyro_pitch_rate": d_pitch,
            "tilt_risk": self._tilt_risk,
            "predicted_tilt": self._predicted_tilt_rad(roll, pitch),
            "leg_mode": self.leg_mode,
            "stale": int(bool(self.driver.stats.stale)),
            "lb_mode": m.get("mode"),
            "lb_shift": m.get("shift"),
            "lb_lift": m.get("lift"),
            "lb_gate": m.get("gate"),
            "lb_stance_margin": m.get("stance_margin"),
            "lb_lean_scale": m.get("lean_scale"),
            "lb_crouch_u": m.get("crouch_u"),
            "lb_crouch_cue": m.get("crouch_cue"),
            "lb_conf": m.get("confidence"),
            "lb_lift_source": m.get("lift_source"),
            "lb_rejected": m.get("rejected"),
            # Quoted-safe: the CSV writer escapes it, and it is the one field that
            # names the limiting factor in words.
            "lb_why": m.get("why"),
            "lb_ff_pitch": m.get("com_shift_pitch"),
            "lb_ff_roll": m.get("com_shift_roll"),
            "lb_ff_width": m.get("com_shift_width"),
            "lb_fb_pitch": m.get("com_fb_pitch"),
            "lb_fb_roll": m.get("com_fb_roll"),
            "lb_com_margin": m.get("com_margin"),
            "cop_share_l": (
                fsr["L"] / (fsr["L"] + fsr["R"])
                if fsr.get("L") is not None and fsr.get("R") is not None
                and (fsr["L"] + fsr["R"]) > 1.0 else None
            ),
            "head_height": self._head_height,
            "reloads": self._reloads,
            "clip_planned": self._clip_planned,
            "clip_status": ("playing" if self.motion.active else self._clip_status),
            "clips_available": len(self.motion.available),
            "yaw_stable": int(bool(self.yaw_servo.stable())),
            "yaw_error": self.yaw_servo.error(yaw),
            "yaw_latched": int(bool(self.yaw_servo.latched)),
            "fsr_l": fsr.get("L"), "fsr_r": fsr.get("R"),
            "gait_state": gait.get("state"),
            "gait_cadence": gait.get("cadence_hz"),
            "gait_conf": gait.get("conf"),
            "body_yaw": gait.get("body_yaw_rad"),
        }
        balance = self.driver.balance
        if balance is not None:
            try:
                state = dict(self.driver.commanded)
                state.update(self.driver.measured)
                mx, my = balance.model.support_margins(state)
                out["support_margin_x"] = mx
                out["support_margin_y"] = my
            except Exception:  # noqa: BLE001 - diagnostics must never break control
                pass
        return out

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
            logger.info("  locomotion: %d clips | planned=%s | %s | yaw %s",
                        len(self.motion.available), self._clip_planned or "-",
                        self._clip_status,
                        "steady" if self.yaw_servo.stable() else "TOO NOISY to turn on")
            d = self.yaw_servo.diagnostics(self._imu_rpy()[2])
            # The reference pair is logged alongside the error on purpose: an
            # error alone cannot distinguish a loop that is converging from one
            # stuck at a fixed offset, and a fixed offset is what a recorded
            # session actually showed.
            logger.info(
                "  heading: error %+.0f deg | human %+.0f -> %+.0f | robot %+.0f "
                "-> %+.0f | servo %s",
                math.degrees(d["error"]),
                math.degrees(d["human_ref"]), math.degrees(d["human_now"]),
                math.degrees(d["robot_ref"]), math.degrees(d["robot_now"]),
                "latched" if self.yaw_servo.latched else "waiting to latch",
            )
        # Repeated, not printed once at startup: a silently absent balance layer
        # is the single most expensive failure mode this controller has.
        if self._imu_zero is None:
            logger.warning(
                "  IMU tilt zero not yet learned (%d standing samples, need %d): "
                "tilt is reported as level, so the balance loop and every tilt "
                "gate are idle. Is the robot standing with its feet loaded?",
                len(self._imu_cal), IMU_CALIBRATION_MIN_SAMPLES,
            )
        if self.driver.balance is not None:
            warning = self.driver.balance.diverging()
            if warning:
                logger.error("  %s", warning)
        if self._head_height is not None:
            logger.info("  head %.3f m above the soles (fall below %.2f); "
                        "recoveries this session: %d",
                        self._head_height, FALL_HEAD_HEIGHT_M, self._reloads)
        for reason in self.driver.degraded:
            logger.error("  DEGRADED: %s", reason)
        for name in self.driver.stuck_motors():
            logger.error(
                "Motor '%s' is not tracking its command (avg err %.3f rad). If "
                "this persists the joint is mechanically blocked -- most likely "
                "against the torso or the opposite limb.",
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
            # The arms and head need telling too, or they hold the departed
            # human's last pose indefinitely (see upper_body_stand_down).
            self.driver.upper_body_stand_down()

        self.driver.read_feedback()
        raw_roll, raw_pitch, yaw = self._imu_rpy()
        fsr = self._read_fsr()
        self._calibrate_imu(now, raw_roll, raw_pitch, fsr)
        roll, pitch = self._corrected_tilt(raw_roll, raw_pitch)
        if self._fallen(now, roll, pitch, fsr) and self._recover_from_fall(now):
            return
        self._update_tilt_risk(now, roll, pitch)
        if self.driver.balance is not None:
            self.driver.balance.note_tilt(now, roll, pitch)
        self._update_yaw_servo(now, yaw)
        self._drive_legs(now, roll, pitch, yaw, fsr=fsr)

        if self.trajectory_log is not None:
            self.trajectory_log.record(
                now, self.frame_count, self.driver.commanded, self.driver.measured,
                self._diagnostics(roll, pitch, yaw),
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
