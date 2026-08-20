"""End-to-end test of the Webots controller against a mocked Webots runtime.

The controller file itself (sockets, device lookup, motion playback, and above all
the lower-body *arbiter*) cannot be reached by the library unit tests, yet it is
where the wiring bugs live: a renamed method, a layer that never gets ticked, two
layers commanding the legs at once. Since ``main/libraries`` is deliberately
Webots-free, the only thing standing between those tests and a full-loop test is
the ``controller`` module -- so we fake it.

The fake robot tracks its commands perfectly (position sensors echo the last
commanded angle), which is enough to exercise every control path: real UDP
packets go in, and we assert on which layer drove the legs and what it commanded.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import socket
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_DIR = os.path.join(REPO, "main", "controllers", "pose_imitation_controller")
sys.path.insert(0, os.path.join(REPO, "main", "libraries"))

from pose_control_utils import get_default_motor_configs  # noqa: E402

CONFIGS = get_default_motor_configs()
TIMESTEP_MS = 20


# ---------------------------------------------------------------------------
# Fake Webots
# ---------------------------------------------------------------------------
class FakeMotor:
    def __init__(self, name):
        self.name = name
        self.position = 0.0
        self.velocity = 0.0
        self.commands = 0

    def setPosition(self, value):  # noqa: N802 - Webots API name
        self.position = value
        self.commands += 1

    def setVelocity(self, value):  # noqa: N802
        self.velocity = value


class FakeSensor:
    """Position sensor echoing its motor: a perfect-tracking robot."""

    def __init__(self, motor):
        self.motor = motor

    def enable(self, _ms):
        pass

    def getValue(self):  # noqa: N802
        return self.motor.position


class FakeInertialUnit:
    def __init__(self):
        self.rpy = [0.0, 0.0, 0.0]

    def enable(self, _ms):
        pass

    def getRollPitchYaw(self):  # noqa: N802
        return list(self.rpy)


class FakeVector3:
    def __init__(self, values=(0.0, 0.0, 0.0)):
        self.values = list(values)

    def enable(self, _ms):
        pass

    def getValues(self):  # noqa: N802
        return list(self.values)


class FakeMotion:
    """Stand-in for Webots' Motion: finishes after ``STEPS`` polls.

    ``NEVER_OVER`` reproduces the failure mode that matters most: a clip that
    plays but never reports being over. Because playback suspends per-joint
    commanding for the WHOLE body, that used to freeze the entire robot
    indefinitely with no diagnostic.
    """

    STEPS = 20
    NEVER_OVER = False
    DURATION_MS = 1200.0
    played = []

    def __init__(self, path):
        self.path = path
        self.loop = False
        self.time = 0.0
        self.rewinds = 0
        self._remaining = 0

    def isValid(self):  # noqa: N802
        return True

    def setLoop(self, value):  # noqa: N802
        self.loop = value

    def setTime(self, ms):  # noqa: N802
        self.time = ms
        if ms == 0:
            self.rewinds += 1

    def play(self):
        self._remaining = self.STEPS
        FakeMotion.played.append(os.path.basename(self.path))
        return True

    def stop(self):
        self._remaining = 0

    def getDuration(self):  # noqa: N802
        return self.DURATION_MS

    def isOver(self):  # noqa: N802
        if FakeMotion.NEVER_OVER:
            return False
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True


class FakeFsr:
    """Foot force sensor that responds to the robot's lean, like the real one.

    A static symmetric reading would be wrong in an informative way: the step
    gate rightly refuses to unload a foot the sensors say is still carrying half
    the robot, so a fixed 50/50 mock would test nothing but the veto. A positive
    same-sign hip roll carries the pelvis toward the robot's RIGHT, so it loads
    the right foot -- that is the relation modelled here.
    """

    TOTAL_N = 52.0
    SENSITIVITY = 3.0

    def __init__(self, robot, side):
        self.robot = robot
        self.side = side

    def enable(self, _ms):
        pass

    def getValues(self):  # noqa: N802
        lean = 0.5 * (self.robot.motors["LHipRoll"].position
                      + self.robot.motors["RHipRoll"].position)
        share_right = min(1.0, max(0.0, 0.5 + self.SENSITIVITY * lean))
        share = share_right if self.side == "R" else 1.0 - share_right
        # Non-zero shear on x/y: the reader must take fz, not the vector norm.
        return [4.0, -3.0, self.TOTAL_N * share]

    def getValue(self):  # noqa: N802
        raise RuntimeError("getValue() is not supported for a force-3d sensor")


class FakeRobot:
    def __init__(self, *, with_fsr=True):
        self.time = 0.0
        self.motors = {name: FakeMotor(name) for name in CONFIGS}
        self.devices = {}
        for name, motor in self.motors.items():
            self.devices[name] = motor
            self.devices[name + "S"] = FakeSensor(motor)
        self.imu = FakeInertialUnit()
        self.devices["inertial unit"] = self.imu
        self.devices["gyro"] = FakeVector3()
        self.devices["accelerometer"] = FakeVector3((0.0, 0.0, -9.81))
        if with_fsr:
            self.devices["LFsr"] = FakeFsr(self, "L")
            self.devices["RFsr"] = FakeFsr(self, "R")

    def getBasicTimeStep(self):  # noqa: N802
        return TIMESTEP_MS

    def getDevice(self, name):  # noqa: N802
        return self.devices.get(name)

    def getTime(self):  # noqa: N802
        return self.time

    def step(self, ms):
        self.time += ms / 1000.0
        return 0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def controller_module(monkeypatch, tmp_path):
    """Import the real controller file with a fake ``controller`` package."""
    fake = types.ModuleType("controller")
    fake.Robot = FakeRobot
    fake.Motion = FakeMotion
    monkeypatch.setitem(sys.modules, "controller", fake)
    monkeypatch.syspath_prepend(CONTROLLER_DIR)

    sys.modules.pop("pose_imitation_controller", None)
    mod = importlib.import_module("pose_imitation_controller")
    mod = importlib.reload(mod)

    clips = tmp_path / "motions"
    clips.mkdir()
    for name in ("Forwards.motion", "TurnLeft60.motion", "TurnRight60.motion"):
        (clips / name).write_text("#WEBOTS_MOTION,V1.0\n", encoding="utf-8")

    monkeypatch.setattr(mod, "MOTION_SEARCH_DIRS_EXTRA", [str(clips)])
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", False)
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())
    FakeMotion.played = []
    FakeMotion.NEVER_OVER = False
    yield mod
    FakeMotion.NEVER_OVER = False
    sys.modules.pop("pose_imitation_controller", None)


class Harness:
    """Drives a real ``PoseImitationController`` over real UDP."""

    def __init__(self, mod):
        self.mod = mod
        self.ctl = mod.PoseImitationController()
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = mod.UDP_PORT

    def send(self, keypoints=None, gait=None):
        payload = {"timestamp_s": 0.0, "frame_index": 0, "joint_angles_rad": {}}
        if keypoints:
            payload["keypoints"] = keypoints
        if gait:
            payload["gait"] = gait
        self.tx.sendto(json.dumps(payload).encode(), ("127.0.0.1", self.port))

    def spin(self, steps, keypoints=None, gait=None, every=2):
        """Advance the REAL control loop, feeding a frame every ``every`` steps.

        Calls ``PoseImitationController.tick`` rather than reimplementing it, so
        the harness cannot drift out of sync with the controller it is testing.
        """
        c = self.ctl
        for i in range(steps):
            if keypoints is not None and i % every == 0:
                self.send(keypoints, gait)
            c.robot.step(c.timestep)
            c.tick()
        return c.leg_mode

    def angle(self, name):
        return self.ctl.robot.motors[name].position

    def close(self):
        self.tx.close()
        self.ctl._cleanup()


@pytest.fixture
def harness(controller_module):
    h = Harness(controller_module)
    yield h
    h.close()


# ---------------------------------------------------------------------------
# Synthetic subject
# ---------------------------------------------------------------------------
def subject(*, left_leg=(0.0, 0.0, 0.0), right_leg=(0.0, 0.0, 0.0), yaw=0.0):
    """Landmarks for a subject with each leg at ``(roll_mag, hip_pitch, knee)``."""
    kps = {}
    a = math.radians(yaw)
    for name, half, y in (("shoulder", 0.06, 0.30), ("hip", 0.04, 0.55)):
        for side, sgn in (("left", -1.0), ("right", +1.0)):
            kps[f"{side}_{name}"] = [
                0.5 + sgn * half * math.cos(a), y, -sgn * half * math.sin(a), 1.0,
            ]
    for side, sgn in (("left", -1.0), ("right", +1.0)):
        kps[f"{side}_elbow"] = [0.5 + sgn * 0.07, 0.41, 0.0, 1.0]
        kps[f"{side}_wrist"] = [0.5 + sgn * 0.08, 0.52, 0.0, 1.0]
    kps["nose"] = [0.5, 0.18, 0.0, 1.0]

    for side, legs in (("L", left_leg), ("R", right_leg)):
        roll, hip, knee = legs
        out = -1.0 if side == "L" else 1.0
        pre = "left_" if side == "L" else "right_"
        origin = kps[pre + "hip"]
        knee_pt = _seg(origin, out, 0.18, roll, hip)
        ankle_pt = _seg(knee_pt, out, 0.18, roll, hip + knee)
        kps[pre + "knee"] = knee_pt
        kps[pre + "ankle"] = ankle_pt
    return kps


def _seg(origin, out, length, roll, pitch):
    lateral = math.sin(roll) * math.cos(pitch)
    vertical = math.cos(roll) * math.cos(pitch)
    forward = -math.sin(pitch)
    return [
        origin[0] + out * length * lateral,
        origin[1] + length * vertical,
        origin[2] - length * forward,
        1.0,
    ]


STANDING = subject()
LEFT_LEG_UP = subject(left_leg=(0.0, -1.0, 1.4))
SQUAT = subject(left_leg=(0.0, -0.55, 1.1), right_leg=(0.0, -0.55, 1.1))
MARCH_GAIT = {"state": "march", "cadence_hz": 0.9, "phase": 0.5, "swing_side": 1,
              "intensity": 0.8, "turn": 0.0, "conf": 0.95,
              "body_yaw_rad": 0.0, "yaw_conf": 0.95}
IDLE_GAIT = {"state": "idle", "cadence_hz": 0.0, "phase": 0.0, "swing_side": 0,
             "intensity": 0.0, "turn": 0.0, "conf": 0.95,
             "body_yaw_rad": 0.0, "yaw_conf": 0.95}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def test_controller_finds_all_devices_and_layers(harness) -> None:
    c = harness.ctl
    assert len(c.driver.motors) == len(CONFIGS)
    assert len(c.driver.sensors) == len(CONFIGS)
    assert c.imu is not None and c.gyro is not None
    assert c.driver.balance is not None          # CoM balance
    assert c.driver.lower_body is not None       # per-leg pose imitation
    assert c.driver.gait_engine is not None      # march engine
    assert set(c.motion.available) == {"forward", "turn_left", "turn_right"}


def test_foot_sensors_are_read_as_three_axis(harness) -> None:
    """NAO's FSRs are force-3d: getValue() raises on them, and of the three axes
    only the vertical one is the load the step gate should trust."""
    loads = harness.ctl._read_fsr()
    assert loads is not None
    assert loads["L"] == pytest.approx(26.0)
    assert loads["R"] == pytest.approx(26.0)
    # ... and they follow the lean, so the step gate has something to check.
    harness.ctl.robot.motors["LHipRoll"].position = 0.15
    harness.ctl.robot.motors["RHipRoll"].position = 0.15
    leaning = harness.ctl._read_fsr()
    assert leaning["R"] > leaning["L"]


def test_no_foot_sensors_reports_none_rather_than_zeros(controller_module) -> None:
    """Zeros would look like "no weight anywhere" and veto every step."""
    mod = controller_module
    ctl = mod.PoseImitationController.__new__(mod.PoseImitationController)
    ctl.fsr = {"L": [], "R": []}
    assert ctl._read_fsr() is None


# ---------------------------------------------------------------------------
# Leg arbitration
# ---------------------------------------------------------------------------
def test_standing_still_uses_pose_imitation(harness) -> None:
    mode = harness.spin(120, STANDING, IDLE_GAIT)
    assert mode == "pose"
    meta = harness.ctl.driver.lower_body_meta
    assert meta["mode"] == "double"
    assert not FakeMotion.played           # no clip fires while standing still


def test_squat_reaches_the_knees(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    straight = harness.angle("LKneePitch")
    harness.spin(120, SQUAT, IDLE_GAIT)
    assert harness.angle("LKneePitch") > straight + 0.15
    # Symmetric: a squat must not become a lean.
    assert abs(harness.angle("LKneePitch") - harness.angle("RKneePitch")) < 0.05


def test_raising_one_leg_transfers_weight_then_lifts(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["mode"] == "single"
    assert meta["stance_side"] == "R"
    assert meta["shift"] > 0.9              # weight moved first
    assert meta["lift"] > 0.2               # then the foot came up
    assert meta["stance_margin"] > 0.0      # the CoM model agreed
    # The commanded robot really lifted the matching leg.
    assert harness.angle("LKneePitch") > harness.angle("RKneePitch") + 0.3
    assert harness.angle("LHipPitch") < harness.angle("RHipPitch") - 0.2


def test_lowering_the_leg_returns_to_a_symmetric_stance(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    assert harness.ctl.driver.lower_body_meta["mode"] == "single"
    harness.spin(200, STANDING, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["mode"] == "double"
    assert abs(harness.angle("LKneePitch") - harness.angle("RKneePitch")) < 0.05


def test_marching_plays_a_forward_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    assert mode == "motion:forward"
    assert "Forwards.motion" in FakeMotion.played


def test_a_clip_suspends_per_joint_commanding(harness) -> None:
    """While a clip owns the body, our targets must not fight its keyframes --
    and the velocity caps must be lifted or it cannot reach them."""
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(4, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is True
    motor = harness.ctl.robot.motors["LKneePitch"]
    assert motor.velocity == pytest.approx(CONFIGS["LKneePitch"].max_velocity)
    before = motor.commands
    harness.spin(10, STANDING, MARCH_GAIT)
    assert motor.commands == before          # nothing commanded during playback


def test_a_replayed_clip_is_rewound_first(harness) -> None:
    """A finished clip resumed without a rewind returns immediately, so the
    robot would take one step and then stand there looking stuck."""
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(200, STANDING, MARCH_GAIT)
    assert FakeMotion.played.count("Forwards.motion") >= 2   # replayed
    clip = harness.ctl.motion._cache["forward"]
    assert clip.rewinds >= 2
    assert clip.loop is False                                # never looped


def test_a_multi_clip_rotation_keeps_going_until_aligned(harness) -> None:
    """Turning must converge across clip boundaries, not stall one clip short."""
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=1.8)               # ~103 deg
    kps = subject(yaw=60.0)
    for _ in range(6):
        harness.spin(40, kps, turned)
        if harness.ctl.motion.active:
            # Pretend the clip turned the robot by its nominal 60 deg.
            harness.ctl.imu.rpy[2] += math.radians(60)
    assert FakeMotion.played.count("TurnLeft60.motion") >= 2
    assert harness.ctl._turning is False                     # converged
    assert abs(harness.ctl.yaw_servo.error(harness.ctl.imu.rpy[2])) < 0.6


def test_control_is_reclaimed_when_the_clip_ends(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(4, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is True
    mode = harness.spin(200, STANDING, IDLE_GAIT)
    assert harness.ctl.driver.suspended is False
    assert mode == "pose"
    # The step sequencer was reset, not resumed mid-transfer.
    assert harness.ctl.driver.lower_body_meta["mode"] == "double"


def test_falling_aborts_the_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(4, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is True
    harness.ctl.imu.rpy = [0.6, 0.0, 0.0]     # well past TILT_ABORT_RAD
    harness.spin(6, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is False
    assert not harness.ctl.motion.active


def test_gyro_predicts_a_fall_before_the_tilt_crosses_the_limit(harness) -> None:
    c = harness.ctl
    c.imu.rpy = [0.30, 0.0, 0.0]              # below TILT_ABORT_RAD on its own
    assert c._falling(0.30, 0.0) is False
    c.gyro.values = [2.0, 0.0, 0.0]           # ... but tipping fast
    assert c._falling(0.30, 0.0) is True


def test_losing_the_human_stands_the_robot_down(harness) -> None:
    """Without an expiry on the latched observation the robot would hold a
    one-legged stance forever after the human walked out of frame."""
    harness.spin(120, STANDING, IDLE_GAIT)
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    assert harness.ctl.driver.lower_body_meta["mode"] == "single"
    # Stop sending frames entirely.
    mode = harness.spin(300, None)
    assert mode == "pose"
    assert harness.ctl.driver.stats.stale is True
    assert harness.ctl.driver.lower_body_meta["mode"] == "double"
    assert abs(harness.angle("LKneePitch") - harness.angle("RKneePitch")) < 0.05


# ---------------------------------------------------------------------------
# Turning
# ---------------------------------------------------------------------------
def test_body_rotation_triggers_a_turn_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=0.9)
    mode = harness.spin(20, subject(yaw=50.0), turned)
    assert mode == "motion:turn_left"
    assert "TurnLeft60.motion" in FakeMotion.played


def test_turn_direction_follows_the_sign_of_the_rotation(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(20, subject(yaw=-50.0), dict(IDLE_GAIT, body_yaw_rad=-0.9))
    assert "TurnRight60.motion" in FakeMotion.played


def test_turning_stops_once_the_robot_has_caught_up(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=0.9)
    harness.spin(20, subject(yaw=50.0), turned)
    assert harness.ctl.motion.active
    # The clip physically turned the robot: report the new heading.
    harness.ctl.imu.rpy = [0.0, 0.0, 0.9]
    mode = harness.spin(200, subject(yaw=50.0), turned)
    assert mode == "pose"
    assert not harness.ctl.motion.active


def test_turning_takes_priority_over_walking(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    both = dict(MARCH_GAIT, body_yaw_rad=0.9)
    harness.spin(20, subject(yaw=50.0), both)
    assert FakeMotion.played[0] == "TurnLeft60.motion"


def test_a_small_rotation_gets_a_hip_yaw_bias_not_a_clip(harness) -> None:
    """Below the turn gate the robot should still acknowledge the rotation."""
    harness.spin(60, STANDING, IDLE_GAIT)
    mode = harness.spin(120, subject(yaw=12.0), dict(IDLE_GAIT, body_yaw_rad=0.2))
    assert mode == "pose"
    assert not FakeMotion.played
    bias = harness.angle("LHipYawPitch")
    assert 0.0 < bias <= 0.12 + 1e-6


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "keypoints,gait",
    [(STANDING, IDLE_GAIT), (LEFT_LEG_UP, IDLE_GAIT), (SQUAT, IDLE_GAIT),
     (STANDING, MARCH_GAIT)],
)
def test_no_command_ever_leaves_the_joint_limits(harness, keypoints, gait) -> None:
    harness.spin(200, keypoints, gait)
    for name, motor in harness.ctl.robot.motors.items():
        cfg = CONFIGS[name]
        assert math.isfinite(motor.position)
        assert cfg.min_angle - 1e-9 <= motor.position <= cfg.max_angle + 1e-9, name


def test_malformed_packets_are_ignored(harness) -> None:
    harness.tx.sendto(b"not json at all", ("127.0.0.1", harness.port))
    harness.tx.sendto(b"\xff\xfe\x00", ("127.0.0.1", harness.port))
    mode = harness.spin(40, STANDING, IDLE_GAIT)
    assert mode == "pose"


def test_only_one_layer_commands_the_legs_per_step(harness) -> None:
    """The invariant the whole arbiter exists for: two commanders means a fall."""
    harness.spin(60, STANDING, IDLE_GAIT)
    motor = harness.ctl.robot.motors["LKneePitch"]
    motor.commands = 0
    harness.spin(10, LEFT_LEG_UP, IDLE_GAIT, every=1)
    assert motor.commands == 10


# ---------------------------------------------------------------------------
# Freeze resistance
#
# Motion playback suspends per-joint commanding for the WHOLE body, so anything
# that stops a clip from ending stops the entire robot -- which is exactly how
# "the robot is completely frozen" happens. These tests assert that no single
# failure can hold the body indefinitely.
# ---------------------------------------------------------------------------
def test_a_clip_that_never_ends_cannot_freeze_the_robot(harness) -> None:
    mod = harness.mod
    FakeMotion.NEVER_OVER = True
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(10, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is True      # clip took the body

    # Spin well past the watchdog budget.
    steps = int((mod.MOTION_WATCHDOG_S + 2.0) / (TIMESTEP_MS / 1000.0))
    mode = harness.spin(steps, STANDING, IDLE_GAIT)

    assert harness.ctl.driver.suspended is False     # ... and gave it back
    assert mode == "pose"
    # The offending clip is not tried again.
    assert "forward" not in harness.ctl.motion.available


def test_the_watchdog_uses_the_clips_own_duration(harness) -> None:
    """A 1.2 s clip must not be able to hold the body for the full hard cap."""
    mod = harness.mod
    FakeMotion.NEVER_OVER = True
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(4, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is True
    budget = harness.ctl._motion_deadline - harness.ctl._motion_started_at
    assert budget < mod.MOTION_WATCHDOG_S
    assert budget == pytest.approx(FakeMotion.DURATION_MS / 1000.0 * 1.5 + 1.0)


def test_repeated_bad_locomotion_gives_up_on_clips(harness) -> None:
    """Falling over again and again is worse than never walking."""
    mod = harness.mod
    harness.spin(60, STANDING, IDLE_GAIT)
    for _ in range(mod.MOTION_MAX_FAILURES):
        harness.spin(6, STANDING, MARCH_GAIT)
        harness.ctl.imu.rpy = [0.6, 0.0, 0.0]        # tilt abort
        harness.spin(4, STANDING, MARCH_GAIT)
        harness.ctl.imu.rpy = [0.0, 0.0, 0.0]
    assert harness.ctl.motion.available == {}
    mode = harness.spin(60, STANDING, IDLE_GAIT)
    assert mode == "pose"                            # still imitating
    assert harness.ctl.driver.suspended is False


def test_a_clip_is_not_started_while_the_robot_is_wobbling(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.ctl.gyro.values = [3.0, 0.0, 0.0]        # tipping fast
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    assert not FakeMotion.played
    assert mode == "pose"
    harness.ctl.gyro.values = [0.0, 0.0, 0.0]
    harness.spin(20, STANDING, MARCH_GAIT)
    assert "Forwards.motion" in FakeMotion.played    # ... and once calm, it goes


def test_a_failing_step_does_not_end_the_loop_or_limp_the_robot(harness) -> None:
    """One transient error used to break run(), which then zeroed every motor
    velocity -- a permanently dead robot from a single bad frame."""
    c = harness.ctl
    calls = {"n": 0}
    real = c.driver.lower_body_tick

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] in (3, 4, 5):
            raise RuntimeError("synthetic sensor glitch")
        return real(*a, **kw)

    c.driver.lower_body_tick = flaky
    for i in range(40):
        if i % 2 == 0:
            harness.send(STANDING, IDLE_GAIT)
        c.robot.step(c.timestep)
        try:
            c.tick()
        except Exception:
            c._errors += 1
            c._recover_from_error()

    assert calls["n"] > 5                 # kept going past the failures
    assert c.driver.suspended is False    # recovery forced control back
    for motor in c.robot.motors.values():
        assert motor.velocity > 0.0       # never left limp


def test_recovery_forces_the_body_back_if_a_step_fails_mid_clip(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(4, STANDING, MARCH_GAIT)
    assert harness.ctl.driver.suspended is True
    harness.ctl._recover_from_error()
    assert harness.ctl.driver.suspended is False
    assert not harness.ctl.motion.active


# ---------------------------------------------------------------------------
# Legs must respond even when the camera crops the lower body
# ---------------------------------------------------------------------------
def _crop(keypoints, *names):
    out = dict(keypoints)
    for name in names:
        out[name] = list(out[name][:3]) + [0.05]
    return out


def test_a_leg_lift_is_seen_with_the_feet_out_of_frame(harness) -> None:
    """Standing close to a webcam crops the shins. The feet-based ground line is
    then unavailable, and without the knee fallback the lift reads as exactly
    zero however high the leg goes -- i.e. "leg movement is not working"."""
    standing = _crop(STANDING, "left_ankle", "right_ankle")
    lifted = _crop(LEFT_LEG_UP, "left_ankle", "right_ankle")
    harness.spin(150, standing, IDLE_GAIT)
    assert harness.ctl.driver.lower_body_meta["lift_source"] == "knees"
    harness.spin(250, lifted, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["lift_source"] == "knees"
    assert meta["stance_side"] == "R"
    assert meta["lift"] > 0.2
    # With no ankle in view the knee bend is genuinely unobservable, so the lift
    # shows up as hip flexion (a raised straight leg) rather than a knee fold.
    # That is the honest reading of what the camera can see -- and it is still a
    # clearly raised leg, which is the point.
    assert harness.angle("LHipPitch") < harness.angle("RHipPitch") - 0.3
    assert harness.angle("LKneePitch") <= harness.angle("RKneePitch") + 0.05


def test_legs_without_knees_or_feet_say_so_instead_of_failing_silently(harness) -> None:
    blind = _crop(LEFT_LEG_UP, "left_ankle", "right_ankle",
                  "left_knee", "right_knee")
    harness.spin(120, blind, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["lift"] == 0.0
    assert "out of frame" in str(meta["why"]) or "landmarks" in str(meta["why"])


def test_the_status_line_always_explains_itself(harness) -> None:
    harness.spin(120, STANDING, IDLE_GAIT)
    why = harness.ctl.driver.lower_body_meta["why"]
    assert isinstance(why, str) and why
    harness.spin(200, LEFT_LEG_UP, IDLE_GAIT)
    assert "stepping" in harness.ctl.driver.lower_body_meta["why"]
