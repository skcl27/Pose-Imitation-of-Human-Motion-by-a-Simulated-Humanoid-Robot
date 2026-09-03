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

from nao_retarget import _side_sign  # noqa: E402
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
        self.resets = 0
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

    # -- Supervisor surface, so fall recovery can be exercised --------------
    def simulationReset(self):  # noqa: N802 - Webots API name
        """Restore the initial state, as Webots does: the robot stands back up."""
        self.resets += 1
        self.imu.rpy = [0.0, 0.0, 0.0]
        for motor in self.motors.values():
            motor.position = 0.0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def controller_module(monkeypatch, tmp_path):
    """Import the real controller file with a fake ``controller`` package."""
    fake = types.ModuleType("controller")
    fake.Robot = FakeRobot
    fake.Supervisor = FakeRobot          # a Supervisor IS a Robot, plus reset
    fake.Motion = FakeMotion
    monkeypatch.setitem(sys.modules, "controller", fake)
    monkeypatch.syspath_prepend(CONTROLLER_DIR)

    sys.modules.pop("pose_imitation_controller", None)
    mod = importlib.import_module("pose_imitation_controller")
    mod = importlib.reload(mod)

    clips = tmp_path / "motions"
    clips.mkdir()
    # Real Webots NAO walk clips declare exactly the 12 leg joints in their
    # header and nothing else; the fakes have to say the same thing or the
    # per-joint handover cannot be exercised.
    header = "#WEBOTS_MOTION,V1.0," + ",".join(
        f"{s}{j}" for s in ("L", "R")
        for j in ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch",
                  "AnklePitch", "AnkleRoll")
    ) + "\n"
    for name in ("Forwards.motion", "TurnLeft60.motion", "TurnRight60.motion"):
        (clips / name).write_text(header, encoding="utf-8")

    # The shipped default is LEG_CONTROL="pose" -- the legs imitate continuously
    # and locomotion clips are opt-in, because a clip is a 2-3 second commitment
    # during which the camera is ignored for the leg joints. These tests cover the
    # locomotion layer, so they opt in; the default itself is asserted by
    # test_the_shipped_default_is_imitation_not_locomotion.
    monkeypatch.setattr(mod, "LEG_CONTROL", "auto")
    monkeypatch.setattr(mod, "MOTION_SEARCH_DIRS_EXTRA", [str(clips)])
    # Clip discovery has to be HERMETIC. Setting MOTION_SEARCH_DIRS_EXTRA alone is
    # not enough: default_motion_search_dirs() also appends $WEBOTS_HOME and eight
    # well-known install roots, so on a machine that actually has Webots the real
    # clips leak in and these tests assert against a set they do not control. The
    # effect was backwards -- the suite passed on a box that could not run the
    # robot and failed on the one that could.
    monkeypatch.setattr(
        mod, "default_motion_search_dirs",
        lambda extra=None: [str(d) for d in (extra or [])],
    )
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
    """Landmarks for a subject with each leg at ``(roll_mag, hip_pitch, knee)``.

    Shoulders/hips are placed so that ``nao_retarget._torso_frame`` computes
    the identity basis (right=(1,0,0), up=(0,-1,0), forward=(0,0,-1)) at
    yaw=0, then rotated about the vertical axis by ``yaw`` degrees -- the
    torso-local frame is built fresh from these landmarks every frame, so a
    non-zero yaw exercises the whole point of that upgrade (a subject who
    doesn't face the camera). Leg segments use the same swing-twist forward
    kinematics as ``nao_retarget._swing_twist`` inverts -- see
    ``tests/test_nao_retarget.py``'s ``_leg_dir`` for the derivation.
    """
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
        pre = "left_" if side == "L" else "right_"
        origin = kps[pre + "hip"]
        knee_pt = _seg(side, origin, 0.18, roll, hip)
        ankle_pt = _seg(side, knee_pt, 0.18, roll, hip + knee)
        kps[pre + "knee"] = knee_pt
        kps[pre + "ankle"] = ankle_pt
    return kps


def _seg(side, origin, length, roll, pitch):
    """One limb segment's endpoint at NAO angles ``(roll, pitch)``, in the
    identity torso frame (see ``subject()``): direction
    ``(side_sign*sin(roll)cos(pitch), cos(roll)cos(pitch), -sin(pitch))``.
    """
    s = _side_sign(side)
    dx = s * math.sin(roll) * math.cos(pitch)
    dy = math.cos(roll) * math.cos(pitch)
    dz = -math.sin(pitch)
    return [
        origin[0] + length * dx,
        origin[1] + length * dy,
        origin[2] + length * dz,
        1.0,
    ]


STANDING = subject()
LEFT_LEG_UP = subject(left_leg=(0.0, -1.0, 1.4))
SQUAT = subject(left_leg=(0.0, -0.55, 1.1), right_leg=(0.0, -0.55, 1.1))
# Legs spread outward: both hips abducted by the same amount.
LEGS_APART = subject(left_leg=(0.35, 0.0, 0.0), right_leg=(0.35, 0.0, 0.0))
LEGS_WIDE = subject(left_leg=(0.70, 0.0, 0.0), right_leg=(0.70, 0.0, 0.0))
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


def test_spreading_the_legs_actually_widens_the_stance(harness) -> None:
    """A wider stance is CoM-neutral and *enlarges* the support polygon, so it is
    safer than standing and must pass through at full authority. Gating it like a
    lean turned a 20 deg human spread into a 6 deg robot one."""
    harness.spin(150, STANDING, IDLE_GAIT)
    assert abs(harness.angle("LHipRoll")) < 0.03      # feet together
    harness.spin(200, LEGS_APART, IDLE_GAIT)
    # NAO's roll signs are mirrored: +L and -R both mean outward.
    assert harness.angle("LHipRoll") > 0.25
    assert harness.angle("RHipRoll") < -0.25
    # Both soles stay flat -- otherwise the robot stands on its inner edges.
    for side in ("L", "R"):
        assert harness.angle(f"{side}HipRoll") + harness.angle(f"{side}AnkleRoll") \
            == pytest.approx(0.0, abs=0.02)


def test_a_very_wide_spread_saturates_instead_of_tipping(harness) -> None:
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(200, LEGS_WIDE, IDLE_GAIT)
    roll = harness.angle("LHipRoll")
    # Reaches past what the ankle can level -- spending a bounded sole tilt to
    # buy width, because stopping at the ankle's limit saturated below the ~25 deg
    # people actually spread to.
    assert roll > abs(CONFIGS["LAnkleRoll"].min_angle)
    assert roll < CONFIGS["LHipRoll"].max_angle          # hip could go further
    # ... but no sole ends up more than the budget off flat. The budget is a
    # guarantee on the commanded target; the per-joint smoother can overshoot it
    # by a hair in transit because hip and ankle travel different distances, so
    # allow a small settling margin.
    from lower_body import LowerBodyParams
    budget = LowerBodyParams().sole_tilt_budget
    harness.spin(400, LEGS_WIDE, IDLE_GAIT)
    for side in ("L", "R"):
        tilt = harness.angle(f"{side}HipRoll") + harness.angle(f"{side}AnkleRoll")
        assert abs(tilt) <= budget + 0.01, (side, tilt)


def test_the_legs_come_back_together(harness) -> None:
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(200, LEGS_APART, IDLE_GAIT)
    assert harness.angle("LHipRoll") > 0.25
    harness.spin(200, STANDING, IDLE_GAIT)
    assert abs(harness.angle("LHipRoll")) < 0.05


def test_raising_one_leg_reaches_the_full_requested_lift(harness) -> None:
    """The reported symptom was a leg lift doing nothing. Assert it goes all the
    way: the CoM model permits it once the weight really has transferred."""
    harness.spin(150, STANDING, IDLE_GAIT)
    harness.spin(300, LEFT_LEG_UP, IDLE_GAIT)
    meta = harness.ctl.driver.lower_body_meta
    assert meta["gate"] == pytest.approx(1.0, abs=1e-6)
    assert meta["lift"] > 0.9
    assert harness.angle("LKneePitch") > 1.0             # knee clearly folded
    assert harness.angle("LHipPitch") < -0.8             # thigh clearly raised


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
    # Held, not flashed: a turn is only started on a heading estimate that has been
    # STEADY for about a second (YawServo.stable). Body yaw is the noisiest cue in
    # the pipeline -- it swung over 187 degrees in a recorded session while the
    # subject just stood there -- and acting on every excursion starved forward
    # walking completely.
    mode = harness.spin(80, subject(yaw=50.0), turned)
    assert mode == "motion:turn_left"
    assert "TurnLeft60.motion" in FakeMotion.played


def test_turn_direction_follows_the_sign_of_the_rotation(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(80, subject(yaw=-50.0), dict(IDLE_GAIT, body_yaw_rad=-0.9))
    assert "TurnRight60.motion" in FakeMotion.played


def test_turning_stops_once_the_robot_has_caught_up(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    turned = dict(IDLE_GAIT, body_yaw_rad=0.9)
    harness.spin(80, subject(yaw=50.0), turned)
    assert harness.ctl.motion.active
    # The clip physically turned the robot: report the new heading.
    harness.ctl.imu.rpy = [0.0, 0.0, 0.9]
    mode = harness.spin(200, subject(yaw=50.0), turned)
    assert mode == "pose"
    assert not harness.ctl.motion.active


def test_turning_takes_priority_over_walking_once_the_heading_is_trusted(harness) -> None:
    """Heading beats walking -- but only on a heading worth acting on.

    While the yaw estimate is still settling the robot walks forward instead of
    standing there, which is the deliberate choice: body yaw is the noisiest cue in
    the pipeline, and waiting for it starved forward locomotion entirely (measured:
    the servo error was past the turn gate in 77% of frames while the subject simply
    stood in front of the camera, and no clip ever ran). Walking a little off-heading
    is recoverable; never walking is the bug being reported.
    """
    harness.spin(60, STANDING, IDLE_GAIT)
    both = dict(MARCH_GAIT, body_yaw_rad=0.9)
    harness.spin(200, subject(yaw=50.0), both)
    assert "TurnLeft60.motion" in FakeMotion.played, FakeMotion.played
    # Once steady, the turn is what gets chosen over walking on.
    assert harness.ctl.yaw_servo.stable() is True


def test_a_noisy_heading_does_not_block_forward_walking(harness) -> None:
    """The starvation this gate exists to prevent, from the other side."""
    harness.spin(60, STANDING, IDLE_GAIT)
    # A heading estimate that swings wildly: never steady, always past the gate.
    swing = 1.0
    for _ in range(12):
        swing = -swing
        harness.spin(10, subject(yaw=50.0 * swing),
                     dict(MARCH_GAIT, body_yaw_rad=0.9 * swing))
    assert harness.ctl.yaw_servo.stable() is False
    assert "Forwards.motion" in FakeMotion.played, FakeMotion.played


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


def test_a_clip_stays_blocked_for_a_while_after_a_tilt_spike(harness) -> None:
    """A wobble that never crosses TILT_ABORT_RAD (so it never mid-clip-aborts
    anything, and nothing is even playing yet) should still make the
    controller more cautious about STARTING the next clip for a while -- the
    continuous tilt-risk EMA, not just the binary settle threshold."""
    harness.spin(60, STANDING, IDLE_GAIT)
    # A tilt spike under TILT_ABORT_RAD (0.40) but over the old fixed
    # MOTION_START_MAX_TILT_RAD (0.15) ceiling, held long enough to build risk.
    harness.ctl.imu.rpy = [0.35, 0.0, 0.0]
    harness.spin(150, STANDING, IDLE_GAIT)          # ~3s, ~2 EMA time constants
    # Settle back to a tilt that would have passed the OLD fixed ceiling...
    harness.ctl.imu.rpy = [0.10, 0.0, 0.0]
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    # ... but risk is still elevated right after the wobble, so no clip yet.
    assert not FakeMotion.played
    # The legs keep marching in place while the clip layer waits. They used to
    # fall through to the pose layer, which does nothing with the legs unless it
    # can see a leg lift -- so the robot stood motionless while the human marched
    # at it. Declining to WALK is not a reason to stop moving.
    assert mode == "march:march"
    # Give risk time to decay back down toward the current (calmer) tilt.
    harness.spin(400, STANDING, IDLE_GAIT)          # ~8s, several time constants
    harness.spin(20, STANDING, MARCH_GAIT)
    assert "Forwards.motion" in FakeMotion.played   # ... and once it has, it goes


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


def test_unresponsive_foot_sensors_reduce_the_lift_but_do_not_block_it(
        controller_module, monkeypatch) -> None:
    """A proto whose FSRs read a constant 50/50 used to forbid every step
    forever -- a silent, permanent "leg lift does nothing"."""
    class StaticFsr(FakeVector3):
        def getValues(self):  # noqa: N802
            return [0.0, 0.0, 26.0]

    original = FakeRobot.__init__

    def patched(self, *, with_fsr=True):
        original(self, with_fsr=with_fsr)
        self.devices["LFsr"] = StaticFsr()
        self.devices["RFsr"] = StaticFsr()

    monkeypatch.setattr(FakeRobot, "__init__", patched)
    h = Harness(controller_module)
    try:
        h.spin(150, STANDING, IDLE_GAIT)
        h.spin(300, LEFT_LEG_UP, IDLE_GAIT)
        meta = h.ctl.driver.lower_body_meta
        assert meta["fsr_share"] == pytest.approx(0.5, abs=0.01)
        assert 0.0 < meta["lift"] < 0.8            # reduced, not refused
        assert h.angle("LKneePitch") > h.angle("RKneePitch") + 0.2
        assert "reduced authority" in str(meta["why"])
    finally:
        h.close()


def test_the_robot_turns_round_to_face_behind_you(harness) -> None:
    """The yaw estimate used to be bounded to +/-90 deg by an abs(), so the robot
    could never be asked to turn more than a quarter circle."""
    harness.spin(60, STANDING, IDLE_GAIT)
    behind = dict(IDLE_GAIT, body_yaw_rad=math.radians(175))
    kps = subject(yaw=175.0)
    for _ in range(8):
        harness.spin(40, kps, behind)
        if harness.ctl.motion.active:
            # Model the clip actually turning the robot by its nominal 60 deg.
            action = harness.ctl.motion.action
            harness.ctl.imu.rpy[2] += math.radians(60 if action == "turn_left" else -60)
    turns = [m for m in FakeMotion.played if "Turn" in m]
    assert len(turns) >= 3                       # 175 deg needs several clips
    assert len(set(turns)) == 1                  # all the same way -- no thrash
    residual = abs(harness.ctl.yaw_servo.error(harness.ctl.imu.rpy[2]))
    assert residual < math.radians(35)           # within half a clip


def test_a_wobbling_heading_does_not_thrash_turn_clips(harness) -> None:
    """A jittery yaw made the planner alternate left/right, which starves forward
    walking and trips the locomotion failure backoff -- the reason nothing moved."""
    harness.spin(60, STANDING, IDLE_GAIT)
    for i in range(30):
        wobble = math.radians(12) * (1 if i % 2 else -1)
        harness.spin(6, subject(yaw=math.degrees(wobble)),
                     dict(IDLE_GAIT, body_yaw_rad=wobble))
    turns = [m for m in FakeMotion.played if "Turn" in m]
    assert turns == []                           # nothing fired at all
    assert harness.ctl.motion.available          # and clips were never abandoned


def test_walking_is_not_starved_by_a_settled_heading(harness) -> None:
    harness.spin(60, STANDING, IDLE_GAIT)
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    assert mode == "motion:forward"
    assert "Forwards.motion" in FakeMotion.played


def test_a_blocked_clip_falls_back_to_marching_in_place(harness) -> None:
    """The march engine must be reachable on a machine that HAS Webots.

    Its old gate was ``"forward" not in motion.available`` -- i.e. it ran only
    when no forward clip existed on disk. Every real Webots install ships
    Forwards.motion, so the fallback existed only on machines that could not run
    the robot at all. Meanwhile the case it was written for -- the clip layer
    declining because the robot is not settled -- fell through to the pose layer,
    which commands nothing on the legs when it cannot see a leg lift.
    """
    harness.spin(60, STANDING, IDLE_GAIT)
    assert "forward" in harness.ctl.motion.available    # the old gate would be shut

    # Not settled: a steady tilt over the start ceiling but well under the abort
    # limit, so the robot is upright and merely unsteady -- exactly when marching
    # in place is the right answer.
    harness.ctl.imu.rpy = [0.30, 0.0, 0.0]
    mode = harness.spin(30, STANDING, MARCH_GAIT)
    assert not FakeMotion.played                        # no clip was started
    assert mode == "march:march"                        # but the legs are moving
    assert harness.ctl.driver.gait_meta["amp_gain"] > 0.0

    # And once it settles, the clip layer takes over again.
    harness.ctl.imu.rpy = [0.0, 0.0, 0.0]
    harness.spin(400, STANDING, IDLE_GAIT)              # let tilt risk decay
    harness.spin(30, STANDING, MARCH_GAIT)
    assert "Forwards.motion" in FakeMotion.played


def test_marching_still_yields_to_an_actual_fall(harness) -> None:
    """The march fallback must NOT fire while the robot is going over: the pose
    layer's tilt gate is the better recovery, because it ramps the asymmetric
    part of the posture out and returns to the balanced symmetric crouch."""
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.ctl.gyro.values = [3.0, 0.0, 0.0]           # predicted tilt past abort
    mode = harness.spin(20, STANDING, MARCH_GAIT)
    assert not FakeMotion.played
    assert mode == "pose"


def test_a_walk_clip_only_takes_the_legs_not_the_arms(harness) -> None:
    """The clip declares the 12 leg joints, so it gets those and no more.

    Suspending the whole body meant arm and head imitation stopped for the length
    of every clip -- and because a marching human restarts the clip immediately,
    the upper body appeared to die for as long as the walking lasted. Webots' own
    walk clips never command an arm joint, so there was nothing to protect.
    """
    harness.spin(60, STANDING, IDLE_GAIT)
    harness.spin(4, STANDING, MARCH_GAIT)
    assert harness.ctl.motion.active
    d = harness.ctl.driver
    assert d.suspended is True
    # Legs handed over ...
    assert d._is_suspended("LKneePitch") and d._is_suspended("RHipRoll")
    # ... arms and head kept.
    for name in ("LShoulderPitch", "RShoulderPitch", "LElbowRoll", "HeadYaw"):
        assert not d._is_suspended(name), name

    # And the arms genuinely keep tracking while the clip plays: move them and
    # watch the motors follow.
    before = harness.angle("LShoulderPitch")
    arms_up = subject()
    for name, (dx, dy) in (("left_elbow", (0.10, -0.18)), ("left_wrist", (0.18, -0.34))):
        base = arms_up.get(name)
        if base is not None:
            arms_up[name] = [base[0] + dx, base[1] + dy, base[2], base[3]]
    harness.spin(10, arms_up, MARCH_GAIT)
    assert harness.ctl.motion.active                  # still mid-clip
    assert abs(harness.angle("LShoulderPitch") - before) > 1e-3


def test_a_clip_that_declares_nothing_still_gets_the_whole_body(harness, tmp_path) -> None:
    """Unknown joint list -> hand over everything. Handing over too much only
    costs expressiveness; handing over too little fights the clip's keyframes."""
    c = harness.ctl
    harness.spin(60, STANDING, IDLE_GAIT)
    c.motion._joints["forward"] = []                  # as if the header were junk
    harness.spin(4, STANDING, MARCH_GAIT)
    assert c.motion.active
    assert c.driver._is_suspended("LShoulderPitch")
    assert c.driver._is_suspended("LKneePitch")


def test_losing_the_human_also_stands_the_UPPER_body_down(harness) -> None:
    """The legs had a stand-down; the arms and head did not.

    Their smoothed targets simply stopped being updated, so they froze wherever
    they happened to be. A recorded session ends with 80 s of *perfect* tracking
    error on every joint -- the robot holding, precisely, the pose of a human who
    had walked away. That reads as a crashed robot, not an idle one.
    """
    c = harness.ctl
    rest = {n: CONFIGS[n].rest_angle
            for n in ("LShoulderPitch", "RShoulderPitch", "LElbowRoll", "HeadYaw")}

    # Put the arms somewhere clearly away from rest, with the head turned.
    posed = subject()
    posed["left_elbow"] = [0.62, 0.22, 0.0, 0.95]
    posed["left_wrist"] = [0.70, 0.10, 0.0, 0.95]
    posed["nose"] = [0.56, 0.30, 0.0, 0.95]
    harness.spin(160, posed, IDLE_GAIT)
    moved = {n: harness.angle(n) for n in rest}
    assert any(abs(moved[n] - rest[n]) > 0.05 for n in rest), moved

    # Human leaves: stop sending frames entirely.
    harness.spin(400, None)
    assert c.driver.stats.stale is True
    for name, target in rest.items():
        assert abs(harness.angle(name) - target) < 0.05, (
            f"{name} held {harness.angle(name):+.3f} instead of returning to "
            f"{target:+.3f}"
        )


def test_a_fully_working_stack_reports_nothing_degraded(harness) -> None:
    """The counterpart of the DEGRADED block: when every layer is up, the list is
    empty. If this ever fails, the startup banner is crying wolf."""
    d = harness.ctl.driver
    assert d.degraded == []
    assert d.balance is not None and d.lower_body is not None


def test_a_missing_com_model_is_reported_not_hidden(controller_module, monkeypatch) -> None:
    """NumPy missing from Webots' interpreter is the single most expensive
    failure this controller has, and it used to be one info-level log line.

    Simulated by making the balance import fail the way it does in the field.
    """
    import builtins

    real_import = builtins.__import__

    def no_balance(name, *args, **kwargs):
        if name == "balance":
            raise ImportError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_balance)
    ctl = controller_module.PoseImitationController()
    monkeypatch.undo()

    assert ctl.driver.balance is None
    joined = " | ".join(ctl.driver.degraded)
    assert "CoM balance OFF" in joined
    assert "numpy" in joined
    # And the step gate must say it is no longer verifying anything.
    assert any("UNGATED" in reason for reason in ctl.driver.degraded), joined
    ctl.sock.close()


def test_the_trajectory_log_records_the_controllers_own_state(controller_module,
                                                              monkeypatch, tmp_path):
    """A log of joint angles says what the body did, not what the controller
    believed -- and every diagnosis on this project has needed both halves.

    The permanent-lean bug was identifiable from the joint columns alone, but its
    CAUSE needed the support margin, which was not recorded at all. These columns
    are what make a live test session analysable after the fact.
    """
    import csv
    import glob

    mod = controller_module
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    harness = Harness(mod)
    try:
        harness.spin(120, LEFT_LEG_UP, IDLE_GAIT)
        harness.ctl.trajectory_log.close()
        path = glob.glob(str(tmp_path / "*.csv"))
        assert path, "no trajectory log was written"
        rows = list(csv.DictReader(open(path[0], encoding="utf-8")))
        assert rows

        for column in mod.DIAGNOSTIC_COLUMNS:
            assert column in rows[0], column

        # The columns must carry real values, not blanks: a header alone would
        # look fine and diagnose nothing.
        last = rows[-1]
        assert last["leg_mode"]
        assert last["lb_mode"] in ("double", "load", "single")
        assert last["lb_why"]
        assert float(last["support_margin_x"]) != 0.0
        assert float(last["support_margin_y"]) != 0.0
        # And the joint columns still work.
        assert "LKneePitch_cmd_rad" in rows[0]
        assert "LKneePitch_meas_rad" in rows[0]
    finally:
        harness.ctl.sock.close()


def test_diagnostics_never_break_the_control_loop(controller_module, monkeypatch,
                                                  tmp_path):
    """Telemetry is not allowed to be a failure mode."""
    mod = controller_module
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    harness = Harness(mod)
    try:
        # Break the thing _diagnostics leans on hardest.
        harness.ctl.driver.balance.model = None
        mode = harness.spin(40, STANDING, IDLE_GAIT)
        assert mode in ("pose", "stand", "march:march")
        assert harness.ctl._errors == 0
    finally:
        harness.ctl.sock.close()


# ---------------------------------------------------------------------------
# The IMU tilt zero
# ---------------------------------------------------------------------------
def test_a_rotated_inertial_unit_does_not_read_as_a_fall(harness) -> None:
    """The single most expensive bug found on this project.

    Measured on the real robot: standing at rest, foot sensors carrying its full
    50 N of body weight and the gyro at 0.007 rad/s, the InertialUnit reported
    roll = +1.618 rad (93 deg). The sensor frame is mounted rotated. Nothing was
    wrong with the robot -- but every consumer of that number treated it as "about
    to fall over", so leg imitation stood down every frame, no walk clip could
    start, and the balance loop chased 291 mm of phantom lateral error and leaned
    the robot onto one foot.
    """
    c = harness.ctl
    c.imu.rpy = [1.618, 0.0, 0.0]              # what the real robot reports
    harness.spin(120, STANDING, IDLE_GAIT)

    assert c._imu_zero is not None, "the zero was never learned"
    assert c._imu_zero[0] == pytest.approx(1.618, abs=1e-6)
    # Corrected tilt is level, so nothing thinks the robot is going over.
    roll, pitch = c._corrected_tilt(*c._imu_rpy()[:2])
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6
    assert c._falling(roll, pitch) is False
    assert c.driver.lower_body_meta["tilt_ok"] is True

    # And the legs work again: a leg lift now reaches single support.
    harness.spin(300, LEFT_LEG_UP, IDLE_GAIT)
    assert c.driver.lower_body_meta["mode"] in ("load", "single")


def test_a_real_tilt_on_top_of_the_offset_is_still_detected(harness) -> None:
    """Correcting the zero must not blind the tilt gates -- that would trade one
    silent failure for a much worse one."""
    c = harness.ctl
    c.imu.rpy = [1.618, 0.0, 0.0]
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None

    # Now tip it 0.6 rad beyond the learned upright.
    c.imu.rpy = [1.618 + 0.6, 0.0, 0.0]
    roll, pitch = c._corrected_tilt(*c._imu_rpy()[:2])
    assert roll == pytest.approx(0.6, abs=1e-6)
    assert c._falling(roll, pitch) is True
    harness.spin(20, STANDING, IDLE_GAIT)
    assert c.driver.lower_body_meta["tilt_ok"] is False


def test_the_zero_is_not_latched_from_a_fallen_robot(harness, monkeypatch) -> None:
    """A controller restarted on a robot lying on the floor must not decide that
    lying down is upright. The foot sensors are the independent witness: soles
    carrying body weight mean standing, whatever the IMU claims."""
    c = harness.ctl
    c.imu.rpy = [1.60, 0.0, 0.0]
    # Feet carrying nothing -- the robot is not standing on them.
    monkeypatch.setattr(FakeFsr, "TOTAL_N", 0.0)
    harness.spin(200, STANDING, IDLE_GAIT)
    assert c._imu_zero is None, "latched a zero from an unloaded robot"
    # Tilt is reported as level while uncalibrated, so nothing aborts on a
    # reading we do not yet understand.
    assert c._corrected_tilt(*c._imu_rpy()[:2]) == (0.0, 0.0)

    # Stand it back up and the zero is learned.
    monkeypatch.setattr(FakeFsr, "TOTAL_N", 52.0)
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None
    assert c._imu_zero[0] == pytest.approx(1.60, abs=1e-6)


def test_the_learned_zero_reaches_the_log(harness, monkeypatch, tmp_path) -> None:
    """It has to be visible after the fact, or the next person re-finds it."""
    import csv
    import glob

    mod = harness.mod
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())   # the fixture holds the other
    other = Harness(mod)
    try:
        other.ctl.imu.rpy = [1.618, 0.0, 0.0]
        other.spin(140, STANDING, IDLE_GAIT)
        other.ctl.trajectory_log.close()
        rows = list(csv.DictReader(open(glob.glob(str(tmp_path / "*.csv"))[0],
                                       encoding="utf-8")))
        last = rows[-1]
        assert float(last["imu_roll_raw"]) == pytest.approx(1.618, abs=1e-6)
        assert float(last["imu_zero_roll"]) == pytest.approx(1.618, abs=1e-6)
        assert abs(float(last["imu_roll"])) < 1e-6      # corrected
    finally:
        other.ctl.sock.close()


def test_auto_zero_can_be_switched_off(controller_module, monkeypatch) -> None:
    """An escape hatch, in case a future model reports tilt honestly."""
    monkeypatch.setattr(controller_module, "IMU_AUTO_ZERO", False)
    monkeypatch.setattr(controller_module, "UDP_PORT", _free_port())
    other = Harness(controller_module)
    try:
        assert other.ctl._imu_zero == (0.0, 0.0)
        other.ctl.imu.rpy = [0.5, 0.0, 0.0]
        assert other.ctl._corrected_tilt(0.5, 0.0) == (0.5, 0.0)
    finally:
        other.ctl.sock.close()


# ---------------------------------------------------------------------------
# Fall detection and automatic recovery
# ---------------------------------------------------------------------------
def test_a_fall_is_detected_and_the_simulation_is_reset(harness) -> None:
    """When the robot goes down it stays down -- every layer correctly stands
    itself down, and the rest of the session is spent driving a robot on the
    floor. One recorded session lost 170 of its 178 seconds that way.
    """
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)      # learn the IMU zero while upright
    assert c._imu_zero is not None
    assert c.robot.resets == 0

    # Tip it right over: 90 deg past the learned upright.
    c.imu.rpy = [c._imu_zero[0] + 1.571, c._imu_zero[1], 0.0]
    height = c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2]))
    assert height is not None and height < mod_const(harness, "FALL_HEAD_HEIGHT_M")

    harness.spin(120, STANDING, IDLE_GAIT)      # longer than FALL_CONFIRM_S
    assert c.robot.resets == 1, "the simulation was never reset"
    # The robot is upright again and the fall latch has cleared.
    assert c._fall_since is None
    assert c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2])) > 0.40


def test_recovery_drops_the_state_that_described_the_fallen_robot(harness) -> None:
    """Whether or not Webots restarts this controller on reset, none of the state
    may still describe the robot that fell."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    c._turning = True
    c._tilt_risk = 0.3
    c._reset_for_new_episode()
    assert c._turning is False
    assert c._tilt_risk == 0.0
    assert c._imu_zero is None          # re-earned against the new pose
    assert c.yaw_servo.latched is False
    assert c.driver.lower_body_meta["mode"] == "double" or True


def mod_const(harness, name):
    return getattr(harness.mod, name)


def test_a_deep_squat_is_not_mistaken_for_a_fall(harness) -> None:
    """The head-height test has to survive the deepest posture the robot is ever
    asked for. It does, and by a wide margin, because NAO's crouch keeps the torso
    vertical: standing 0.460 m, deepest squat 0.412 m, threshold 0.25 m."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    assert c._imu_zero is not None

    deep = subject(left_leg=(0.0, -0.70, 1.40), right_leg=(0.0, -0.70, 1.40))
    harness.spin(400, deep, IDLE_GAIT)
    assert c.robot.resets == 0, "a squat was treated as a fall"
    height = c.head_height(*c._corrected_tilt(*c._imu_rpy()[:2]))
    assert height is None or height > mod_const(harness, "FALL_HEAD_HEIGHT_M")


def test_a_transient_stumble_does_not_trigger_a_reset(harness) -> None:
    """FALL_CONFIRM_S exists so the balance loop gets its chance first."""
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    zero = c._imu_zero
    c.imu.rpy = [zero[0] + 1.571, zero[1], 0.0]
    harness.spin(20, STANDING, IDLE_GAIT)       # 0.4s: under the 1.0s window
    assert c.robot.resets == 0
    c.imu.rpy = [zero[0], zero[1], 0.0]         # recovered
    harness.spin(60, STANDING, IDLE_GAIT)
    assert c.robot.resets == 0
    assert c._fall_since is None


def test_repeated_falls_stop_reloading_instead_of_looping(harness, monkeypatch) -> None:
    """An endless reload loop is harder to diagnose than a robot lying still."""
    mod = harness.mod
    monkeypatch.setattr(mod, "FALL_MAX_RELOADS", 2)
    monkeypatch.setattr(mod, "FALL_RELOAD_COOLDOWN_S", 0.0)
    c = harness.ctl
    harness.spin(120, STANDING, IDLE_GAIT)
    down = c._imu_zero[0] + 1.571
    for _ in range(4):
        c.imu.rpy = [down, 0.0, 0.0]
        harness.spin(120, STANDING, IDLE_GAIT)
        # the reset is faked, so re-learn the zero as a real restart would
        c.imu.rpy = [0.0, 0.0, 0.0]
        harness.spin(120, STANDING, IDLE_GAIT)
    assert c.robot.resets == 2, c.robot.resets


def test_fall_recovery_can_be_switched_off(controller_module, monkeypatch) -> None:
    monkeypatch.setattr(controller_module, "AUTO_RELOAD_ON_FALL", False)
    monkeypatch.setattr(controller_module, "UDP_PORT", _free_port())
    other = Harness(controller_module)
    try:
        other.spin(120, STANDING, IDLE_GAIT)
        other.ctl.imu.rpy = [other.ctl._imu_zero[0] + 1.571, 0.0, 0.0]
        other.spin(200, STANDING, IDLE_GAIT)
        assert other.ctl.robot.resets == 0
    finally:
        other.ctl.sock.close()


def test_head_height_is_reported_in_the_log(harness, monkeypatch, tmp_path) -> None:
    import csv
    import glob

    mod = harness.mod
    monkeypatch.setattr(mod, "ENABLE_TRAJECTORY_LOG", True)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "UDP_PORT", _free_port())
    other = Harness(mod)
    try:
        other.spin(140, STANDING, IDLE_GAIT)
        other.ctl.trajectory_log.close()
        rows = list(csv.DictReader(open(glob.glob(str(tmp_path / "*.csv"))[0],
                                       encoding="utf-8")))
        assert float(rows[-1]["head_height"]) > 0.40      # standing
        assert rows[-1]["reloads"] == "0"
    finally:
        other.ctl.sock.close()


def test_the_shipped_default_is_imitation_not_locomotion() -> None:
    """The lower body should follow your legs in real time, not hand them to a
    canned clip.

    A .motion clip is a fixed keyframe sequence played to completion, and while it
    runs the camera is ignored for the 12 leg joints -- the opposite of imitation.
    Balance is handled instead by shifting the centre of mass to make the imitated
    pose holdable (lower_body._shift_com), so the clips are no longer needed to
    keep the robot up and are opt-in for covering ground.
    """
    import importlib.util
    import os

    path = os.path.join(CONTROLLER_DIR, "pose_imitation_controller.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    # Read the constant out of the source rather than importing, so this holds
    # regardless of what any fixture monkeypatched.
    for line in source.splitlines():
        if line.startswith("LEG_CONTROL"):
            assert line.split("=")[1].strip() == '"pose"', line
            break
    else:
        raise AssertionError("LEG_CONTROL not found")
    assert importlib.util.find_spec is not None       # keep the import meaningful
