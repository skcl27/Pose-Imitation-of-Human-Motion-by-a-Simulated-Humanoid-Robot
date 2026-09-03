"""Tests for NAO walk-motion discovery and selection (main/libraries/walk_motion.py).

These cover the pure, Webots-free logic: finding motion files on disk (with
filename fallbacks and search-dir ordering) and mapping a gait command to a walk
action. The actual Webots Motion playback is exercised on the test machine.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from walk_motion import (  # noqa: E402
    STAND,
    LocomotionParams,
    YawServo,
    default_motion_search_dirs,
    find_motion_files,
    motion_joints,
    motion_nominal_yaw,
    plan_action,
    wrap_pi,
)

CLIPS = {
    "forward": "/w/Forwards.motion",
    "turn_left": "/w/TurnLeft60.motion",
    "turn_right": "/w/TurnRight60.motion",
}


def _touch(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#WEBOTS_MOTION,V1.0\n")


def test_find_motion_files_picks_first_existing_candidate(tmp_path) -> None:
    d = tmp_path / "motions"
    d.mkdir()
    _touch(str(d / "Forwards.motion"))
    _touch(str(d / "TurnLeft40.motion"))
    _touch(str(d / "TurnRight40.motion"))
    found = find_motion_files([str(d)])
    assert found["forward"].endswith("Forwards.motion")
    assert found["turn_left"].endswith("TurnLeft40.motion")
    assert found["turn_right"].endswith("TurnRight40.motion")
    # No SideStep / Backwards files present -> those actions are omitted.
    assert "backward" not in found
    assert "side_left" not in found


def test_find_motion_files_respects_search_dir_order(tmp_path) -> None:
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    _touch(str(d2 / "Forwards.motion"))
    _touch(str(d1 / "Forwards.motion"))
    found = find_motion_files([str(d1), str(d2)])
    # d1 comes first in the search order, so its file wins.
    assert found["forward"] == str(d1 / "Forwards.motion")


def test_find_motion_files_empty_when_nothing_present(tmp_path) -> None:
    assert find_motion_files([str(tmp_path)]) == {}


def test_default_search_dirs_include_webots_home(monkeypatch) -> None:
    monkeypatch.setenv("WEBOTS_HOME", "/opt/webots-test")
    dirs = default_motion_search_dirs()
    assert any(d.startswith("/opt/webots-test") and d.endswith("motions") for d in dirs)
    # extra dirs are searched first.
    dirs2 = default_motion_search_dirs(extra=["/repo/motions"])
    assert dirs2[0] == "/repo/motions"


def test_default_search_dirs_dedup() -> None:
    dirs = default_motion_search_dirs(extra=["/x", "/x"])
    assert dirs.count("/x") == 1


def _gait(state="march", cadence=1.0, conf=0.9, turn=0.0):
    return {"state": state, "cadence_hz": cadence, "conf": conf, "turn": turn}


def test_motion_nominal_yaw_reads_the_angle_off_the_filename() -> None:
    assert abs(motion_nominal_yaw("/w/TurnLeft60.motion") - math.radians(60)) < 1e-9
    assert abs(motion_nominal_yaw("/w/TurnRight40.motion") + math.radians(40)) < 1e-9
    # Unnumbered clips get a documented default rather than 0 (0 would disable
    # the overshoot guard entirely).
    assert motion_nominal_yaw("/w/TurnLeft.motion") > 0.0
    assert motion_nominal_yaw("/w/Forwards.motion") == 0.0
    assert motion_nominal_yaw(None) == 0.0


def test_plan_walks_forward_when_marching_and_aligned() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=0.0, gait=gait, available=CLIPS).action == "forward"


def test_plan_stands_still_when_idle_and_aligned() -> None:
    assert plan_action(yaw_error_rad=0.0, gait=None, available=CLIPS) == STAND
    idle = {"state": "idle", "cadence_hz": 0.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=0.0, gait=idle, available=CLIPS).action is None


def test_plan_turns_while_standing_still() -> None:
    """The whole point of the yaw servo: rotating in front of the camera has to
    move the robot's body, not just its head, with no marching involved."""
    assert plan_action(yaw_error_rad=1.2, gait=None, available=CLIPS).action == "turn_left"
    assert plan_action(yaw_error_rad=-1.2, gait=None, available=CLIPS).action == "turn_right"


def test_turning_takes_priority_over_walking_forward() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    plan = plan_action(yaw_error_rad=1.2, gait=gait, available=CLIPS)
    assert plan.action == "turn_left" and plan.is_turn


def test_plan_refuses_a_clip_that_would_overshoot() -> None:
    # A 60 deg clip must not be fired at a 15 deg error (it would leave a bigger
    # error, of the opposite sign, than it started with).
    plan = plan_action(yaw_error_rad=math.radians(15), gait=None, available=CLIPS)
    assert plan.action is None
    # A 28 deg error is past the entry gate, yet still too small for the 60 deg
    # clip -- and exactly right for a 40 deg one.
    small = dict(CLIPS, turn_left="/w/TurnLeft40.motion")
    assert plan_action(yaw_error_rad=math.radians(28), gait=None,
                       available=CLIPS).action is None
    assert plan_action(yaw_error_rad=math.radians(28), gait=None,
                       available=small).action == "turn_left"


def test_plan_never_asks_for_a_clip_that_is_not_on_disk() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=1.2, gait=gait, available={}).action is None
    only_turn = {"turn_left": "/w/TurnLeft60.motion"}
    assert plan_action(yaw_error_rad=0.0, gait=gait, available=only_turn).action is None


def test_plan_hysteresis_lowers_the_gate_once_turning() -> None:
    # A fine-grained clip, so the overshoot guard is not what decides this test.
    small = {"turn_left": "/w/TurnLeft20.motion"}
    p = LocomotionParams()
    err = 0.5 * (p.turn_stop_rad + p.turn_start_rad)   # between the two gates
    assert plan_action(yaw_error_rad=err, available=small, params=p).action is None
    # Mid-rotation the gate drops, so the robot finishes the turn instead of
    # stalling one clip short of facing the right way.
    assert plan_action(yaw_error_rad=err, available=small, params=p,
                       turning=True).action == "turn_left"


def test_plan_ignores_a_non_finite_yaw_error() -> None:
    gait = {"state": "march", "cadence_hz": 1.0, "conf": 0.9}
    assert plan_action(yaw_error_rad=float("nan"), gait=gait,
                       available=CLIPS).action == "forward"


def test_plan_requires_confident_marching_to_walk() -> None:
    weak = {"state": "march", "cadence_hz": 1.0, "conf": 0.2}
    assert plan_action(yaw_error_rad=0.0, gait=weak, available=CLIPS).action is None
    slow = {"state": "march", "cadence_hz": 0.01, "conf": 0.9}
    assert plan_action(yaw_error_rad=0.0, gait=slow, available=CLIPS).action is None


def _latch(servo, human, robot, *, t0=0.0, n=8, dt=0.15):
    """Feed the burst of frames YawServo needs before it latches a reference.

    The servo deliberately no longer latches on a single frame: body yaw is the
    noisiest cue in the pipeline, and taking one arbitrary frame as the origin is
    how the whole heading loop ends up with a fixed offset (measured: the hip-yaw
    bias pinned to one side in 81% of a recorded session). It latches the median
    of a short burst instead, so tests have to supply one.

    Returns the timestamp just after the latch, so callers can carry on.
    """
    t = t0
    for _ in range(n):
        servo.update(human_yaw=human, conf=1.0, robot_yaw=robot, now_s=t)
        t += dt
    assert servo.latched, "burst should have latched the reference"
    return t


def test_yaw_servo_tracks_a_relative_rotation() -> None:
    servo = YawServo()
    # Latching zeroes the error: the subject's and the robot's initial headings
    # are both arbitrary, so only the change matters.
    t = _latch(servo, 0.3, -2.0)
    assert abs(servo.error(-2.0)) < 1e-9
    # Human turns 0.5 rad -> the robot is asked to turn the same way.
    servo.update(human_yaw=0.8, conf=1.0, robot_yaw=-2.0, now_s=t)
    assert abs(servo.error(-2.0) - 0.5) < 1e-9
    # ... and the error closes as the robot actually gets there.
    assert abs(servo.error(-1.5)) < 1e-9


def test_yaw_servo_ignores_unusable_measurements() -> None:
    servo = YawServo()
    servo.update(human_yaw=None, conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.update(human_yaw=float("nan"), conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.update(human_yaw=0.5, conf=0.1, robot_yaw=0.0, now_s=0.0)
    assert not servo.latched
    assert servo.error(0.0) == 0.0


def test_yaw_servo_relatches_after_losing_the_subject() -> None:
    servo = YawServo(relatch_after_s=2.0)
    t = _latch(servo, 0.0, 0.0)
    servo.update(human_yaw=1.0, conf=1.0, robot_yaw=0.0, now_s=t)
    assert abs(servo.error(0.0) - 1.0) < 1e-9
    # Subject walks off and comes back facing somewhere else entirely: chasing
    # the stale error would spin the robot for no reason. The gap drops the old
    # reference immediately; the new one comes from a fresh burst.
    servo.update(human_yaw=-1.0, conf=1.0, robot_yaw=0.0, now_s=t + 10.0)
    assert not servo.latched
    assert servo.error(0.0) == 0.0
    _latch(servo, -1.0, 0.0, t0=t + 10.0)
    assert abs(servo.error(0.0)) < 1e-9


def test_yaw_servo_sign_flips_the_mapping() -> None:
    servo = YawServo(sign=-1.0)
    t = _latch(servo, 0.0, 0.0)
    servo.update(human_yaw=0.4, conf=1.0, robot_yaw=0.0, now_s=t)
    assert abs(servo.error(0.0) + 0.4) < 1e-9


def test_yaw_servo_wraps_across_the_discontinuity() -> None:
    servo = YawServo()
    t = _latch(servo, 0.0, 3.0)
    servo.update(human_yaw=0.4, conf=1.0, robot_yaw=3.0, now_s=t)
    # desired = 3.4 rad, which wraps past pi; the error must stay small and
    # correctly signed instead of demanding a near-full turn the other way.
    assert abs(servo.error(3.0) - 0.4) < 1e-9
    assert abs(servo.error(3.4 - 2 * math.pi)) < 1e-9


def test_wrap_pi() -> None:
    for angle in (3 * math.pi, -3 * math.pi, 5 * math.pi):
        assert abs(abs(wrap_pi(angle)) - math.pi) < 1e-9
    assert abs(wrap_pi(0.5) - 0.5) < 1e-12
    assert abs(wrap_pi(2 * math.pi + 0.25) - 0.25) < 1e-9
    assert -math.pi <= wrap_pi(123.456) < math.pi


def test_reset_unlatches() -> None:
    servo = YawServo()
    _latch(servo, 0.2, 0.0)
    servo.reset()
    assert not servo.latched and servo.error(1.0) == 0.0


# ---------------------------------------------------------------------------
# Turning through large angles, and noise immunity
# ---------------------------------------------------------------------------
def _converge(target_yaw, *, clip="/w/TurnLeft60.motion", steps=40, noise=0.0):
    """Settle facing forward, then have the human turn to ``target_yaw``.

    The settle phase matters: :class:`YawServo` latches the human/robot heading
    pair on first sight, so a *constant* offset is absorbed by the latch and is
    not an error at all -- only a change from the latched reference is. The robot
    is then turned by discrete clips of the size the filename implies.

    Returns ``(final |error| rad, clips played, left/right direction flips)``.
    """
    clips = {"turn_left": clip, "turn_right": clip.replace("Left", "Right")}
    nominal = abs(motion_nominal_yaw(clip))
    servo = YawServo()
    robot = 0.0
    played = flips = 0
    previous = None
    turning = False
    t = 0.0
    for _ in range(5):                      # settle: human faces forward
        servo.update(human_yaw=0.0, conf=1.0, robot_yaw=robot, now_s=t)
        t += 0.1
    for i in range(steps):                  # then they turn, with some wobble
        wobble = noise * (1 if i % 2 else -1)
        servo.update(human_yaw=target_yaw + wobble, conf=1.0,
                     robot_yaw=robot, now_s=t)
        t += 0.1
        plan = plan_action(yaw_error_rad=servo.error(robot),
                           available=clips, turning=turning)
        turning = plan.is_turn
        if plan.is_turn:
            played += 1
            if previous and previous != plan.action:
                flips += 1
            previous = plan.action
            robot += nominal * (1 if plan.action == "turn_left" else -1)
    return abs(wrap_pi(servo.error(robot))), played, flips


def test_the_servo_turns_the_robot_all_the_way_round() -> None:
    """Half a turn takes several 60 deg clips; it must converge, not stall."""
    for target in (math.radians(150), math.radians(180), -math.radians(170)):
        err, played, flips = _converge(target)
        assert played >= 2, target
        assert err <= math.radians(35), (target, math.degrees(err))
        assert flips == 0, (target, flips)


def test_the_servo_converges_for_any_target_heading() -> None:
    for degrees in range(-180, 181, 15):
        err, _, flips = _converge(math.radians(degrees))
        assert err <= math.radians(35), (degrees, math.degrees(err))
        assert flips <= 1, (degrees, flips)


def test_the_servo_takes_the_short_way_round() -> None:
    """A 170 deg target must not be chased the 190 deg way."""
    _, played_left, _ = _converge(math.radians(170))
    _, played_right, _ = _converge(-math.radians(170))
    assert played_left == played_right          # symmetric effort
    assert played_left <= 4                     # 170/60 -> 3 clips, not 4+


def test_measurement_noise_does_not_thrash_the_turn_direction() -> None:
    """The real failure: a jittery yaw made the planner alternate left/right,
    which starves forward walking and trips the locomotion failure backoff."""
    err, played, flips = _converge(0.0, noise=math.radians(12), steps=60)
    assert flips == 0
    assert played == 0            # 12 deg of wobble is below the turn gate
    assert err <= math.radians(15)


def test_a_small_turn_does_not_start_a_clip() -> None:
    """Shifting 20 deg is below the gate: firing a 60 deg clip would leave a
    bigger error, of the opposite sign, than it started with."""
    err, played, _ = _converge(math.radians(20), steps=80)
    assert played == 0
    assert err == pytest.approx(math.radians(20), abs=1e-6)


def test_a_constant_offset_is_absorbed_by_the_latch() -> None:
    """Standing habitually a little off-square is not an error to correct: the
    servo tracks rotation *relative* to where it first saw you."""
    servo = YawServo()
    for i in range(10):
        servo.update(human_yaw=math.radians(25), conf=1.0,
                     robot_yaw=0.0, now_s=i * 0.1)
    assert servo.error(0.0) == pytest.approx(0.0, abs=1e-9)
    assert plan_action(yaw_error_rad=servo.error(0.0),
                       available={"turn_left": "/w/TurnLeft60.motion"}).action is None


def test_yaw_servo_latch_ignores_a_spiking_frame() -> None:
    """The reason the latch is a median of a burst rather than one frame.

    The body-yaw cue is documented to spike to its +/-90 deg bound on a minority
    of frames while the subject stands square to the camera. If such a frame is
    the one that sets the origin, every subsequent error inherits the offset --
    which is exactly the fixed one-sided bias measured in a recorded session.
    """
    servo = YawServo()
    t = 0.0
    for i in range(9):
        # One frame in three is a bogus +90 deg spike; the truth is 0.0.
        human = math.pi / 2 if i % 3 == 0 else 0.0
        servo.update(human_yaw=human, conf=1.0, robot_yaw=0.0, now_s=t)
        t += 0.15
    assert servo.latched
    # The median rejected the spikes, so standing still is not a heading error.
    servo.update(human_yaw=0.0, conf=1.0, robot_yaw=0.0, now_s=t)
    assert abs(servo.error(0.0)) < 1e-9


def test_yaw_servo_diagnostics_separate_offset_from_tracking() -> None:
    servo = YawServo()
    t = _latch(servo, 0.2, 1.0)
    servo.update(human_yaw=0.9, conf=1.0, robot_yaw=1.0, now_s=t)
    d = servo.diagnostics(1.0)
    assert d["latched"] == 1.0
    assert d["human_ref"] == pytest.approx(0.2)
    assert d["robot_ref"] == pytest.approx(1.0)
    assert d["human_now"] == pytest.approx(0.9)
    assert d["error"] == pytest.approx(0.7, abs=1e-9)


# ---------------------------------------------------------------------------
# Which joints a clip actually drives
# ---------------------------------------------------------------------------
def test_motion_joints_reads_the_clip_header(tmp_path) -> None:
    """Webots' NAO walk clips drive only the legs. Knowing that is what lets the
    controller keep the arms and head imitating while the robot walks, instead of
    suspending the whole body for the length of every step."""
    path = tmp_path / "Forwards.motion"
    path.write_text(
        "#WEBOTS_MOTION,V1.0,LHipYawPitch,LHipRoll,LHipPitch,LKneePitch,"
        "LAnklePitch,LAnkleRoll,RHipYawPitch,RHipRoll,RHipPitch,RKneePitch,"
        "RAnklePitch,RAnkleRoll\n"
        "00:00:000,Pose1,0,0.027,-0.505,1.042,-0.537,-0.027,0,0.027,-0.505,"
        "1.042,-0.537,-0.027\n",
        encoding="utf-8",
    )
    joints = motion_joints(str(path))
    assert len(joints) == 12
    assert "LHipPitch" in joints and "RAnkleRoll" in joints
    # Crucially, no arm or head joint is in there.
    assert not [j for j in joints if "Shoulder" in j or "Elbow" in j or "Head" in j]


def test_motion_joints_returns_empty_for_anything_it_cannot_read(tmp_path) -> None:
    """An empty list means "unknown", and the caller must then hand over the
    whole body -- handing over too little would fight the clip's keyframes."""
    assert motion_joints(None) == []
    assert motion_joints(str(tmp_path / "missing.motion")) == []
    junk = tmp_path / "junk.motion"
    junk.write_text("not a motion file\n", encoding="utf-8")
    assert motion_joints(str(junk)) == []
    bare = tmp_path / "bare.motion"
    bare.write_text("#WEBOTS_MOTION,V1.0\n", encoding="utf-8")
    assert motion_joints(str(bare)) == []
