"""Tests for NAO walk-motion discovery and selection (main/libraries/walk_motion.py).

These cover the pure, Webots-free logic: finding motion files on disk (with
filename fallbacks and search-dir ordering) and mapping a gait command to a walk
action. The actual Webots Motion playback is exercised on the test machine.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from walk_motion import (  # noqa: E402
    default_motion_search_dirs,
    find_motion_files,
    select_action,
)


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


def test_select_action_forward_when_marching() -> None:
    assert select_action(_gait()) == "forward"


def test_select_action_none_when_idle_or_unconfident_or_stopped() -> None:
    assert select_action(None) is None
    assert select_action(_gait(state="idle")) is None
    assert select_action(_gait(cadence=0.0)) is None
    assert select_action(_gait(conf=0.2)) is None


def test_select_action_turns_on_strong_turn_cue() -> None:
    assert select_action(_gait(turn=0.6)) == "turn_left"
    assert select_action(_gait(turn=-0.6)) == "turn_right"
    # weak turn cue stays forward
    assert select_action(_gait(turn=0.1)) == "forward"


# ---------------------------------------------------------------------------
# Yaw servo / locomotion planning
# ---------------------------------------------------------------------------
import math  # noqa: E402

from walk_motion import (  # noqa: E402
    STAND,
    LocomotionParams,
    YawServo,
    motion_nominal_yaw,
    plan_action,
    wrap_pi,
)

CLIPS = {
    "forward": "/w/Forwards.motion",
    "turn_left": "/w/TurnLeft60.motion",
    "turn_right": "/w/TurnRight60.motion",
}


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


def test_yaw_servo_tracks_a_relative_rotation() -> None:
    servo = YawServo()
    # Latching zeroes the error: the subject's and the robot's initial headings
    # are both arbitrary, so only the change matters.
    servo.update(human_yaw=0.3, conf=1.0, robot_yaw=-2.0, now_s=0.0)
    assert servo.latched
    assert abs(servo.error(-2.0)) < 1e-9
    # Human turns 0.5 rad -> the robot is asked to turn the same way.
    servo.update(human_yaw=0.8, conf=1.0, robot_yaw=-2.0, now_s=0.1)
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
    servo.update(human_yaw=0.0, conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.update(human_yaw=1.0, conf=1.0, robot_yaw=0.0, now_s=0.1)
    assert abs(servo.error(0.0) - 1.0) < 1e-9
    # Subject walks off and comes back facing somewhere else entirely: chasing
    # the stale error would spin the robot for no reason.
    servo.update(human_yaw=-1.0, conf=1.0, robot_yaw=0.0, now_s=10.0)
    assert abs(servo.error(0.0)) < 1e-9


def test_yaw_servo_sign_flips_the_mapping() -> None:
    servo = YawServo(sign=-1.0)
    servo.update(human_yaw=0.0, conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.update(human_yaw=0.4, conf=1.0, robot_yaw=0.0, now_s=0.1)
    assert abs(servo.error(0.0) + 0.4) < 1e-9


def test_yaw_servo_wraps_across_the_discontinuity() -> None:
    servo = YawServo()
    servo.update(human_yaw=0.0, conf=1.0, robot_yaw=3.0, now_s=0.0)
    servo.update(human_yaw=0.4, conf=1.0, robot_yaw=3.0, now_s=0.1)
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
    servo.update(human_yaw=0.2, conf=1.0, robot_yaw=0.0, now_s=0.0)
    servo.reset()
    assert not servo.latched and servo.error(1.0) == 0.0
