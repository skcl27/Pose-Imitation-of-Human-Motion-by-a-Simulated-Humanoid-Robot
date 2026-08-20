"""Tests for monocular gait-cue extraction (src/perception/gait_cues.py).

The extractor must turn a stream of 2D keypoints into a stable gait command:
detect marching cadence under non-uniform frame timing, reject arm-swing
aliasing, report idle when the legs are still or out of frame, and stay
scale-invariant as the subject moves toward/away from the camera.
"""
from __future__ import annotations

import math

from src.perception.gait_cues import GaitCueExtractor
from src.type_defs import Keypoint, PoseFrame

TWO_PI = 2.0 * math.pi


def _march_frame(
    t: float,
    idx: int,
    *,
    freq_hz: float = 1.0,
    scale: float = 1.0,
    knee_amp: float = 0.10,
    cx: float = 0.5,
    leg_vis: float = 1.0,
) -> PoseFrame:
    """A synthetic frontal marcher: shoulders span ``scale``; knees oscillate
    anti-phase by ``knee_amp`` (image y is DOWN, raised knee -> smaller y)."""
    phase = TWO_PI * freq_hz * t
    hip_y = 0.5
    knee_base_y = 0.7
    left_knee_y = knee_base_y - knee_amp * math.sin(phase)   # up when sin>0
    right_knee_y = knee_base_y + knee_amp * math.sin(phase)  # anti-phase
    kps = {
        "left_shoulder": Keypoint(cx - scale / 2, 0.2, 0.0, 1.0),
        "right_shoulder": Keypoint(cx + scale / 2, 0.2, 0.0, 1.0),
        "left_hip": Keypoint(cx - 0.2 * scale, hip_y, 0.0, leg_vis),
        "right_hip": Keypoint(cx + 0.2 * scale, hip_y, 0.0, leg_vis),
        "left_knee": Keypoint(cx - 0.2 * scale, left_knee_y, 0.0, leg_vis),
        "right_knee": Keypoint(cx + 0.2 * scale, right_knee_y, 0.0, leg_vis),
        "left_ankle": Keypoint(cx - 0.2 * scale, 0.9, 0.0, leg_vis),
        "right_ankle": Keypoint(cx + 0.2 * scale, 0.9, 0.0, leg_vis),
    }
    return PoseFrame(timestamp_s=t, keypoints=kps, frame_index=idx)


def _still_frame(t: float, idx: int, *, arm_swing: bool = False) -> PoseFrame:
    """A standing person; optionally swinging arms (legs perfectly still)."""
    wrist_y = 0.5 + (0.1 * math.sin(TWO_PI * 1.0 * t) if arm_swing else 0.0)
    kps = {
        "left_shoulder": Keypoint(0.4, 0.2, 0.0, 1.0),
        "right_shoulder": Keypoint(0.6, 0.2, 0.0, 1.0),
        "left_hip": Keypoint(0.45, 0.5, 0.0, 1.0),
        "right_hip": Keypoint(0.55, 0.5, 0.0, 1.0),
        "left_knee": Keypoint(0.45, 0.7, 0.0, 1.0),
        "right_knee": Keypoint(0.55, 0.7, 0.0, 1.0),
        "left_wrist": Keypoint(0.4, wrist_y, 0.0, 1.0),
        "right_wrist": Keypoint(0.6, wrist_y, 0.0, 1.0),
    }
    return PoseFrame(timestamp_s=t, keypoints=kps, frame_index=idx)


def _feed(ex: GaitCueExtractor, frame_fn, *, duration_s: float, start_t: float = 0.0):
    """Feed frames with deliberately NON-uniform dt (mimics AdaptiveFPSController)."""
    t = start_t
    idx = 0
    last = None
    while t < start_t + duration_s:
        last = ex.update(frame_fn(t, idx))
        dt = 0.03 if idx % 2 == 0 else 0.05  # ~25-33 fps, jittered
        t += dt
        idx += 1
    return last


def test_detects_marching_cadence_under_jittered_timing() -> None:
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=4.0)
    assert cmd.state == "march"
    assert 0.7 <= cmd.cadence_hz <= 1.3  # ~1 Hz despite non-uniform dt
    assert cmd.conf >= 0.9
    assert 0.0 <= cmd.phase < TWO_PI
    assert cmd.intensity > 0.0


def test_faster_march_reads_higher_cadence() -> None:
    slow = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=0.8), duration_s=5.0)
    fast = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.6), duration_s=5.0)
    assert fast.cadence_hz > slow.cadence_hz


def test_arm_swing_while_legs_still_is_rejected() -> None:
    # Vigorous arm swing, legs perfectly still -> must NOT read as marching.
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _still_frame(t, i, arm_swing=True), duration_s=4.0)
    assert cmd.state == "idle"
    assert cmd.cadence_hz == 0.0


def test_standing_still_is_idle() -> None:
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _still_frame(t, i, arm_swing=False), duration_s=3.0)
    assert cmd.state == "idle"


def test_low_leg_visibility_forces_idle() -> None:
    ex = GaitCueExtractor()
    cmd = _feed(ex, lambda t, i: _march_frame(t, i, leg_vis=0.1), duration_s=3.0)
    assert cmd.state == "idle"
    assert cmd.conf < 0.6


def test_cadence_is_scale_invariant() -> None:
    # Same march, subject near (large) vs far (small) from the camera.
    near = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.0, scale=1.6),
                 duration_s=4.0)
    far = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.0, scale=0.5),
                duration_s=4.0)
    assert near.state == "march" and far.state == "march"
    assert abs(near.cadence_hz - far.cadence_hz) < 0.25


def test_swing_side_flips_within_a_cycle() -> None:
    ex = GaitCueExtractor()
    # Warm up to "march", then sample swing_side across half a cycle.
    _feed(ex, lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=3.0)
    sides = set()
    t, idx = 3.0, 1000
    for _ in range(40):
        cmd = ex.update(_march_frame(t, idx, freq_hz=1.0))
        sides.add(cmd.swing_side)
        t += 0.03
        idx += 1
    assert 1 in sides and -1 in sides  # both knees lead at some point


def test_stops_promptly_when_marching_ceases() -> None:
    ex = GaitCueExtractor()
    _feed(ex, lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=4.0)
    # Now stand still; the command must fall back to idle quickly.
    cmd = _feed(ex, lambda t, i: _still_frame(t, i), duration_s=2.0, start_t=4.0)
    assert cmd.state == "idle"


def test_as_dict_is_json_friendly() -> None:
    import json
    cmd = _feed(GaitCueExtractor(), lambda t, i: _march_frame(t, i, freq_hz=1.0), duration_s=4.0)
    d = cmd.as_dict()
    json.dumps(d)  # must not raise
    assert set(d) == {
        "state", "cadence_hz", "phase", "swing_side", "intensity", "turn", "conf",
        "body_yaw_rad", "yaw_conf",
    }


# ---------------------------------------------------------------------------
# Torso yaw (body rotation)
# ---------------------------------------------------------------------------
def _yaw_frame(t: float, yaw_deg: float, idx: int = 0, vis: float = 1.0,
               span: float = 1.0) -> PoseFrame:
    """A subject standing still, rotated ``yaw_deg`` about the vertical axis.

    The shoulder and hip lines are body-fixed horizontal segments, so rotating
    the body shrinks their image-plane extent and separates their endpoints in
    depth -- exactly the projection the yaw solve inverts.
    """
    a = math.radians(yaw_deg)
    kps = {}
    # MediaPipe's convention, verified against recorded runs: for a subject
    # facing the camera the LEFT landmark sits on the image's RIGHT.
    for name, half, y in (("shoulder", 0.09 * span, 0.30), ("hip", 0.06 * span, 0.55)):
        for side, sgn in (("left", +1.0), ("right", -1.0)):
            kps[f"{side}_{name}"] = Keypoint(
                x=0.5 + sgn * half * math.cos(a),
                y=y,
                z=+sgn * half * math.sin(a),
                visibility=vis,
            )
    for side, sgn in (("left", -1.0), ("right", +1.0)):
        kps[f"{side}_knee"] = Keypoint(0.5 + sgn * 0.06, 0.75, 0.0, vis)
        kps[f"{side}_ankle"] = Keypoint(0.5 + sgn * 0.06, 0.92, 0.0, vis)
    return PoseFrame(timestamp_s=t, keypoints=kps, frame_index=idx)


def _settle_yaw(yaw_deg: float, frames: int = 40, vis: float = 1.0):
    ex = GaitCueExtractor()
    cmd = None
    for i in range(frames):
        cmd = ex.update(_yaw_frame(i / 30.0, yaw_deg, i, vis))
    return cmd


def test_facing_the_camera_is_zero_yaw() -> None:
    cmd = _settle_yaw(0.0)
    assert abs(cmd.body_yaw_rad) < 0.02
    assert abs(cmd.turn) < 0.05
    assert cmd.yaw_conf > 0.9


def test_yaw_is_signed_and_monotonic() -> None:
    values = [_settle_yaw(d).body_yaw_rad for d in (-60, -30, 0, 30, 60)]
    assert values == sorted(values)
    assert values[0] < -0.4 and values[-1] > 0.4


def test_yaw_magnitude_tracks_the_real_rotation() -> None:
    for deg in (20, 45, -35):
        got = math.degrees(_settle_yaw(deg).body_yaw_rad)
        # Depth is deliberately down-weighted, so the estimate under-reports
        # rather than over-reports; the robot must never overshoot a turn.
        assert 0.6 * abs(deg) <= abs(got) <= abs(deg) + 2.0
        assert (got > 0) == (deg > 0)


def test_yaw_is_reported_while_standing_perfectly_still() -> None:
    """The regression that made body rotation move only the head: the yaw used
    to be gated behind the marching state, so standing and turning did nothing."""
    cmd = _settle_yaw(45.0)
    assert cmd.state == "idle"          # not marching...
    assert cmd.body_yaw_rad > 0.4       # ...but the rotation is still reported
    assert cmd.turn > 0.5


def test_yaw_survives_the_legs_leaving_the_frame() -> None:
    ex = GaitCueExtractor()
    cmd = None
    for i in range(40):
        frame = _yaw_frame(i / 30.0, 40.0, i)
        kps = dict(frame.keypoints)
        for name in ("left_knee", "right_knee", "left_ankle", "right_ankle"):
            kps[name] = Keypoint(kps[name].x, kps[name].y, kps[name].z, 0.0)
        cmd = ex.update(PoseFrame(frame.timestamp_s, kps, i))
    assert cmd.conf < 0.6               # lower body not trusted
    assert cmd.body_yaw_rad > 0.3       # yaw still usable (shoulders + hips)
    assert cmd.yaw_conf > 0.9


def test_invisible_torso_yields_no_yaw_confidence() -> None:
    cmd = _settle_yaw(45.0, vis=0.1)
    assert cmd.yaw_conf == 0.0


def test_yaw_is_smoothed_not_snapped() -> None:
    ex = GaitCueExtractor()
    for i in range(30):
        ex.update(_yaw_frame(i / 30.0, 0.0, i))
    jump = ex.update(_yaw_frame(1.0, 60.0, 30))
    # One frame must not deliver the whole 60 deg step, or turn clips chatter.
    assert 0.0 < jump.body_yaw_rad < math.radians(45)


# ---------------------------------------------------------------------------
# Full-circle yaw
#
# The estimate used abs(lateral), which folded the front/back halves together and
# bounded it to +/-90 deg -- so the robot could never be asked to turn round. It
# also spiked to that bound on 7-18% of frames in recorded runs, which made the
# controller demand turn clips in alternating directions and starved walking.
# ---------------------------------------------------------------------------
def test_yaw_covers_the_whole_circle() -> None:
    for target in (0, 45, 90, 135, 180, -135, -90, -45):
        got = math.degrees(_settle_yaw(target).body_yaw_rad)
        # Wrapped difference, since +180 and -180 are the same heading.
        err = (got - target + 180.0) % 360.0 - 180.0
        assert abs(err) < 12.0, (target, got)


def test_facing_away_is_not_confused_with_facing_forward() -> None:
    forward = _settle_yaw(0.0).body_yaw_rad
    away = _settle_yaw(180.0).body_yaw_rad
    assert abs(forward) < 0.15
    assert abs(away) > math.radians(160)


def test_yaw_is_monotonic_through_ninety_degrees() -> None:
    """+/-90 deg used to be a hard bound; it must now be an ordinary waypoint."""
    values = [_settle_yaw(d).body_yaw_rad for d in (60, 75, 90, 105, 120)]
    assert values == sorted(values)
    assert values[-1] > math.radians(100)


def test_a_collapsed_detection_is_rejected_not_read_as_a_big_angle() -> None:
    """The failure that produced the +/-90 spikes: when the shoulder line
    collapses, atan2 reports a large angle from what is really just noise."""
    ex = GaitCueExtractor()
    for i in range(120):
        cmd = ex.update(_yaw_frame(i / 30.0, 0.0, i))
    assert cmd.yaw_conf > 0.9
    for i in range(15):
        cmd = ex.update(_yaw_frame(4.0 + i / 30.0, 0.0, 200 + i, span=0.2))
    assert cmd.yaw_conf == 0.0            # not trusted at all
    assert abs(cmd.body_yaw_rad) < 0.2    # and the estimate did not run away


def test_a_single_frame_jump_is_de_weighted() -> None:
    ex = GaitCueExtractor()
    for i in range(120):
        ex.update(_yaw_frame(i / 30.0, 0.0, i))
    jumped = ex.update(_yaw_frame(4.1, 85.0, 200))
    assert jumped.yaw_conf <= 0.55         # halved: too fast to be real
    assert abs(jumped.body_yaw_rad) < math.radians(45)   # and not followed fully


def test_yaw_crosses_the_seam_by_the_short_way() -> None:
    ex = GaitCueExtractor()
    for i in range(120):
        ex.update(_yaw_frame(i / 30.0, 175.0, i))
    for i in range(60):
        cmd = ex.update(_yaw_frame(5.0 + i / 30.0, -175.0, 300 + i))
    # 10 deg apart across the seam; going the long way would pass through 0.
    assert cmd.body_yaw_rad < -math.radians(150)


def test_yaw_survives_a_steady_rotation_all_the_way_round() -> None:
    """A continuous turn must not stick, fold or jump direction anywhere."""
    ex = GaitCueExtractor()
    seen = []
    for i in range(361):
        cmd = ex.update(_yaw_frame(i / 30.0, i - 180.0, i))
        if i > 60:
            seen.append(cmd.body_yaw_rad)
    unwrapped = [seen[0]]
    for value in seen[1:]:
        prev = unwrapped[-1]
        unwrapped.append(prev + ((value - prev + math.pi) % (2 * math.pi) - math.pi))
    # Monotonically increasing heading, covering most of a full turn.
    assert unwrapped[-1] - unwrapped[0] > math.radians(250)
    assert all(b >= a - 1e-6 for a, b in zip(unwrapped, unwrapped[1:], strict=False))
