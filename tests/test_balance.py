"""Tests for the model-based CoM balance feedback (main/libraries/balance.py)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import (  # noqa: E402
    TILT_PITCH_SIGN,
    TILT_ROLL_SIGN,
    BalanceController,
    NaoCoMModel,
    clip_torso_speeds,
    fibonacci_spiral,
    safe_exit_times,
)
from conftest import cyclic_poses  # noqa: E402


def test_total_mass_is_realistic() -> None:
    # NAO H25 is ~4.8-5.3 kg depending on version.
    assert 4.5 < NaoCoMModel().total_mass < 5.6


def test_standing_com_sits_over_the_feet() -> None:
    m = NaoCoMModel()
    frames = m.frames({})
    com = m.com({}, frames)
    support = 0.5 * (m.foot_sole_center("L", frames) + m.foot_sole_center("R", frames))
    # Standing, the CoM must be within a few cm (a foot half-length) of centre.
    assert abs(com[0] - support[0]) < 0.05
    assert abs(com[1] - support[1]) < 0.02  # near-symmetric left/right


def test_com_is_above_the_feet() -> None:
    m = NaoCoMModel()
    frames = m.frames({})
    com = m.com({}, frames)
    sole = m.foot_sole_center("L", frames)
    assert com[2] > sole[2] + 0.2  # CoM well above the soles


def test_crouch_keeps_com_centred() -> None:
    # Symmetric crouch should not move the CoM far horizontally.
    m = NaoCoMModel()
    u = 0.35
    crouch = {
        "LHipPitch": -u, "RHipPitch": -u,
        "LKneePitch": 2 * u, "RKneePitch": 2 * u,
        "LAnklePitch": -u, "RAnklePitch": -u,
    }
    frames = m.frames(crouch)
    com = m.com(crouch, frames)
    support = 0.5 * (m.foot_sole_center("L", frames) + m.foot_sole_center("R", frames))
    assert abs(com[0] - support[0]) < 0.06


def test_balanced_pose_needs_no_correction() -> None:
    bc = BalanceController(NaoCoMModel())
    corr = bc.compute_correction({}, (0.0, 0.0))
    assert all(abs(v) < 1e-6 for v in corr.values())


# On this NAO a FORWARD tilt reads as a POSITIVE pitch: Nao.proto mounts the
# InertialUnit rolled +90 deg, Webots decomposes its attitude as
# Z(yaw) Y(pitch) X(roll), so the reported pitch is the torso's rotation about its
# own +y (left) axis -- nose-down positive. Derived in
# test_pitch_sign_matches_webots_convention_for_the_mounted_sensor below and
# confirmed by the unrotated Gyro and by 19 of 20 logged pitch-fall onsets (see
# balance.TILT_PITCH_SIGN). Naming it here keeps the raw sign out of every
# individual test.
FORWARD = +1.0
BACKWARD = -1.0


def _com_in_foot(model, angles) -> float:
    """Fore/aft position of the CoM within the left sole, in metres.

    Positive is forward. This is the quantity balance actually controls: the
    torso-frame x is not it, because a correction moves the foot relative to the
    torso as well as the mass above it.
    """
    frames = model.frames(angles)
    com = model.com(angles, frames)
    T = frames["LAnkleRoll"]
    return float((T[:3, :3].T @ (com - T[:3, 3]))[0])


def test_search_reduces_predicted_imbalance() -> None:
    """A BACKWARD torso tilt is the imbalance that matters, and the search must
    respond to it and reduce it.

    Backward, not forward: NAO's sole runs only 30 mm behind the ankle against
    100 mm in front, so leaning back is what actually runs out of foot. A forward
    lean of the same size is still comfortably supported, and the loop correctly
    ignores it -- see test_forward_and_backward_are_not_symmetric.
    """
    bc = BalanceController(NaoCoMModel())
    posture = standing()
    tilt = (0.0, BACKWARD * 0.25)             # 0.25 rad leaning back
    before = bc._imbalance(posture, tilt)
    assert before[0] > 0.0                    # it IS an imbalance
    for _ in range(60):
        corr = bc.compute_correction(posture, tilt)
    assert any(abs(v) > 1e-3 for v in corr.values())          # it did something
    after = bc._imbalance(bc._apply_corr(posture, bc._state), tilt)
    assert after[0] < before[0]                                # cost fell
    assert after[1] > before[1]                                # margin grew


def test_forward_and_backward_are_not_symmetric() -> None:
    """The foot is not symmetric, so the response must not be either.

    NAO's sole runs 100 mm ahead of the ankle and only 30 mm behind it, so the
    same lean is markedly more dangerous backwards than forwards. Both cost
    something at 0.25 rad; backwards must cost more.
    """
    posture = standing()
    fwd = BalanceController(NaoCoMModel())._imbalance(posture, (0.0, FORWARD * 0.25))[0]
    back = BalanceController(NaoCoMModel())._imbalance(posture, (0.0, BACKWARD * 0.25))[0]
    assert back > fwd
    assert fwd >= 0.0


def test_a_forward_lean_is_corrected_BACKWARD() -> None:
    """The regression that matters most in this file now.

    The sign of this correction was inverted, and an inverted balance loop does
    not fail quietly: it corrects hard, at its clamp, in the wrong direction. In a
    recorded session it drove the CoM forward at its +0.25 rad limit for 70
    seconds while the robot pitched from -0.02 to -0.52 rad and fell on its face.
    Positive c moves the CoM forward, so a forward lean must produce NEGATIVE c.
    """
    model = NaoCoMModel()
    bc = BalanceController(model)
    posture = standing()
    for _ in range(80):
        bc.compute_correction(posture, (0.0, FORWARD * 0.30))
    assert bc._state["pitch"] < -0.02, bc._state

    # And the correction really does move the centre of mass backwards -- measured
    # in the FOOT's frame, which is the one that matters: the correction moves the
    # foot relative to the torso too, so a torso-frame comparison is meaningless.
    assert _com_in_foot(model, bc._apply_corr(posture, bc._state)) \
        < _com_in_foot(model, posture)


def test_a_backward_lean_is_corrected_FORWARD() -> None:
    bc = BalanceController(NaoCoMModel())
    for _ in range(80):
        bc.compute_correction(standing(), (0.0, BACKWARD * 0.30))
    assert bc._state["pitch"] > 0.02, bc._state


def test_corrections_are_clamped() -> None:
    bc = BalanceController(NaoCoMModel())
    p = bc.params
    ceiling = max(p.max_pitch_corr, p.max_roll_corr)
    # Drive many cycles of a large imbalance; corrections must stay bounded.
    for _ in range(50):
        corr = bc.compute_correction(standing(), (0.4, -0.4))
    for name, value in corr.items():
        assert abs(value) <= ceiling + 1e-9, name


def test_every_correction_keeps_both_soles_flat() -> None:
    """The property the whole correction is built around.

    A sole's attitude is ``Hip + Knee + Ankle``; the correction adds ``+c`` to the
    hip and ``-c`` to the ankle on both legs, so that sum -- and therefore sole
    contact -- is untouched no matter how large the correction grows.

    It is not a nicety. Once the support polygon is measured from the sole corners
    actually touching the floor, tilting a sole by 0.05 rad cuts the lateral margin
    from 88 mm to 14 mm. A correction that tilted the soles (the previous version
    moved hip and ankle roll independently) destroyed support faster than it moved
    the CoM, so every candidate scored worse than standing still and the loop went
    inert exactly when it was needed.
    """
    bc = BalanceController(NaoCoMModel())
    for rp in ((0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, -0.4), (0.35, -0.25)):
        bc = BalanceController(NaoCoMModel())
        for _ in range(60):
            corr = bc.compute_correction(standing(), rp)
        for side in ("L", "R"):
            tilt = corr[f"{side}HipRoll"] + corr[f"{side}AnkleRoll"]
            assert abs(tilt) < 1e-12, (rp, side, tilt)
            pitch = corr[f"{side}HipPitch"] + corr[f"{side}AnklePitch"]
            assert abs(pitch) < 1e-12, (rp, side, pitch)


# ---------------------------------------------------------------------------
# The permanent-lean regression
# ---------------------------------------------------------------------------
def standing(u: float = 0.10) -> dict:
    """The posture the robot actually holds: arms down, legs in the base crouch."""
    from nao_retarget import crouch_posture
    from pose_control_utils import standing_posture
    ang = standing_posture()
    ang.update(crouch_posture(u))
    return ang


def test_a_standing_robot_gets_no_correction_at_all() -> None:
    """The regression that matters most in this file.

    The objective used to be "distance from the midpoint of the two foot
    centroids". A sole runs 30 mm behind the ankle and 100 mm in front, so that
    midpoint is 35 mm FORWARD of the ankle -- while a standing NAO rests with its
    CoM about 7 mm forward of it. The loop therefore measured a balanced robot as
    28 mm out of balance, above its own 20 mm action threshold, and pushed
    continuously: the ankle-roll correction saturated against its clamp, the
    sole-tilt limiter dragged both hips across to protect sole contact, and the
    robot stood in a fixed 0.300 rad lean for 98% of a recorded session.
    """
    for u in (0.10, 0.25, 0.40, 0.70):
        bc = BalanceController(NaoCoMModel())
        posture = standing(u)
        assert bc._imbalance(posture, (0.0, 0.0))[0] == 0.0, u
        for _ in range(120):
            corr = bc.compute_correction(posture, (0.0, 0.0))
        assert max(abs(v) for v in corr.values()) < 1e-4, (u, corr)
        # And specifically: no lean. Hip roll is applied with the SAME sign to both
        # legs, so any residue here is a whole-body lean.
        assert abs(corr["LHipRoll"] + corr["RHipRoll"]) < 1e-6, u


def test_the_search_reduces_cost_on_the_axis_it_acts_on() -> None:
    """Sanity: the chosen correction must actually lower the objective, including
    the InertialUnit term, not merely change something."""
    m = NaoCoMModel()
    for rp in ((0.0, BACKWARD * 0.25), (0.35, 0.0), (-0.35, 0.0)):
        bc = BalanceController(m)
        before = bc._imbalance(standing(), rp)[0]
        assert before > 0.0, rp
        for _ in range(80):
            bc.compute_correction(standing(), rp)
        after = bc._imbalance(bc._apply_corr(standing(), bc._state), rp)[0]
        assert after < before, rp


def test_a_real_lean_is_detected_and_reduced() -> None:
    """The loop must still catch genuine tipping -- it just must not invent it."""
    m = NaoCoMModel()
    tipped = standing()
    for j in ("LHipRoll", "RHipRoll"):
        tipped[j] = 0.55                     # same sign both legs = a lean
    assert m.support_margin(tipped) < 0.0     # genuinely outside the polygon
    bc = BalanceController(m)
    before = bc._imbalance(tipped, (0.0, 0.0))[1]
    for _ in range(80):
        bc.compute_correction(tipped, (0.0, 0.0))
    after = bc._imbalance(bc._apply_corr(tipped, bc._state), (0.0, 0.0))[1]
    assert after > before + 0.01


def test_support_margin_is_positive_at_every_posture_the_robot_holds() -> None:
    m = NaoCoMModel()
    for u in (0.0, 0.10, 0.25, 0.40, 0.70):
        assert m.support_margin(standing(u)) > 0.03, u


def test_a_correction_uses_the_axis_the_disturbance_is_on() -> None:
    """Cross-axis waste matters: everything the balance loop spends is subtracted
    from the pose the robot is supposed to be imitating. The spiral perturbs pitch
    and roll together, so without the effort penalty a purely lateral disturbance
    dragged the pitch correction to its clamp for free."""
    m = NaoCoMModel()
    bc = BalanceController(m)
    for _ in range(80):
        bc.compute_correction(standing(), (0.35, 0.0))     # roll only
    assert abs(bc._state["roll"]) > 0.02                   # answered on roll
    assert abs(bc._state["pitch"]) < 0.05                  # and not on pitch

    bc2 = BalanceController(m)
    for _ in range(80):
        bc2.compute_correction(standing(), (0.0, BACKWARD * 0.30))  # pitch only
    assert abs(bc2._state["pitch"]) > 0.03                 # answered on pitch
    assert abs(bc2._state["roll"]) < 0.02                  # and not on roll


def test_fibonacci_spiral_is_bounded_and_even() -> None:
    pts = fibonacci_spiral(50, 0.2)
    assert len(pts) == 50
    assert all(math.hypot(x, y) <= 0.2 + 1e-9 for x, y in pts)
    # Golden-angle spacing => points are spread, not collinear.
    angles = {round(math.atan2(y, x), 2) for x, y in pts}
    assert len(angles) > 25


# ---------------------------------------------------------------------------
# The divergence detector
# ---------------------------------------------------------------------------
def test_divergence_is_detected_when_tilt_grows_against_a_saturated_correction() -> None:
    """An inverted balance loop is silent, and that silence cost 70 seconds and a
    fall. Growing tilt plus a saturated correction is the signature; it must be
    reported rather than merely survived."""
    bc = BalanceController(NaoCoMModel())
    assert bc.diverging() == ""
    # Force the state to the clamp, as a diverging loop does.
    bc._state["pitch"] = bc.params.max_pitch_corr
    t = 0.0
    tilt = 0.10
    for _ in range(40):                       # ~4s of steadily worsening tilt
        bc.note_tilt(t, 0.0, -tilt)
        t += 0.1
        tilt += 0.01
    assert "DIVERGING on pitch" in bc.diverging()
    assert "PITCH_SIGN" in bc.diverging()


def test_no_divergence_when_the_correction_is_working() -> None:
    """Tilt that is being reduced, or a correction well short of its clamp, must
    not be reported -- a detector that cries wolf gets ignored."""
    bc = BalanceController(NaoCoMModel())
    t = 0.0
    tilt = 0.40
    for _ in range(40):
        bc._state["pitch"] = bc.params.max_pitch_corr
        bc.note_tilt(t, 0.0, -tilt)
        t += 0.1
        tilt -= 0.005                         # recovering
    assert bc.diverging() == ""

    bc2 = BalanceController(NaoCoMModel())
    t = 0.0
    tilt = 0.10
    for _ in range(40):
        bc2._state["pitch"] = 0.01           # nowhere near the clamp
        bc2.note_tilt(t, 0.0, -tilt)
        t += 0.1
        tilt += 0.01
    assert bc2.diverging() == ""


# --------------------------------------------------- per-axis divergence window
# A lateral fall is far faster than a fore/aft one: NAO's sole is long in x and
# narrow in y, so the CoM has ~30 mm to travel sideways versus most of a foot
# length forwards. Measured on this robot: the fore/aft divergence that set
# PITCH_SIGN took 70 s; a lateral one took 1.2 s from the first non-zero roll
# correction to both feet off the ground.
def _diverging_controller():
    from balance import BalanceController
    ctrl = BalanceController.__new__(BalanceController)
    ctrl._tilt_history = {}
    ctrl._divergence = ""
    ctrl._state = {"pitch": 0.0, "roll": 0.0}

    class _P:
        max_pitch_corr = 0.25
        max_roll_corr = 0.20
    ctrl.params = _P()
    return ctrl


def test_a_fast_lateral_divergence_is_caught() -> None:
    """The regression: with a single 3 s window this could not fire at all,
    because the robot was on the floor after 1.2 s."""
    c = _diverging_controller()
    c._state["roll"] = 0.19          # saturated against max_roll_corr 0.20
    for i in range(13):              # 1.2 s at 100 Hz-ish sampling
        c.note_tilt(i * 0.1, roll=0.02 + i * 0.02, pitch=0.0)
    assert "DIVERGING on roll" in c.diverging()
    assert "ROLL_SIGN" in c.diverging()


def test_a_slow_fore_aft_divergence_is_still_caught() -> None:
    """The case that set PITCH_SIGN must keep working."""
    c = _diverging_controller()
    c._state["pitch"] = 0.24
    for i in range(40):
        c.note_tilt(i * 0.1, roll=0.0, pitch=0.02 + i * 0.01)
    assert "DIVERGING on pitch" in c.diverging()


def test_a_steady_lean_is_not_called_divergence() -> None:
    """Holding a tilt is not the same as an increasing one."""
    c = _diverging_controller()
    c._state["roll"] = 0.19
    for i in range(20):
        c.note_tilt(i * 0.1, roll=0.15, pitch=0.0)
    assert c.diverging() == ""


def test_growth_without_saturation_is_not_divergence() -> None:
    """A loop still inside its authority is correcting, not fighting itself."""
    c = _diverging_controller()
    c._state["roll"] = 0.01          # nowhere near the clamp
    for i in range(13):
        c.note_tilt(i * 0.1, roll=0.02 + i * 0.02, pitch=0.0)
    assert c.diverging() == ""


def test_the_roll_window_is_shorter_than_the_pitch_window() -> None:
    """The whole point: the two axes fall on different timescales."""
    from balance import BalanceController
    assert BalanceController.DIVERGENCE_S["roll"] < BalanceController.DIVERGENCE_S["pitch"]


# ---------------------------------------------------------------------------
# Sensor conventions, from first principles
# ---------------------------------------------------------------------------
def _rot_x(a: float):
    import numpy as np
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a: float):
    import numpy as np
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _quat_xyzw(R):
    """Rotation matrix -> unit quaternion (x, y, z, w), Webots' component order."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = max(range(3), key=lambda k: R[k, k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def _webots_enu_rpy(q):
    """Exactly wb_inertial_unit_get_roll_pitch_yaw for an ENU world
    (src/controller/c/inertial_unit.c, Webots R2025a): e = Z(yaw) Y(pitch) X(roll)."""
    roll = math.atan2(2.0 * (q[3] * q[0] + q[1] * q[2]),
                      1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]))
    t2 = max(-1.0, min(1.0, 2.0 * (q[3] * q[1] - q[2] * q[0])))
    pitch = math.asin(t2)
    yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                     1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
    return roll, pitch, yaw


def test_pitch_sign_matches_webots_convention_for_the_mounted_sensor() -> None:
    """Derive the tilt signs from the sensor mounting instead of from a remembered
    fall. Nao.proto mounts the InertialUnit ``rotation 1 0 0 1.5708`` (rolled +90
    deg about the torso x axis); Webots' ENU API decomposes the sensor's world
    attitude as Z(yaw) Y(pitch) X(roll). So tilt a torso and see what the API
    would report.

    The previous value (-1) was the single most expensive bug on this project: it
    turned both CoM shifters into positive feedback in pitch, and every pitch fall
    in the 2026-09-03 logs shows their clamps saturated in the direction of the
    fall.
    """
    mount = _rot_x(math.pi / 2)
    upright = _webots_enu_rpy(_quat_xyzw(mount))
    assert upright[0] == pytest.approx(math.pi / 2, abs=1e-9)   # the raw roll the
    assert upright[1] == pytest.approx(0.0, abs=1e-9)           # controller zeroes

    for b in (0.1, 0.3):
        # Rotation about the torso's +y axis by +b sends the forward axis x to
        # (cos b, 0, -sin b): the nose goes DOWN. A forward tilt.
        nose_down = _rot_y(b) @ mount
        roll, pitch, _ = _webots_enu_rpy(_quat_xyzw(nose_down))
        assert pitch == pytest.approx(b, abs=1e-9)              # reads POSITIVE
        assert roll - math.pi / 2 == pytest.approx(0.0, abs=1e-9)
        # Rotation about +x by +a sends the up axis z to (0, -sin a, cos a): the top
        # of the robot moves toward -y, i.e. to the RIGHT.
        right = _rot_x(b) @ mount
        roll, pitch, _ = _webots_enu_rpy(_quat_xyzw(right))
        assert roll - math.pi / 2 == pytest.approx(b, abs=1e-9)  # reads POSITIVE
        assert pitch == pytest.approx(0.0, abs=1e-9)

    # Hence a positive pitch is a forward tilt, whose gravity projection moves the
    # CoM FORWARD (+x); a positive roll is a right tilt, whose projection moves it
    # to the RIGHT (-y). The model must displace the CoM the same way.
    assert TILT_PITCH_SIGN == +1.0
    assert TILT_ROLL_SIGN == +1.0
    m = NaoCoMModel()
    level = m.tilted_com_xy({}, (0.0, 0.0))
    forward = m.tilted_com_xy({}, (0.0, +0.1))
    right = m.tilted_com_xy({}, (+0.1, 0.0))
    assert forward[0] > level[0] + 0.02                          # ~31 mm forward
    assert right[1] < level[1] - 0.02                            # ~31 mm to the right


def test_the_tilt_lever_arm_follows_the_squat() -> None:
    """A deep squat lowers the CoM; the tilt term must not charge it the standing
    lever arm. Standing measures ~0.31 m, the deepest squat ~0.27 m."""
    m = NaoCoMModel()
    stand = m.frames({})
    h_stand = m.com_height(stand, m.com({}, stand))
    u = 0.70
    deep = {"LHipPitch": -u, "RHipPitch": -u, "LKneePitch": 2 * u, "RKneePitch": 2 * u,
            "LAnklePitch": -u, "RAnklePitch": -u}
    frames = m.frames(deep)
    h_deep = m.com_height(frames, m.com(deep, frames))
    assert 0.29 < h_stand < 0.33
    assert h_deep < h_stand - 0.02


def test_the_correction_is_rate_limited() -> None:
    """The hip motors deliver at most ~1.8 rad/s under the driver's caps; a
    correction that moves faster than that runs ahead of the joint and then
    corrects its own tracking lag. One 20 ms call may move at most max_rate*dt."""
    bc = BalanceController(NaoCoMModel())
    cap = bc.params.max_rate * 0.02
    prev = dict(bc._state)
    for _ in range(40):
        bc.compute_correction(standing(), (0.0, BACKWARD * 0.35), dt_s=0.02)
        for k in prev:
            assert abs(bc._state[k] - prev[k]) <= cap + 1e-9, (k, bc._state, prev)
        prev = dict(bc._state)
    assert abs(bc._state["pitch"]) > 0.05          # and it did get somewhere


def test_gyro_lead_makes_the_loop_act_early() -> None:
    """Level right now but tipping backward fast: with the rate term the loop
    already corrects; without it (lead 0) it would sit still until the tilt
    itself crosses the margin."""
    model = NaoCoMModel()
    with_lead = BalanceController(model)
    for _ in range(25):
        with_lead.compute_correction(standing(), (0.0, 0.0),
                                     tilt_rate=(0.0, BACKWARD * 2.0), dt_s=0.02)
    assert with_lead._state["pitch"] > 0.02, with_lead._state   # CoM pushed forward

    from balance import BalanceParams
    no_lead = BalanceController(model, BalanceParams(tilt_lead_s=0.0))
    for _ in range(25):
        no_lead.compute_correction(standing(), (0.0, 0.0),
                                   tilt_rate=(0.0, BACKWARD * 2.0), dt_s=0.02)
    assert abs(no_lead._state["pitch"]) < 1e-6


def test_safe_exit_times_accepts_double_support_and_refuses_a_lifted_foot() -> None:
    """What makes a moment safe to stop a clip at: both feet down, both soles
    flat, the centre of mass inside the contact-filtered polygon, and nothing
    about to move fast. A clip BOUNDARY has those properties, which is why clips
    used to be played to completion -- but so do many of a walk clip's interior
    keyframes, and that is what lets a long clip be stopped promptly.
    """
    from balance import safe_exit_times

    u = 0.51                                    # the crouch every walk clip opens in
    crouch = {"LHipPitch": -u, "RHipPitch": -u,
              "LKneePitch": 2 * u, "RKneePitch": 2 * u,
              "LAnklePitch": -u, "RAnklePitch": -u}
    # A pose with the left foot clearly in the air.
    lifted = dict(crouch)
    lifted.update({"LHipPitch": -u - 0.5, "LKneePitch": 2 * u + 0.6})

    # Held still, the crouch is a safe exit; the lifted foot never is.
    assert safe_exit_times([(0.0, crouch), (0.04, crouch)]) == [0.0, 0.04]
    assert safe_exit_times([(0.0, lifted), (0.04, lifted)]) == []

    # A keyframe about to move fast is refused even in double support: stopping
    # there would freeze the robot mid-lurch.
    far = dict(crouch)
    far["LKneePitch"] = 2 * u + 0.5             # 12.5 rad/s over one 40 ms frame
    assert safe_exit_times([(0.0, crouch), (0.04, far)])[0:1] == []

    # And the interior of a real sequence is found, not just its ends. Note the
    # opening crouch is NOT offered here: the very next keyframe lifts a foot, so
    # stopping at t=0 would freeze the robot on the way into a stride.
    seq = [(0.0, crouch), (0.04, lifted), (0.08, crouch)]
    assert safe_exit_times(seq) == [0.08]


# ---------------------------------------------------------------------------
# Clip odometry and safe exits
# ---------------------------------------------------------------------------
def test_clip_torso_speeds_tells_a_walk_from_a_march() -> None:
    """The only measure available of how fast a clip's own kinematics travel.

    There is no sensor for this -- the clip is a file, not a run -- but the
    stance foot is on the floor, so the torso moves over it by exactly as much
    as forward kinematics says the hip swings. Distinguishing a gait that goes
    somewhere from one that marches in place is what stops a turn clip from being
    looped, so it has to be right about the sign as well as the magnitude.
    """
    walking = clip_torso_speeds(cyclic_poses(translate=True))
    marching = clip_torso_speeds(cyclic_poses(translate=False))
    dt = 0.04
    assert sum(walking) * dt > 0.20        # ~0.40 m over three strides
    assert abs(sum(marching) * dt) < 0.02  # goes nowhere, both directions
    # Note that the march is not STILL -- its torso rocks at up to 0.21 m/s
    # instantaneously. Net travel is what separates a gait from a shuffle, and
    # it is the net that decides whether a clip may be looped.
    assert max(abs(v) for v in marching) > 0.10
    # Inside the repeating stride the walk is always going forwards. The speed is
    # not constant (stance and swing modulate it) but it never reverses, which is
    # the difference between a cycle and the closing settle of a one-shot clip --
    # that settle runs BACKWARD, and it is what makes chained clips lurch.
    cycle = walking[8:34]
    assert min(cycle) > 0.0
    assert max(cycle) / min(cycle) < 12.0


def test_safe_exit_times_refuses_a_pose_that_is_still_moving() -> None:
    """Stopping the legs does not stop the ROBOT, and the static tests miss that.

    A pose can be in double support, soles flat, CoM inside the polygon -- and
    still be travelling at 0.18 m/s, because that is what walking is. Freeze the
    legs there and the body keeps its momentum: the capture point sits v/omega
    ahead of the CoM (omega = sqrt(g/h) ~= 5.9 rad/s at this crouch), so 0.18 m/s
    is 30 mm of excursion against a margin budget of 40-60 mm. 13 of the 46 poses
    that passed the static tests in the real short clip were like that. Handing
    the body back at one of them means handing back a robot that then walks
    itself over.
    """
    poses = cyclic_poses(translate=True)
    speeds = clip_torso_speeds(poses)
    times = [t for t, _ in poses]

    permissive = safe_exit_times(poses, max_torso_speed=float("inf"))
    gated = safe_exit_times(poses, max_torso_speed=0.05)
    assert gated, "the gate must not reject every pose in a real walk"
    assert set(gated) <= set(permissive), "the gate can only ever remove poses"
    assert len(gated) < len(permissive), "this clip does travel; some must go"
    # Every pose that survives is genuinely near rest.
    for t in gated:
        assert abs(speeds[times.index(t)]) <= 0.05
    # ...and every pose that was dropped for momentum was moving.
    for t in set(permissive) - set(gated):
        assert abs(speeds[times.index(t)]) > 0.05


def test_the_momentum_gate_only_ever_removes_and_removes_in_order() -> None:
    """Two properties that keep the gate from being a trap of its own.

    Monotone: a tighter threshold can only remove exits, never invent one. And
    a pose genuinely at rest is never removed at any threshold, so a clip that
    comes to a stop keeps somewhere to stop -- which is why the turn clips, which
    settle repeatedly, keep prompt early exits (49 of TurnLeft40's 73 keyframes,
    longest wait 0.52 s) while a continuous walk keeps very few.
    """
    poses = cyclic_poses(translate=True)
    speeds = clip_torso_speeds(poses)
    times = [t for t, _ in poses]
    thresholds = [0.01, 0.05, 0.2, float("inf")]
    sets = [set(safe_exit_times(poses, max_torso_speed=v)) for v in thresholds]
    for tighter, looser in zip(sets, sets[1:]):
        assert tighter <= looser

    # Anything at rest survives the tightest threshold used here.
    at_rest = {t for t, v in zip(times, speeds) if abs(v) < 0.005}
    assert at_rest & sets[-1] <= sets[0]
