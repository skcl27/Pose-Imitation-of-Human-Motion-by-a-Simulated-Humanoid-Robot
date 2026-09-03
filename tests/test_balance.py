"""Tests for the model-based CoM balance feedback (main/libraries/balance.py)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

from balance import (  # noqa: E402
    BalanceController,
    NaoCoMModel,
    fibonacci_spiral,
)


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


# On this NAO the InertialUnit reports a FORWARD tilt as a negative pitch --
# measured from a session where the robot fell forward while imu_pitch went to
# -0.52. Naming it here keeps the raw sign out of every individual test.
FORWARD = -1.0
BACKWARD = +1.0


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
