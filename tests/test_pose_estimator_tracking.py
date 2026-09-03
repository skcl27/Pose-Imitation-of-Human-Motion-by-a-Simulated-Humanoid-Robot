"""Tests for the detector-skipping fast path (src/perception/pose_estimator.py).

MeTRAbs' ``detect_poses`` runs YOLOv4 AND the pose network. Measured on this
project's machine the detector alone is 39.7 ms of a 75.9 ms frame -- more than
half the budget, spent re-finding a person who moved a few pixels. Reusing the
previous frame's box costs 36.2 ms instead.

The risk it introduces is the reason for every test here: ``estimate_poses`` has
no detector, so given a box it ALWAYS returns a skeleton -- including of an empty
room after the subject walks away. A dropped frame makes the robot hold still; an
invented one makes it move to a phantom.

No TensorFlow and no model: the arithmetic that decides WHEN to trust the cheap
path is ordinary Python and is what actually breaks.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.perception.pose_estimator import PoseEstimator


def _est(**kw) -> PoseEstimator:
    """An estimator with no model loaded -- only the tracking logic is exercised."""
    return PoseEstimator(use_metrabs=False, allow_synthetic_fallback=True, **kw)


# ------------------------------------------------------------ continuity gate
def test_a_still_subject_is_believed() -> None:
    est = _est()
    pose = np.array([[0.0, 0.0, 2000.0], [100.0, 50.0, 2000.0], [0.0, 500.0, 2000.0],
                     [50.0, 900.0, 2000.0]])
    est._prev_poses3d = pose
    assert est._pose_is_plausible(pose + 5.0) is True


def test_a_teleporting_subject_is_disbelieved() -> None:
    """The signature of the box drifting onto something that is not a person."""
    est = _est()
    pose = np.array([[0.0, 0.0, 2000.0], [100.0, 50.0, 2000.0], [0.0, 500.0, 2000.0],
                     [50.0, 900.0, 2000.0]])
    est._prev_poses3d = pose
    assert est._pose_is_plausible(pose + 900.0) is False


def test_the_first_tracked_frame_is_accepted() -> None:
    """Straight after a detection the box is at its freshest; there is nothing
    to compare against and refusing here would defeat the whole optimisation."""
    est = _est()
    est._prev_poses3d = None
    assert est._pose_is_plausible(np.zeros((19, 3))) is True


def test_a_shape_change_is_accepted_rather_than_crashing() -> None:
    est = _est()
    est._prev_poses3d = np.zeros((19, 3))
    assert est._pose_is_plausible(np.zeros((17, 3))) is True


def test_the_jump_threshold_is_a_median_not_a_maximum() -> None:
    """One jittery joint must not force a re-detection of the whole body."""
    est = _est(max_joint_jump_mm=300.0)
    prev = np.zeros((19, 3))
    est._prev_poses3d = prev
    moved = prev.copy()
    moved[0] = [5000.0, 0.0, 0.0]          # a single wild joint
    assert est._pose_is_plausible(moved) is True


# --------------------------------------------------------------- box tracking
def _joints(cx: float, cy: float, w: float, h: float, n: int = 19) -> np.ndarray:
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    ys = np.linspace(cy - h / 2, cy + h / 2, n)
    return np.stack([xs, ys], axis=1)


def test_the_tracked_box_keeps_the_detectors_size_not_the_joint_span() -> None:
    """MeTRAbs projects out-of-frame joints, so a box drawn round the joints is
    ~1.3x wider and ~1.9x taller than YOLOv4's -- a crop the pose network was
    never calibrated for, worth up to 250 mm of joint error. The detector's own
    box geometry has to be carried forward."""
    est = _est()
    est._tracked_box = np.array([100.0, 50.0, 400.0, 800.0, 0.9], dtype=np.float32)
    est._ref_extent = (500.0, 1400.0)                 # joint span at detection
    box = est._box_from_joints(_joints(300.0, 450.0, 500.0, 1400.0), 1920, 1080)
    assert box is not None
    assert box[2] == pytest.approx(400.0, rel=1e-3)   # width unchanged
    assert box[3] == pytest.approx(800.0, rel=1e-3)   # height unchanged


def test_the_tracked_box_follows_the_subject() -> None:
    est = _est()
    est._tracked_box = np.array([100.0, 50.0, 400.0, 800.0, 0.9], dtype=np.float32)
    est._ref_extent = (500.0, 1400.0)
    box = est._box_from_joints(_joints(800.0, 450.0, 500.0, 1400.0), 1920, 1080)
    assert box is not None
    assert box[0] + box[2] / 2 == pytest.approx(800.0, abs=1.0)   # re-centred


def test_the_tracked_box_grows_as_the_subject_approaches() -> None:
    est = _est()
    est._tracked_box = np.array([100.0, 50.0, 400.0, 800.0, 0.9], dtype=np.float32)
    est._ref_extent = (500.0, 1400.0)
    box = est._box_from_joints(_joints(300.0, 450.0, 750.0, 2100.0), 1920, 1080)
    assert box is not None
    assert box[2] == pytest.approx(600.0, rel=0.05)   # 1.5x nearer -> 1.5x box


def test_an_implausible_scale_change_forces_a_redetect() -> None:
    est = _est()
    est._tracked_box = np.array([100.0, 50.0, 400.0, 800.0, 0.9], dtype=np.float32)
    est._ref_extent = (500.0, 1400.0)
    assert est._box_from_joints(_joints(300.0, 450.0, 50.0, 140.0), 1920, 1080) is None
    assert est._box_from_joints(_joints(300.0, 450.0, 5000.0, 14000.0), 1920, 1080) is None


def test_no_reference_means_no_tracking() -> None:
    est = _est()
    est._tracked_box = None
    est._ref_extent = None
    assert est._box_from_joints(_joints(300.0, 450.0, 500.0, 1400.0), 1920, 1080) is None


def test_degenerate_joints_force_a_redetect() -> None:
    est = _est()
    est._tracked_box = np.array([100.0, 50.0, 400.0, 800.0, 0.9], dtype=np.float32)
    est._ref_extent = (500.0, 1400.0)
    nan = np.full((19, 2), np.nan)
    assert est._box_from_joints(nan, 1920, 1080) is None


def test_detect_interval_of_one_disables_tracking_entirely() -> None:
    """The escape hatch: interval 1 must be exactly the old behaviour."""
    est = _est(detect_interval=1)
    assert est.detect_interval == 1


# ------------------------------------------------------- visibility proxy
# nao_retarget.py drops any joint whose visibility is under VIS_THRESHOLD = 0.5,
# so anything that scales the proxy down uniformly does not "reduce confidence"
# -- it deletes the joint. Measured on the project camera, only 13.1% of
# joint-observations cleared the gate against 31.4% on the in-frame test alone.
_W, _H = 1920, 1080


def _centred_box(conf: float = 0.49) -> np.ndarray:
    """A person-shaped detection box, centred, 400x900 px."""
    return np.array([760.0, 90.0, 400.0, 900.0, conf], dtype=np.float32)


def test_detector_confidence_does_not_scale_a_joints_visibility() -> None:
    """``box[4]`` answers "is someone there", not "is this joint trustworthy".

    Folding it in multiplied every joint by the same sub-1.0 factor, which on
    this camera (confidence 0.45-0.50) put a plainly visible shoulder at 0.489
    -- 0.011 under the gate, and therefore discarded.
    """
    est = _est()
    joint = np.array([960.0, 540.0])       # dead centre of image and box
    low = est._visibility_proxy(joint, _centred_box(conf=0.49), _W, _H)
    high = est._visibility_proxy(joint, _centred_box(conf=0.99), _W, _H)
    assert low == pytest.approx(high)
    assert low >= 0.5


def test_a_fully_visible_joint_clears_the_retargeters_gate() -> None:
    est = _est()
    joint = np.array([960.0, 540.0])
    assert est._visibility_proxy(joint, _centred_box(), _W, _H) == pytest.approx(1.0)


def test_an_extremity_on_the_box_edge_survives_the_padding() -> None:
    """Ankles sit on the box floor and shoulders at its widest BY CONSTRUCTION.

    Without ``box_padding`` applied they scored ~0 however plainly visible they
    were, which is what emptied the lower body: the controller reported "no
    lower-body landmarks in frame" while the ankles were in shot.
    """
    est = _est()
    box = _centred_box()
    left, top, w, h = box[0], box[1], box[2], box[3]
    ankle = np.array([left + w * 0.5, top + h])     # exactly on the box floor
    shoulder = np.array([left, top + h * 0.25])     # exactly on the box's left edge
    assert est._visibility_proxy(ankle, box, _W, _H) >= 0.5
    assert est._visibility_proxy(shoulder, box, _W, _H) >= 0.5


def test_a_joint_outside_the_image_still_scores_zero() -> None:
    """The in-frame term is the one doing real work and must keep working:
    a joint off-screen was extrapolated by the pose network, not seen."""
    est = _est()
    box = _centred_box()
    assert est._visibility_proxy(np.array([-40.0, 540.0]), box, _W, _H) == 0.0
    assert est._visibility_proxy(np.array([960.0, _H + 200.0]), box, _W, _H) == 0.0


def test_a_joint_far_outside_the_padded_box_still_scores_zero() -> None:
    """Padding must not become "accept anything": a joint nowhere near the
    person is the signature of the box having drifted onto something else."""
    est = _est()
    box = _centred_box()
    stray = np.array([box[0] + box[2] + 4.0 * box[2], 540.0])
    assert est._visibility_proxy(stray, box, _W, _H) == 0.0


def test_the_box_margin_scales_to_the_box_not_the_image() -> None:
    """A distant subject's box is smaller than the image-derived margin, which
    made every joint in it score 0 -- the subject vanished by walking away."""
    est = _est()
    small = np.array([900.0, 500.0, 120.0, 260.0, 0.9], dtype=np.float32)
    centre = np.array([small[0] + small[2] * 0.5, small[1] + small[3] * 0.5])
    assert est._visibility_proxy(centre, small, _W, _H) == pytest.approx(1.0)
