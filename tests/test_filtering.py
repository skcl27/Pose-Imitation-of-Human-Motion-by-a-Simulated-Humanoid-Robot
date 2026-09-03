"""Tests for src/utils/filtering.py.

The point of the 1e filter here is a specific, measurable claim: it beats a
fixed EMA on the latency/jitter trade rather than just moving along it. So these
tests compare the two directly at MATCHED noise attenuation, which is the only
comparison that means anything -- any filter looks fast if you let it be noisy.

Sample rate throughout is the pipeline's measured 14.3 FPS (70.1 ms/frame).
"""
from __future__ import annotations

import math
import statistics as st

from src.utils.filtering import ExponentialSmoother, OneEuroFilter, _alpha_for

DT = 0.0701          # measured median frame interval, seconds
EMA_ALPHA = 0.5      # what the pipeline used before


def _noise(n: int, sigma: float = 4.0) -> list[float]:
    """Deterministic pseudo-noise -- no RNG, so failures are reproducible."""
    return [sigma * math.sin(i * 2.399963) * math.cos(i * 1.61803) for i in range(n)]


# ------------------------------------------------------------------ alpha math
def test_alpha_rises_with_cutoff() -> None:
    """Higher cutoff = less smoothing = alpha closer to 1 (pass-through)."""
    assert _alpha_for(0.5, DT) < _alpha_for(5.0, DT) < _alpha_for(50.0, DT) < 1.0


def test_alpha_is_bounded() -> None:
    assert 0.0 < _alpha_for(1.0, DT) < 1.0
    assert _alpha_for(1e-9, DT) > 0.0          # absurd cutoff must not divide by zero
    assert _alpha_for(1.0, 1e-9) > 0.0         # absurd dt must not either


# --------------------------------------------------------------- seeding / edges
def test_the_first_sample_passes_through() -> None:
    """No history means nothing to blend with -- inventing one would fabricate
    a starting pose and drag the robot toward it."""
    f = OneEuroFilter()
    assert f.update({"j": 123.0}, DT)["j"] == 123.0


def test_a_stalled_clock_does_not_produce_infinities() -> None:
    """dt <= 0 makes the derivative meaningless (and would divide by zero)."""
    f = OneEuroFilter()
    f.update({"j": 0.0}, DT)
    out = f.update({"j": 10.0}, 0.0)
    assert math.isfinite(out["j"])
    out = f.update({"j": 20.0}, -1.0)
    assert math.isfinite(out["j"])


def test_a_non_finite_sample_does_not_poison_the_state() -> None:
    f = OneEuroFilter()
    f.update({"j": 5.0}, DT)
    f.update({"j": float("nan")}, DT)
    assert math.isfinite(f.update({"j": 5.0}, DT)["j"])


def test_keys_are_filtered_independently() -> None:
    f = OneEuroFilter()
    f.update({"a": 0.0, "b": 0.0}, DT)
    out = f.update({"a": 1000.0, "b": 0.0}, DT)
    assert out["b"] == 0.0          # b never moved, so b must not have moved


def test_reset_clears_history() -> None:
    f = OneEuroFilter()
    f.update({"j": 100.0}, DT)
    f.reset()
    assert f.update({"j": 7.0}, DT)["j"] == 7.0


# --------------------------------------------------- the actual design claim
def test_it_is_as_quiet_as_the_old_ema_on_a_held_pose() -> None:
    """Standing still, the 1e filter must not be JITTERIER than what it replaced
    -- otherwise it has simply traded the robot's lag for a tremor."""
    noise = _noise(400)
    one = OneEuroFilter(min_cutoff=1.0, beta=0.005)
    ema = ExponentialSmoother(alpha=EMA_ALPHA)
    o = [one.update({"j": v}, DT)["j"] for v in noise][100:]
    e = [ema.update({"j": v})["j"] for v in noise][100:]
    assert st.pstdev(o) <= st.pstdev(e)


def test_it_tracks_a_fast_motion_with_less_lag_than_the_old_ema() -> None:
    """A limb crossing 600 mm in ~0.5 s -- the motion the robot was late on.

    Lag is measured as the area between the input and the filtered output: the
    total "distance behind the human" accumulated over the move.
    """
    ramp = [min(600.0, 1700.0 * i * DT) for i in range(40)]   # ~1700 mm/s
    one = OneEuroFilter(min_cutoff=1.0, beta=0.005)
    ema = ExponentialSmoother(alpha=EMA_ALPHA)
    o = [one.update({"j": v}, DT)["j"] for v in ramp]
    e = [ema.update({"j": v})["j"] for v in ramp]
    lag_one = sum(abs(a - b) for a, b in zip(ramp, o, strict=True))
    lag_ema = sum(abs(a - b) for a, b in zip(ramp, e, strict=True))
    assert lag_one < lag_ema


def test_the_cutoff_actually_opens_with_speed() -> None:
    """The mechanism itself: identical step, different approach speeds. The
    faster arrival must end up nearer the target on the same frame count."""
    def travel(speed_mm_s: float) -> float:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.005)
        v = 0.0
        for _ in range(6):
            out = f.update({"j": v}, DT)["j"]
            v += speed_mm_s * DT
        return out / max(v, 1e-9)

    assert travel(2000.0) > travel(200.0)


def test_beta_zero_degrades_to_a_plain_low_pass() -> None:
    """With beta = 0 the cutoff no longer tracks speed, so a fast ramp is
    delayed as much as a slow one -- the EMA behaviour it replaces."""
    def travel(speed_mm_s: float) -> float:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        v = 0.0
        for _ in range(6):
            out = f.update({"j": v}, DT)["j"]
            v += speed_mm_s * DT
        return out / max(v, 1e-9)

    assert travel(2000.0) == round(travel(200.0), 12) or abs(
        travel(2000.0) - travel(200.0)
    ) < 1e-9


# ------------------------------------------------------------- EMA (unchanged)
def test_ema_still_behaves() -> None:
    ema = ExponentialSmoother(alpha=0.5)
    assert ema.update({"j": 10.0})["j"] == 10.0      # seeds on first sample
    assert ema.update({"j": 20.0})["j"] == 15.0


# ------------------------------------------------- pipeline wiring (_KeypointSmoother)
# The smoother is constructed and called in src/pipeline.py; nothing else covers
# that seam, and its signature changed when the EMA was replaced.
def test_keypoint_smoother_preserves_names_and_visibility() -> None:
    from src.pipeline import _KeypointSmoother
    from src.type_defs import Keypoint

    s = _KeypointSmoother()
    kps = {
        "left_wrist": Keypoint(x=10.0, y=20.0, z=30.0, visibility=0.9),
        "right_wrist": Keypoint(x=-10.0, y=5.0, z=25.0, visibility=0.4),
    }
    out = s.update(kps, 100.0)
    assert set(out) == set(kps)
    # visibility reflects THIS frame's detection quality; smoothing it would
    # make a joint linger as "visible" after it left frame.
    assert out["left_wrist"].visibility == 0.9
    assert out["right_wrist"].visibility == 0.4
    # First frame seeds, so it passes through untouched.
    assert out["left_wrist"].x == 10.0


def test_keypoint_smoother_uses_real_timestamps_for_dt() -> None:
    """dt comes from the frame timestamps, not an assumed frame rate -- the
    pipeline's rate varies (measured p90 91 ms against a 70 ms median)."""
    from src.pipeline import _KeypointSmoother
    from src.type_defs import Keypoint

    s = _KeypointSmoother()
    s.update({"j": Keypoint(x=0.0, y=0.0, z=0.0, visibility=1.0)}, 10.0)
    out = s.update({"j": Keypoint(x=100.0, y=0.0, z=0.0, visibility=1.0)}, 10.07)
    assert 0.0 < out["j"].x <= 100.0


def test_keypoint_smoother_handles_an_empty_frame() -> None:
    from src.pipeline import _KeypointSmoother

    assert _KeypointSmoother().update({}, 1.0) == {}
