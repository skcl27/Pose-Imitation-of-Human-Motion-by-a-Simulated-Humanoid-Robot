"""Frame-to-frame smoothing for pose data.

MeTRAbs estimates every frame independently, so raw keypoints jitter and
something has to damp them. The question is what that damping costs in LAG,
because the robot is being driven in real time.

A fixed exponential moving average cannot win that trade. Its group delay is
``(1 - alpha) / alpha`` samples REGARDLESS of what the subject is doing, so the
alpha that is gentle enough to kill jitter while standing still is also the
alpha that makes a fast arm arrive late. Measured on this pipeline at its real
rate (14.3 FPS, 70.1 ms/frame):

    alpha 0.5   -> 1.00 frames ->  70 ms
    alpha 0.35  -> 1.86 frames -> 130 ms

70 ms is 47% of ``runtime.latency_budget_ms`` (150 ms) spent on the primary
keypoint channel alone, before inference is counted.

``OneEuroFilter`` breaks the trade by making the cutoff a function of speed
(Casiez, Roussel & Vogel, "1e Filter", CHI 2012): heavy smoothing when the
subject is still, so jitter dies; a cutoff that rises with measured velocity
when they move, so a fast motion passes through nearly unfiltered. That is the
right shape for pose imitation -- jitter matters when holding a pose, latency
matters when throwing a punch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ExponentialSmoother:
    """Fixed-alpha EMA. Group delay ``(1 - alpha) / alpha`` samples, always.

    Kept for the legacy joint-angle fallback channel and as the comparison
    baseline in the tests; ``OneEuroFilter`` is what the live keypoint path
    uses.
    """

    alpha: float = 0.3
    _state: dict[str, float] = field(default_factory=dict)

    def update(self, values: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in values.items():
            prev = self._state.get(key, value)
            smoothed = self.alpha * value + (1.0 - self.alpha) * prev
            self._state[key] = smoothed
            out[key] = smoothed
        return out


def _alpha_for(cutoff_hz: float, dt: float) -> float:
    """EMA alpha realising a given low-pass cutoff at sample interval ``dt``."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


@dataclass
class OneEuroFilter:
    """Speed-adaptive low-pass filter, keyed by name like ``ExponentialSmoother``.

    The cutoff is ``min_cutoff + beta * |velocity|``, so:

    * standing still -- velocity ~0, cutoff falls to ``min_cutoff``, and the
      filter is as smooth as a heavy EMA;
    * moving fast -- cutoff rises past the Nyquist rate and the filter becomes
      effectively transparent, so the motion is not delayed.

    Units matter for ``beta``: keypoints are MILLIMETRES, so a brisk limb runs
    at 1000-2000 mm/s. ``beta = 0.005`` turns that into +5 to +10 Hz of cutoff,
    which at a 14.3 FPS sample rate (Nyquist 7.2 Hz) is the difference between
    "smoothed" and "pass-through" -- exactly the intended behaviour. Raise
    ``beta`` if the robot still feels late, raise ``min_cutoff`` if a held pose
    still jitters.
    """

    min_cutoff: float = 1.0
    beta: float = 0.005
    d_cutoff: float = 1.0
    _x: dict[str, float] = field(default_factory=dict)
    _dx: dict[str, float] = field(default_factory=dict)

    def update(self, values: dict[str, float], dt: float) -> dict[str, float]:
        """Filter one sample per key. ``dt`` is the interval since the last call.

        A non-finite or non-positive ``dt`` (first frame, a stalled camera, a
        clock that went backwards) would make the derivative meaningless, so the
        sample passes through and seeds the state instead.
        """
        usable = math.isfinite(dt) and dt > 0.0
        out: dict[str, float] = {}
        for key, value in values.items():
            prev_x = self._x.get(key)

            # A non-finite sample must never reach the state. Storing it would
            # make every later frame NaN too -- one bad detection would freeze
            # that joint for the rest of the run rather than for one frame.
            if not math.isfinite(value):
                out[key] = prev_x if prev_x is not None else value
                continue

            if prev_x is None or not usable:
                self._x[key] = value
                self._dx[key] = 0.0
                out[key] = value
                continue

            # Low-pass the derivative before it is allowed to steer the cutoff,
            # so a single noisy frame cannot punch the filter wide open.
            raw_dx = (value - prev_x) / dt
            prev_dx = self._dx.get(key, 0.0)
            a_d = _alpha_for(self.d_cutoff, dt)
            dx_hat = a_d * raw_dx + (1.0 - a_d) * prev_dx
            self._dx[key] = dx_hat

            cutoff = self.min_cutoff + self.beta * abs(dx_hat)
            a = _alpha_for(cutoff, dt)
            smoothed = a * value + (1.0 - a) * prev_x
            self._x[key] = smoothed
            out[key] = smoothed
        return out

    def reset(self) -> None:
        self._x.clear()
        self._dx.clear()
