"""Shared fixtures and clip builders for the NAO pose-imitation tests.

The interesting one here is :func:`cyclic_poses`. Testing cyclic gait playback
needs a clip with a genuine limit cycle in it -- one that repeats to a
microradian AND actually translates the robot under the project's own forward
kinematics -- and the test suite is deliberately hermetic, so it cannot reach for
the clips a Webots install happens to ship. So the clip is generated from a
parametric gait instead: a stance phase in which the hip sweeps monotonically
(the body passing over a planted foot) and a swing phase in which the hip resets
with the knee lifting the foot clear.

That distinction is the whole trick, and getting it wrong is instructive: a
sinusoidal hip swing repeats perfectly and translates NOTHING, because it
advances and retreats by the same amount with the foot on the floor. Measured
with balance.clip_torso_speeds: -4.8 mm per cycle for the sine, +132 mm for the
stance/swing split. A fixture built the first way would have let a detector that
ignores translation pass, which is exactly the bug that would loop a turn clip
and spin the robot forever.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "main", "libraries"))

# The 12 joints a real Webots NAO walk clip declares, in the order it declares
# them. Nothing else: the arms and head keep imitating right through a clip.
CLIP_JOINTS = [
    f"{side}{joint}"
    for side in ("L", "R")
    for joint in ("HipYawPitch", "HipRoll", "HipPitch", "KneePitch",
                  "AnklePitch", "AnkleRoll")
]


def _leg(phase: float, amp: float, swing: float, lift: float, crouch: float):
    """Hip and knee pitch for one leg at ``phase`` in [0, 1)."""
    if phase < 0.5:
        # Stance. The foot is planted and the hip sweeps one way throughout, so
        # the torso travels over it. Monotonic is what makes the clip translate.
        s = phase / 0.5
        hip = -crouch / 2.0 + amp * swing * (-1.0 + 2.0 * s)
        knee = crouch
    else:
        # Swing. The hip returns to where it started while the knee lifts the
        # foot clear, so the reset costs no ground.
        s = (phase - 0.5) / 0.5
        hip = -crouch / 2.0 + amp * swing * (1.0 - 2.0 * s)
        knee = crouch + amp * lift * math.sin(math.pi * s)
    return hip, knee


def cyclic_poses(period: int = 26, cycles: int = 3, lead: int = 8, tail: int = 10,
                 swing: float = 0.20, lift: float = 0.35, crouch: float = 0.9,
                 dt: float = 0.04, translate: bool = True):
    """``[(seconds, {joint: rad})]`` for a clip shaped like a real walk clip.

    ``lead`` keyframes ramp the gait up (not periodic), ``cycles`` x ``period``
    keyframes repeat it exactly, and ``tail`` keyframes ramp it back down -- the
    same squat / accelerate / stride / decelerate / stand shape Cyberbotics
    authors, at the real 40 ms keyframe spacing.

    ``translate=False`` gives the degenerate case: the same periodicity with the
    hip swinging sinusoidally, which marches in place and must NOT be detected as
    a gait cycle.
    """
    poses: list[tuple[float, dict[str, float]]] = []
    time_s = 0.0

    def pose(phase: float, amp: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for side, offset in (("L", 0.0), ("R", 0.5)):
            u = (phase + offset) % 1.0
            if translate:
                hip, knee = _leg(u, amp, swing, lift, crouch)
            else:
                hip = -crouch / 2.0 + amp * swing * math.sin(2.0 * math.pi * u)
                knee = crouch + amp * lift * max(0.0, math.sin(2.0 * math.pi * u))
            out[f"{side}HipPitch"] = hip
            out[f"{side}KneePitch"] = knee
            # Hip + knee + ankle == 0 keeps the torso vertical and the sole flat,
            # which is how Cyberbotics' own clips are built.
            out[f"{side}AnklePitch"] = -(hip + knee)
            out[f"{side}HipRoll"] = 0.0
            out[f"{side}AnkleRoll"] = 0.0
            out[f"{side}HipYawPitch"] = 0.0
        return out

    for i in range(lead):
        poses.append((round(time_s, 3), pose(0.0, i / max(1, lead))))
        time_s += dt
    for _ in range(cycles):
        for k in range(period):
            poses.append((round(time_s, 3), pose(k / period, 1.0)))
            time_s += dt
    for i in range(tail + 1):
        poses.append((round(time_s, 3), pose(0.0, 1.0 - i / max(1, tail))))
        time_s += dt
    return poses


def write_clip(path, poses) -> str:
    """Write ``poses`` as a Webots ``.motion`` file. Returns the path as a str."""
    names = list(poses[0][1])
    lines = ["#WEBOTS_MOTION,V1.0," + ",".join(names)]
    for index, (time_s, angles) in enumerate(poses):
        ms = int(round(time_s * 1000.0))
        stamp = f"{ms // 60000:02d}:{ms // 1000 % 60:02d}:{ms % 1000:03d}"
        lines.append(f"{stamp},Pose{index + 1},"
                     + ",".join(f"{angles[j]:.6f}" for j in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_cyclic_clip(path, **kwargs) -> str:
    """A walk clip with a real gait cycle in it, written to ``path``."""
    return write_clip(path, cyclic_poses(**kwargs))
