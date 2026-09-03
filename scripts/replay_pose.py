#!/usr/bin/env python3
"""Drive the Webots robot from a scripted or recorded motion, with no camera.

    python scripts/replay_pose.py --motion wave
    python scripts/replay_pose.py --motion march --duration 20
    python scripts/replay_pose.py --from-log logs/run_20260903_105803/pose_keypoints.csv
    python scripts/replay_pose.py --list

Why
---
PRD US-2 ("replay a recorded video and have the robot imitate the same motion
deterministically for evaluation") and acceptance criterion 3 both need a
repeatable input. A webcam is neither: it needs a person present, and the same
person never produces the same motion twice, so a fidelity number measured
against it cannot be compared across runs.

This sends the SAME UDP packets the live pipeline sends -- it reuses
``WebotsBridge``, ``GaitCueExtractor`` and ``RetargetingMapper`` rather than
reimplementing the wire format, so the controller runs its ordinary code path
and there is no second format to keep in sync.

The scripted motions are exact by construction, which is what makes them useful
for NFR-3: the human joint angles are KNOWN, so robot-vs-human error is
measurable. That is impossible with a camera, where the ground truth is only
ever another estimate.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Iterator

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.perception.gait_cues import GaitCueExtractor  # noqa: E402
from src.perception.synthetic_pose import BodyState, build_pose  # noqa: E402
from src.retargeting.mapper import RetargetingMapper, default_joint_limits  # noqa: E402
from src.type_defs import Keypoint, PoseFrame  # noqa: E402
from src.webots_bridge import WebotsBridge  # noqa: E402


def _ease(phase: float) -> float:
    """Smooth 0 -> 1 -> 0 over one cycle. Sinusoidal rather than triangular so
    the velocity is continuous: a rate-limited robot joint turns a corner in the
    input into a lag, which would show up in a fidelity metric as the robot's
    fault rather than the input's."""
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)


# --------------------------------------------------------------------- motions
def m_wave(t: float) -> BodyState:
    """Right arm held up, forearm swinging -- the classic 'does it mirror me'."""
    swing = math.sin(2.0 * math.pi * 0.8 * t)
    return BodyState(
        right_arm_side=math.radians(115.0),
        right_elbow=math.radians(45.0 + 35.0 * swing),
        left_arm_side=math.radians(12.0),
    )


def m_arms_forward(t: float) -> BodyState:
    """Both arms rise forward to horizontal and back. Drives shoulder PITCH,
    the one arm channel that is unambiguous at every point of its range (a
    lateral raise passes through NAO's shoulder gimbal -- see MIN_COS_ROLL)."""
    u = _ease((t * 0.25) % 1.0)
    return BodyState(
        left_arm_fwd=math.radians(90.0) * u,
        right_arm_fwd=math.radians(90.0) * u,
    )


def m_squat(t: float) -> BodyState:
    """Symmetric crouch. Both feet stay planted, so the robot must sink without
    stepping -- the lower-body path with the CoM gate fully engaged."""
    return BodyState(crouch=_ease((t * 0.2) % 1.0))


def m_march(t: float) -> BodyState:
    """Alternating knee lifts at ~1 Hz -- what GaitCueExtractor is looking for."""
    phase = (t * 1.0) % 1.0
    lift = math.radians(65.0)
    knee = math.radians(85.0)
    # _ease runs 0 -> 1 -> 0 across each half-cycle, so one leg completes a full
    # lift-and-plant while the other stays down: never both at once.
    if phase < 0.5:
        u = _ease(phase * 2.0)
        return BodyState(left_hip_flex=lift * u, left_knee_flex=knee * u)
    u = _ease((phase - 0.5) * 2.0)
    return BodyState(right_hip_flex=lift * u, right_knee_flex=knee * u)


def m_turn(t: float) -> BodyState:
    """Subject rotates on the spot -- exercises the yaw servo / turn clips."""
    return BodyState(body_yaw=math.radians(75.0) * math.sin(2.0 * math.pi * 0.12 * t))


def m_look(t: float) -> BodyState:
    """Head yaw and pitch only -- the channel HeadGeometry calibrates."""
    return BodyState(
        head_yaw=math.radians(35.0) * math.sin(2.0 * math.pi * 0.3 * t),
        head_pitch=math.radians(25.0) * math.sin(2.0 * math.pi * 0.2 * t),
    )


def m_tpose(_t: float) -> BodyState:
    """Static reference pose -- if the robot moves at all here, something drifts."""
    return BodyState(left_arm_side=math.radians(90.0), right_arm_side=math.radians(90.0))


MOTIONS = {
    "wave": m_wave,
    "arms-forward": m_arms_forward,
    "squat": m_squat,
    "march": m_march,
    "turn": m_turn,
    "look": m_look,
    "t-pose": m_tpose,
}


def synthetic_frames(motion: str, fps: float, duration: float) -> Iterator[PoseFrame]:
    fn = MOTIONS[motion]
    n = max(1, int(round(fps * duration)))
    for i in range(n):
        t = i / fps
        yield build_pose(fn(t), timestamp_s=t, frame_index=i)


def logged_frames(path: str) -> Iterator[PoseFrame]:
    """Replay a ``pose_keypoints.csv`` written by a previous live run.

    This is the true US-2 path: the exact landmarks a real human produced,
    replayed bit-for-bit as often as needed.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        names = sorted({
            key.rsplit("_", 1)[0] for key in (reader.fieldnames or [])
            if key.endswith(("_x", "_y", "_z"))
        })
        for i, row in enumerate(reader):
            kps = {}
            for name in names:
                try:
                    vis = float(row[f"{name}_visibility"])
                    if vis <= 0.0:
                        continue
                    kps[name] = Keypoint(
                        x=float(row[f"{name}_x"]), y=float(row[f"{name}_y"]),
                        z=float(row[f"{name}_z"]), visibility=vis,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            yield PoseFrame(
                timestamp_s=float(row.get("timestamp_s") or i),
                frame_index=int(row.get("frame_index") or i),
                keypoints=kps,
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="replay_pose",
        description="Stream a scripted or recorded human motion to the Webots controller.",
    )
    p.add_argument("--motion", choices=sorted(MOTIONS), default="wave",
                   help="Scripted motion to replay (default: wave).")
    p.add_argument("--from-log", default=None,
                   help="Replay a pose_keypoints.csv from a previous run instead.")
    p.add_argument("--list", action="store_true", help="List the scripted motions and exit.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=15.0, help="Seconds (scripted only).")
    p.add_argument("--loop", action="store_true", help="Repeat until interrupted.")
    p.add_argument("--dry-run", action="store_true",
                   help="Solve and print, send nothing. Works with no Webots running.")
    args = p.parse_args(argv)

    if args.list:
        for name, fn in sorted(MOTIONS.items()):
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:14s} {summary}")
        return 0

    bridge = None if args.dry_run else WebotsBridge(host=args.host, port=args.port)
    mapper = RetargetingMapper(default_joint_limits())
    gait = GaitCueExtractor()
    period = 1.0 / max(args.fps, 1e-6)

    source = "log " + args.from_log if args.from_log else f"motion '{args.motion}'"
    print(f"Replaying {source} at {args.fps:g} FPS -> "
          f"{'(dry run)' if args.dry_run else f'{args.host}:{args.port}'}")

    sent = 0
    try:
        while True:
            frames = (
                logged_frames(args.from_log) if args.from_log
                else synthetic_frames(args.motion, args.fps, args.duration)
            )
            for pose in frames:
                started = time.perf_counter()
                cue = gait.update(pose)
                command = mapper.map_pose(pose)
                if bridge is not None:
                    bridge.send_pose_frame(command, pose.keypoints, gait=cue.as_dict())
                sent += 1
                if sent % 30 == 0 or args.dry_run:
                    print(f"  frame {sent:5d}  landmarks={len(pose.keypoints):2d}  "
                          f"gait={cue.state:5s}  yaw={math.degrees(cue.body_yaw_rad):+6.1f}deg",
                          flush=True)
                if not args.dry_run:
                    remaining = period - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if bridge is not None:
            bridge.close()
    print(f"Sent {sent} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
