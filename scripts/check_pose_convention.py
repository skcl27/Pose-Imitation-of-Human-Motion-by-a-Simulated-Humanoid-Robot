#!/usr/bin/env python3
"""Measure MeTRAbs' left/right labelling convention against a real person.

    python scripts/check_pose_convention.py            # uses configs/default.yaml
    python scripts/check_pose_convention.py --no-flip  # ignore input.flip_horizontal

Stand squarely facing the camera, arms at your sides, and hold still for a few
seconds.

Why this exists
---------------
src/perception/gait_cues.py's torso-yaw estimator depends on the SIGN of
``left_shoulder.x - right_shoulder.x`` for a front-facing subject. Its own
docstring records that the sign was measured on MediaPipe recordings and says, in
as many words, that it "has not been re-measured against real MeTRAbs output --
worth a quick sanity check ('stand facing the camera, confirm yaw reads ~0') the
first time this runs on the target machine". This is that check, automated, so
the answer is a measurement instead of an assumption.

It matters because the failure is silent and expensive: get the sign wrong and a
person standing still reads as turned 180 degrees, so the controller demands turn
clips forever and never walks -- which is precisely the failure mode gait_cues.py's
docstring describes having already been debugged once.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402

from src.perception import metrabs_model  # noqa: E402
from src.perception.gait_cues import GaitCueExtractor  # noqa: E402
from src.perception.pose_estimator import PoseEstimator  # noqa: E402
from src.perception.video_input import VideoSource  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="check-pose-convention")
    p.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--no-flip", action="store_true",
                   help="Measure WITHOUT the horizontal flip, whatever the config says.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    flip = False if args.no_flip else bool(cfg.get("input.flip_horizontal", True))

    est = PoseEstimator(
        model_url=str(cfg.get("pose.model_url", metrabs_model.DEFAULT_MODEL_URL)),
        skeleton=str(cfg.get("pose.skeleton", metrabs_model.DEFAULT_SKELETON)),
        require_gpu=bool(cfg.get("pose.require_gpu", True)),
        allow_synthetic_fallback=False,
    )
    cap = VideoSource(
        source=0,
        width=int(cfg.get("input.width", 1920)),
        height=int(cfg.get("input.height", 1080)),
    )
    gait = GaitCueExtractor()

    print(f"\nflip_horizontal = {flip}. Stand facing the camera and hold still.\n")
    lateral: list[float] = []
    yaws: list[float] = []
    seen = 0

    try:
        for frame in cap.read_loop(target_period_s=0.0):
            image = cv2.flip(frame.image_bgr, 1) if flip else frame.image_bgr
            pose = est.estimate(image, frame.timestamp_s, frame.frame_index)
            cue = gait.update(pose)
            ls = pose.keypoints.get("left_shoulder")
            rs = pose.keypoints.get("right_shoulder")
            if ls and rs and min(ls.visibility, rs.visibility) > 0.5:
                lateral.append(ls.x - rs.x)
                yaws.append(cue.body_yaw_rad)
                seen += 1
                if seen % 10 == 0:
                    print(f"  {seen:3d} good frames  "
                          f"left.x-right.x = {lateral[-1]:+8.1f} mm  "
                          f"yaw = {math.degrees(yaws[-1]):+7.1f} deg", flush=True)
            if seen >= args.frames or frame.frame_index > args.frames * 8:
                break
    finally:
        cap.release()
        est.close()

    if seen < 5:
        print(f"\nOnly {seen} frames had both shoulders visible -- stand fully in "
              f"frame, well lit, and try again.")
        return 1

    med_lat = statistics.median(lateral)
    med_yaw = math.degrees(statistics.median(yaws))
    positive = sum(1 for v in lateral if v > 0) / len(lateral)

    print("\n" + "=" * 66)
    print(f"frames measured          : {seen}")
    print(f"median (left.x - right.x): {med_lat:+.1f} mm")
    print(f"positive on              : {positive * 100:.0f}% of frames")
    print(f"median reported body yaw : {med_yaw:+.1f} deg")
    print("=" * 66)

    # gait_cues computes atan2(-depth, left.x - right.x), so a POSITIVE lateral
    # term is what makes a front-facing subject read as 0 rather than 180.
    ok_lateral = med_lat > 0
    ok_yaw = abs(med_yaw) < 30.0
    if ok_lateral and ok_yaw:
        print("\nPASS: gait_cues.py's sign convention matches this camera+flip setting.")
        print("      A front-facing subject reads as ~0 deg, as intended.")
        return 0
    print("\nFAIL: the convention does NOT hold with these settings.")
    if not ok_lateral:
        print(f"      left.x - right.x is NEGATIVE ({med_lat:+.1f} mm) for a "
              f"front-facing subject.")
    if not ok_yaw:
        print(f"      A person standing square reads as {med_yaw:+.1f} deg turned, so the "
              f"controller will demand turn clips forever and never walk.")
    print(f"      Fix by flipping input.flip_horizontal (currently {flip}) in "
          f"{args.config}, then re-run --")
    print("      that changes which way round the estimator sees you, without")
    print("      touching the sign in gait_cues.py that the tests pin.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
