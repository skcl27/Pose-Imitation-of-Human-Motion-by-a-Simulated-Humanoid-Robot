#!/usr/bin/env python3
"""Live "can the camera see me?" feedback, so a usable stance can be FOUND.

    python scripts/framing_assist.py

Prints one line whenever the situation changes: whether a person is detected,
where they sit in the frame, how far away they are, and which of the joints the
robot actually needs are being cut off. Emits at most one line every
``--min-interval`` seconds so it can be tailed or piped into a notifier.

Why this exists
---------------
A capture of 600 frames returned zero detections while the operator believed
they were "standing in front of the camera" -- the camera was aimed across the
room at a desk. Describing the room does not fix that; a signal that changes as
you walk does. This is the fastest way from "nothing detected" to "usable data",
and every downstream step (fitting, retargeting, driving the robot) is blocked
until that happens.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402

from src.perception import metrabs_model  # noqa: E402
from src.perception.pose_estimator import PoseEstimator  # noqa: E402
from src.perception.video_input import VideoSource  # noqa: E402
from src.utils.config import load_config  # noqa: E402

CRITICAL = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="framing-assist")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--min-interval", type=float, default=2.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until killed")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    width = int(cfg.get("input.width", 1920))
    height = int(cfg.get("input.height", 1080))
    flip = bool(cfg.get("input.flip_horizontal", True))

    est = PoseEstimator(
        use_metrabs=True,
        model_url=str(cfg.get("pose.model_url", metrabs_model.DEFAULT_MODEL_URL)),
        skeleton=str(cfg.get("pose.skeleton", metrabs_model.DEFAULT_SKELETON)),
        default_fov_degrees=float(cfg.get("pose.default_fov_degrees", 55.0)),
        detector_threshold=float(cfg.get("pose.detector_threshold", 0.3)),
        num_aug=1,
        max_detections=1,
        detect_interval=1,            # always detect: we are hunting for a person
        require_gpu=True,
        allow_synthetic_fallback=False,
    )
    k = est.intrinsics_for(width, height)
    fx, cx, cy = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])

    cap = VideoSource(source=int(cfg.get("input.source", 0)),
                      width=width, height=height, preferred_fps=30.0)
    print("READY - walk around; this reports the moment you are seen.", flush=True)

    last_msg = ""
    last_emit = 0.0
    started = time.time()
    try:
        for frame in cap.read_loop(target_period_s=lambda: 0.0):
            if args.seconds and (time.time() - started) > args.seconds:
                break
            img = cv2.flip(frame.image_bgr, 1) if flip else frame.image_bgr
            pose = est.estimate(img, frame.timestamp_s, frame.frame_index)

            if not pose.keypoints:
                msg = "NOT DETECTED - nobody in the camera's view"
            else:
                cut = []
                us, vs, zs = [], [], []
                for n in CRITICAL:
                    kp = pose.keypoints.get(n)
                    if kp is None or kp.z <= 1e-6:
                        cut.append(n)
                        continue
                    u = fx * kp.x / kp.z + cx
                    v = fx * kp.y / kp.z + cy
                    us.append(u)
                    vs.append(v)
                    zs.append(kp.z / 1000.0)
                    if not (0 <= u < width and 0 <= v < height):
                        cut.append(n)
                if not us:
                    msg = "DETECTED but no usable joints"
                else:
                    n_in = len(CRITICAL) - len(cut)
                    pct = 100.0 * n_in / len(CRITICAL)
                    depth = sum(zs) / len(zs)
                    hints = []
                    mu = sum(us) / len(us)
                    mv = sum(vs) / len(vs)
                    if mu > width:
                        hints.append(f"move LEFT ({mu - width:.0f}px off right edge)")
                    elif mu < 0:
                        hints.append(f"move RIGHT ({-mu:.0f}px off left edge)")
                    elif mu > width * 0.75:
                        hints.append("drift LEFT toward centre")
                    elif mu < width * 0.25:
                        hints.append("drift RIGHT toward centre")
                    if mv > height:
                        hints.append(f"too CLOSE ({mv - height:.0f}px below frame)")
                    need = (1.9 / 2.0) / math.tan(math.atan((height / 2.0) / fx))
                    if depth < need * 0.9:
                        hints.append(f"step BACK (at {depth:.1f}m, need ~{need:.1f}m)")
                    state = "USABLE" if pct >= 100 else (
                        "PARTIAL" if pct >= 50 else "POOR")
                    tail = "; ".join(hints) if hints else "framing OK - hold still"
                    short = ",".join(c.replace("left_", "L.").replace("right_", "R.")
                                     for c in cut[:4])
                    msg = (f"{state} {n_in}/8 joints in frame, depth {depth:.1f}m"
                           + (f", cut: {short}" if cut else "")
                           + f" -> {tail}")

            now = time.time()
            head = msg.split(" ", 1)[0]
            if head != last_msg.split(" ", 1)[0] or (now - last_emit) >= args.min_interval:
                print(msg, flush=True)
                last_msg = msg
                last_emit = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
