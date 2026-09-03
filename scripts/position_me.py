#!/usr/bin/env python3
"""Full-screen live positioning aid: walk until the border turns GREEN.

    python scripts/position_me.py

Shows the camera feed with the detected skeleton drawn on it, a target box
sized to how tall a person must appear to fit head-to-feet, and a border that is
GREEN only when every joint the robot needs is inside the frame. Readable from
across a room, which a terminal is not.

Why this exists
---------------
Three capture attempts produced zero or unusable detections while the operator
believed they were in front of the camera; the camera was aimed across the room.
The blocker is not knowledge, it is feedback latency -- you cannot correct a
stance you cannot see. Everything downstream (fitting, retargeting, driving the
robot) is blocked until a person is reliably in frame, so this comes first.

Press q or Esc to quit.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

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
BONES = (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)
ASSUMED_STATURE_MM = 1750.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="position-me")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--scale", type=float, default=0.5, help="window scale")
    ap.add_argument("--windowed", action="store_true",
                    help="do not go full-screen (default is full-screen so the "
                         "border is readable from where you have to stand)")
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
        num_aug=1, max_detections=1, detect_interval=1,
        require_gpu=True, allow_synthetic_fallback=False,
    )
    k = est.intrinsics_for(width, height)
    fx, cx, cy = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
    # Depth at which a whole person just fits the frame height, with 10% margin.
    need_m = (fx * ASSUMED_STATURE_MM / (height * 0.9)) / 1000.0

    win = "STAND HERE - green border = usable"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(win, int(width * args.scale), int(height * args.scale))
    else:
        # Full-screen on purpose: the whole point is to be legible from the far
        # side of the room, which a half-size window behind an IDE is not. Some
        # window managers (mutter here) ignore the hint on an OpenCV window, so
        # the size is also set explicitly and re-asserted on the first frames --
        # the window does not exist until imshow has run at least once.
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(win, width, height)

    cap = VideoSource(source=int(cfg.get("input.source", 0)),
                      width=width, height=height, preferred_fps=30.0)
    try:
        for frame in cap.read_loop(target_period_s=lambda: 0.0):
            img = cv2.flip(frame.image_bgr, 1) if flip else frame.image_bgr
            pose = est.estimate(img, frame.timestamp_s, frame.frame_index)
            view = img.copy()

            pts: dict[str, tuple[int, int]] = {}
            depth = None
            n_in = 0
            for name, kp in (pose.keypoints or {}).items():
                if kp.z <= 1e-6:
                    continue
                u = fx * kp.x / kp.z + cx
                v = fx * kp.y / kp.z + cy
                pts[name] = (int(u), int(v))
            for name in CRITICAL:
                p = pts.get(name)
                if p and 0 <= p[0] < width and 0 <= p[1] < height:
                    n_in += 1
            if "left_hip" in (pose.keypoints or {}):
                depth = pose.keypoints["left_hip"].z / 1000.0

            for a, b in BONES:
                if a in pts and b in pts:
                    cv2.line(view, pts[a], pts[b], (0, 220, 255), 4)
            for name, p in pts.items():
                col = (0, 0, 255) if name in CRITICAL and not (
                    0 <= p[0] < width and 0 <= p[1] < height) else (0, 255, 120)
                cv2.circle(view, p, 7, col, -1)

            ok = bool(pose.keypoints) and n_in == len(CRITICAL)
            border = (0, 200, 0) if ok else (
                (0, 165, 255) if pose.keypoints else (0, 0, 220))
            cv2.rectangle(view, (0, 0), (width - 1, height - 1), border, 24)

            if not pose.keypoints:
                head = "NOT DETECTED - walk into the camera's view"
            elif ok:
                head = f"USABLE  8/8 joints  depth {depth:.1f}m" if depth \
                    else "USABLE  8/8 joints"
            else:
                head = f"{n_in}/8 joints in frame"
                if depth is not None and depth < need_m * 0.9:
                    head += f"  - STEP BACK (at {depth:.1f}m, need ~{need_m:.1f}m)"

            cv2.rectangle(view, (30, 30), (width - 30, 150), (0, 0, 0), -1)
            cv2.putText(view, head, (50, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        1.9, border, 4, cv2.LINE_AA)
            sub = (f"vertical FOV {2*math.degrees(math.atan((height/2)/fx)):.0f} deg"
                   f" - a whole body needs ~{need_m:.1f} m of standoff")
            cv2.putText(view, sub, (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 255), 3, cv2.LINE_AA)

            cv2.imshow(win, view)
            if not args.windowed and frame.frame_index < 12:
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(win, width, height)
                cv2.moveWindow(win, 0, 0)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
