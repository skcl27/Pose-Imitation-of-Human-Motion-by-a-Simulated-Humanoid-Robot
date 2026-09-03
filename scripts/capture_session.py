#!/usr/bin/env python3
"""Capture a camera session frame by frame, in 2D and 3D, and prove it ran on the GPU.

    python scripts/capture_session.py --frames 600          # ~40 s at 15 FPS
    python scripts/capture_session.py --frames 300 --label squat

Why this exists
---------------
The live pipeline logs 3D keypoints and nothing else, which is not enough to
answer the two questions that actually block this project:

1. "Is it using the GPU?" -- ``require_gpu`` proves a card is USABLE, but not
   that MeTRAbs' own inference landed there. This samples nvidia-smi while the
   model runs and reports per-frame inference time, so CPU execution is
   impossible to miss: on this machine GPU inference is 36-76 ms/frame and CPU
   is seconds.

2. "Is the camera seeing enough to drive a robot?" -- that needs the 2D PIXEL
   position of every joint alongside its 3D estimate, plus the detector's box
   and confidence. Only the 3D was ever logged, so "the ankles were out of
   frame" was invisible in the logs and had to be inferred by reprojecting
   them afterwards. Here both are written per frame.

Output
------
``logs/capture_<label>_<stamp>/frames.csv`` -- one row per frame: timing, the
detector box and confidence, and for every joint its 3D (mm, camera frame), its
2D projection (px), its visibility proxy, and whether it fell inside the image.
``meta.json`` -- device placement, intrinsics, GPU samples, and the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402

from src.perception import metrabs_model  # noqa: E402
from src.perception.landmarks import POSE_LANDMARKS  # noqa: E402
from src.perception.pose_estimator import PoseEstimator  # noqa: E402
from src.perception.video_input import VideoSource  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# Joints that decide whether this session can drive the robot at all. The legs
# are what a leg lift needs; the shoulders gate the torso frame that every arm
# angle is measured in (nao_retarget._torso_frame returns None without them).
CRITICAL = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


class GpuSampler(threading.Thread):
    """Poll nvidia-smi in the background so utilisation can be proved, not assumed."""

    def __init__(self, period_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.period_s = period_s
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4, check=False,
                ).stdout.strip()
            except Exception:  # noqa: BLE001 - a sampler must never kill the capture
                continue
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    try:
                        self.samples.append({
                            "gpu": int(parts[0]),
                            "util_pct": float(parts[1]),
                            "mem_used_mib": float(parts[2]),
                        })
                    except ValueError:
                        pass

    def stop(self) -> None:
        self._stop.set()


def project(x: float, y: float, z: float, k) -> tuple[float, float]:
    if z <= 1e-6:
        return (float("nan"), float("nan"))
    return (float(k[0, 0]) * x / z + float(k[0, 2]),
            float(k[1, 1]) * y / z + float(k[1, 2]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="capture-session")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "default.yaml"))
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--label", default="session")
    ap.add_argument("--progress-every", type=float, default=2.0,
                    help="seconds between live framing reports")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    # ---------------------------------------------------------------- GPU first
    print("=" * 72)
    print("GPU CHECK")
    print("=" * 72)
    try:
        metrabs_model.require_gpu()
    except metrabs_model.MetrabsUnavailableError as exc:
        print(f"REFUSED: {exc}")
        return 2

    tf = metrabs_model.require_tensorflow()
    gpus = tf.config.list_physical_devices("GPU")
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((256, 256)), tf.ones((256, 256)))
    device = probe.device
    print(f"  TensorFlow GPUs      : {[g.name for g in gpus]}")
    print(f"  test op placed on    : {device}")
    if "GPU" not in device.upper():
        print("  REFUSED: ops are landing on CPU.")
        return 2

    width = int(cfg.get("input.width", 1920))
    height = int(cfg.get("input.height", 1080))
    flip = bool(cfg.get("input.flip_horizontal", True))

    estimator = PoseEstimator(
        use_metrabs=True,
        model_url=str(cfg.get("pose.model_url", metrabs_model.DEFAULT_MODEL_URL)),
        skeleton=str(cfg.get("pose.skeleton", metrabs_model.DEFAULT_SKELETON)),
        default_fov_degrees=float(cfg.get("pose.default_fov_degrees", 55.0)),
        detector_threshold=float(cfg.get("pose.detector_threshold", 0.3)),
        num_aug=int(cfg.get("pose.num_aug", 1)),
        max_detections=int(cfg.get("pose.max_detections", 1)),
        detect_interval=int(cfg.get("pose.detect_interval", 2)),
        box_padding=float(cfg.get("pose.box_padding", 0.18)),
        max_joint_jump_mm=float(cfg.get("pose.max_joint_jump_mm", 300.0)),
        require_gpu=True,
        allow_synthetic_fallback=False,
    )
    if not estimator.is_real:
        print("REFUSED: estimator fell back to synthetic; it would not track you.")
        return 2

    k = estimator.intrinsics_for(width, height)
    fov = float(cfg.get("pose.default_fov_degrees", 55.0))
    vert_fov = 2.0 * math.degrees(math.atan((height / 2.0) / float(k[1, 1])))
    print(f"  intrinsics           : focal {float(k[0,0]):.1f} px, "
          f"horizontal FOV {fov:.1f} deg, vertical FOV {vert_fov:.1f} deg")
    print()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO, "logs", f"capture_{args.label}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    names = list(POSE_LANDMARKS)
    header = ["frame", "wall_t", "dt_ms", "infer_ms", "detected",
              "box_left", "box_top", "box_w", "box_h", "box_conf"]
    for n in names:
        header += [f"{n}_x", f"{n}_y", f"{n}_z", f"{n}_u", f"{n}_v",
                   f"{n}_vis", f"{n}_in_image"]

    cap = VideoSource(source=int(cfg.get("input.source", 0)),
                      width=width, height=height,
                      preferred_fps=float(cfg.get("runtime.initial_fps", 30)))
    sampler = GpuSampler()
    sampler.start()

    print("=" * 72)
    print(f"CAPTURING {args.frames} frames -> {out_dir}")
    print("Stand so your WHOLE body is visible, head to feet.")
    print("=" * 72)

    rows_written = 0
    infer_ms: list[float] = []
    detected_frames = 0
    in_image_counts = {n: 0 for n in names}
    vis_pass_counts = {n: 0 for n in names}
    frames_seen = 0
    last_report = time.time()
    prev_t = None

    try:
        with open(os.path.join(out_dir, "frames.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for frame in cap.read_loop(target_period_s=lambda: 0.0):
                if frames_seen >= args.frames:
                    break
                frames_seen += 1
                img = cv2.flip(frame.image_bgr, 1) if flip else frame.image_bgr

                t0 = time.perf_counter()
                pose = estimator.estimate(img, frame.timestamp_s, frame.frame_index)
                dt_infer = (time.perf_counter() - t0) * 1000.0
                infer_ms.append(dt_infer)

                now = time.time()
                dt_ms = 0.0 if prev_t is None else (now - prev_t) * 1000.0
                prev_t = now

                box = getattr(estimator, "_tracked_box", None)
                bl, bt, bw, bh, bc = ("", "", "", "", "")
                if box is not None and len(box) >= 5:
                    bl, bt, bw, bh, bc = (f"{float(box[i]):.2f}" for i in range(5))

                has = bool(pose.keypoints)
                if has:
                    detected_frames += 1
                row = [frame.frame_index, f"{now:.4f}", f"{dt_ms:.1f}",
                       f"{dt_infer:.1f}", int(has), bl, bt, bw, bh, bc]
                for n in names:
                    kp = pose.keypoints.get(n) if has else None
                    if kp is None:
                        row += ["", "", "", "", "", "", ""]
                        continue
                    u, v = project(kp.x, kp.y, kp.z, k)
                    inside = int(0 <= u < width and 0 <= v < height)
                    if inside:
                        in_image_counts[n] += 1
                    if kp.visibility >= 0.5:
                        vis_pass_counts[n] += 1
                    row += [f"{kp.x:.2f}", f"{kp.y:.2f}", f"{kp.z:.2f}",
                            f"{u:.1f}", f"{v:.1f}", f"{kp.visibility:.4f}", inside]
                w.writerow(row)
                rows_written += 1

                if now - last_report >= args.progress_every:
                    last_report = now
                    crit_in = sum(in_image_counts[n] for n in CRITICAL)
                    crit_tot = max(1, detected_frames * len(CRITICAL))
                    pct = 100.0 * crit_in / crit_tot
                    depth = ""
                    if has and "left_hip" in pose.keypoints:
                        depth = f" depth={pose.keypoints['left_hip'].z/1000:.2f}m"
                    state = "GOOD" if pct > 85 else ("PARTIAL" if pct > 40 else "BAD")
                    print(f"  [{frames_seen:4d}/{args.frames}] {state} "
                          f"critical-joints-in-frame={pct:5.1f}% "
                          f"infer={dt_infer:5.1f}ms{depth}", flush=True)
    except KeyboardInterrupt:
        print("\n  interrupted -- keeping what was captured")
    finally:
        sampler.stop()
        cap.release()

    # ------------------------------------------------------------------ summary
    import statistics as st

    per_gpu: dict[int, list[float]] = {}
    per_gpu_mem: dict[int, list[float]] = {}
    for s in sampler.samples:
        per_gpu.setdefault(s["gpu"], []).append(s["util_pct"])
        per_gpu_mem.setdefault(s["gpu"], []).append(s["mem_used_mib"])

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  frames captured      : {rows_written}")
    print(f"  frames with a person : {detected_frames} "
          f"({100.0*detected_frames/max(1,rows_written):.1f}%)")
    if infer_ms:
        print(f"  inference ms         : median {st.median(infer_ms):.1f}, "
              f"p90 {sorted(infer_ms)[int(0.9*len(infer_ms))]:.1f} "
              f"-> {1000.0/max(st.median(infer_ms),1e-6):.1f} FPS ceiling")
        print("    (GPU here is 36-76 ms/frame; CPU would be seconds)")
    for g in sorted(per_gpu):
        print(f"  GPU {g} utilisation    : median {st.median(per_gpu[g]):.0f}%, "
              f"peak {max(per_gpu[g]):.0f}%, "
              f"mem peak {max(per_gpu_mem[g]):.0f} MiB")

    print()
    print(f"  {'joint':18s} {'in-image':>9s} {'vis>=0.5':>9s}")
    worst = []
    for n in CRITICAL:
        a = 100.0 * in_image_counts[n] / max(1, detected_frames)
        b = 100.0 * vis_pass_counts[n] / max(1, detected_frames)
        print(f"  {n:18s} {a:8.1f}% {b:8.1f}%")
        if a < 80.0:
            worst.append((n, a))
    print()
    if worst:
        print("  BLOCKED: these joints are not reliably in frame --")
        for n, a in worst:
            print(f"    {n} ({a:.0f}%)")
        print("  Nothing downstream can imitate a joint the camera cannot see.")
    else:
        print("  All critical joints in frame: this session is usable for fitting.")

    meta = {
        "device": device,
        "gpus": [g.name for g in gpus],
        "intrinsics": {"focal_px": float(k[0, 0]), "cx": float(k[0, 2]),
                       "cy": float(k[1, 2]), "fov_h_deg": fov,
                       "fov_v_deg": vert_fov},
        "width": width, "height": height, "flip_horizontal": flip,
        "frames": rows_written, "detected": detected_frames,
        "infer_ms_median": st.median(infer_ms) if infer_ms else None,
        "gpu_util": {str(g): {"median": st.median(v), "peak": max(v)}
                     for g, v in per_gpu.items()},
        "in_image_pct": {n: 100.0 * in_image_counts[n] / max(1, detected_frames)
                         for n in names},
        "vis_pass_pct": {n: 100.0 * vis_pass_counts[n] / max(1, detected_frames)
                         for n in names},
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n  wrote {out_dir}/frames.csv and meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
