#!/usr/bin/env python3
"""Turn a Webots trajectory log into a ranked list of what went wrong.

    python scripts/analyze_run.py                  # newest log
    python scripts/analyze_run.py logs/webots_joint_trajectory_123.csv
    python scripts/analyze_run.py --from 40 --to 55 # one segment, in sim seconds

Exists because the interesting failures on this project are all statistical --
"the head is pinned 40% of the time", "the legs were bit-identical in 100% of
frames", "the support margin was negative in 80%" -- and none of them are visible
by watching the robot or by reading a log by eye. Each check below corresponds to
a defect that actually shipped.

Reads the diagnostic columns the controller writes (IMU, leg mode, support
margin, why the legs did or did not move). Logs without them still work; the
checks that need them are skipped and say so.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics as st
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "main", "libraries"))
from pose_control_utils import get_default_motor_configs  # noqa: E402

CONFIGS = get_default_motor_configs()

# Fraction of frames at a limit / off target before a channel is worth reporting.
SATURATION_WARN = 0.10
TRACKING_WARN_RAD = 0.15
SUSTAINED_S = 0.5


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def col(rows: list[dict], key: str) -> list[float]:
    return [v for v in (num(r, key) for r in rows) if v is not None]


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def finding(sev: str, title: str, detail: str) -> tuple[int, str, str, str]:
    rank = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[sev]
    return (rank, sev, title, detail)


# --------------------------------------------------------------------- checks
def check_saturation(rows, out):
    """A joint on its mechanical stop is not imitating anything."""
    joints = sorted({k[:-8] for k in rows[0] if k.endswith("_cmd_rad")})
    for j in joints:
        values = col(rows, f"{j}_cmd_rad")
        if not values:
            continue
        cfg = CONFIGS.get(j)
        if cfg is None:
            continue
        lo = sum(1 for v in values if v <= cfg.min_angle + 2e-3)
        hi = sum(1 for v in values if v >= cfg.max_angle - 2e-3)
        share = pct(lo + hi, len(values))
        if share >= SATURATION_WARN * 100:
            sev = "CRITICAL" if share > 30 else "WARNING"
            out.append(finding(
                sev, f"{j} is pinned against a hardware stop",
                f"{share:.1f}% of frames ({pct(lo, len(values)):.1f}% at min, "
                f"{pct(hi, len(values)):.1f}% at max). A saturated joint has lost "
                f"the signal: the mapping gain or its reference is wrong, not the robot.",
            ))


def check_tracking(rows, out):
    """A joint that cannot reach its target is blocked, not slow."""
    joints = sorted({k[:-8] for k in rows[0] if k.endswith("_cmd_rad")})
    dt = frame_dt(rows)
    min_run = max(1, int(SUSTAINED_S / dt)) if dt else 25
    for j in joints:
        errs = []
        for r in rows:
            c, m = num(r, f"{j}_cmd_rad"), num(r, f"{j}_meas_rad")
            errs.append(abs(c - m) if (c is not None and m is not None) else 0.0)
        if not errs:
            continue
        mae = st.mean(errs)
        run = worst = 0
        for e in errs:
            run = run + 1 if e > 0.4 else 0
            worst = max(worst, run)
        if worst >= min_run:
            out.append(finding(
                "CRITICAL", f"{j} stopped following its command",
                f"off target by >0.4 rad for {worst * dt:.1f}s continuously "
                f"(MAE {mae:.3f} rad). Sustained error is a mechanical block -- a "
                f"self-collision or a joint fighting another layer -- not lag.",
            ))
        elif mae > TRACKING_WARN_RAD:
            out.append(finding(
                "WARNING", f"{j} tracks its command poorly",
                f"MAE {mae:.3f} rad. Check the velocity cap and whether two "
                f"layers are commanding it.",
            ))


def check_legs_move(rows, out):
    """The defect that hid for a whole project: legs that never move apart."""
    pairs = (("LHipPitch", "RHipPitch"), ("LKneePitch", "RKneePitch"))
    for left, right in pairs:
        same = 0
        total = 0
        for r in rows:
            a, b = num(r, f"{left}_cmd_rad"), num(r, f"{right}_cmd_rad")
            if a is None or b is None:
                continue
            total += 1
            same += abs(a - b) < 1e-9
        if total and pct(same, total) > 95:
            out.append(finding(
                "CRITICAL", f"{left} and {right} are always identical",
                f"{pct(same, total):.1f}% of {total} frames are bit-identical. The "
                f"legs never moved asymmetrically, so no lean, weight shift or "
                f"single-leg lift happened at all.",
            ))
    lean = []
    for r in rows:
        a, b = num(r, "LHipRoll_cmd_rad"), num(r, "RHipRoll_cmd_rad")
        if a is not None and b is not None:
            lean.append(a + b)
    if lean:
        spread = max(lean) - min(lean)
        if spread < 1e-6 and abs(st.median(lean)) > 0.05:
            out.append(finding(
                "CRITICAL", "the robot holds a constant lean",
                f"LHipRoll+RHipRoll is fixed at {st.median(lean):+.3f} rad for the "
                f"whole run. A balance loop varies; a constant means it is jammed "
                f"against a clamp or a limiter.",
            ))
        elif abs(st.median(lean)) > 0.15:
            out.append(finding(
                "WARNING", "the robot leans persistently to one side",
                f"median lean {st.median(lean):+.3f} rad, p95 "
                f"{sorted(lean)[int(0.95 * len(lean))]:+.3f}. Expect drift and a "
                f"reduced support margin on that side.",
            ))


def check_support(rows, out):
    """Is the commanded posture one the robot can actually stand in?"""
    mx, my = col(rows, "support_margin_x"), col(rows, "support_margin_y")
    if not mx or not my:
        out.append(finding("INFO", "no support-margin columns in this log",
                           "Re-run with the current controller to get them."))
        return
    worst = [min(a, b) for a, b in zip(mx, my, strict=False)]
    outside = sum(1 for v in worst if v < 0)
    share = pct(outside, len(worst))
    # A weight transfer legitimately drives the DOUBLE-support margin negative.
    shifting = 0
    for r, v in zip(rows, worst, strict=False):
        s = num(r, "lb_shift")
        if v < 0 and s is not None and s > 1e-3:
            shifting += 1
    unexplained = outside - shifting
    if share > 20:
        out.append(finding(
            "CRITICAL", "the commanded posture is outside the support polygon",
            f"{share:.1f}% of frames (median margin {st.median(worst):+.4f} m). "
            f"{shifting} of those were mid weight-transfer, which is expected; "
            f"{unexplained} were not. Outside the polygon the robot is statically "
            f"falling, whatever the pose looks like.",
        ))
    elif unexplained and pct(unexplained, len(worst)) > 2:
        out.append(finding(
            "WARNING", "some standing frames leave the support polygon",
            f"{pct(unexplained, len(worst)):.1f}% of frames, excluding weight "
            f"transfers. Median margin {st.median(worst):+.4f} m.",
        ))
    else:
        out.append(finding(
            "INFO", "support margin looks healthy",
            f"median {st.median(worst):+.4f} m, min {min(worst):+.4f}, "
            f"{share:.1f}% of frames outside (all mid-transfer: "
            f"{unexplained == 0}).",
        ))


def check_sole_contact(rows, out):
    """A tilted sole stands on an edge and collapses lateral support."""
    for side in ("L", "R"):
        tilts = []
        for r in rows:
            h, a = num(r, f"{side}HipRoll_cmd_rad"), num(r, f"{side}AnkleRoll_cmd_rad")
            if h is not None and a is not None:
                tilts.append(abs(h + a))
        if not tilts:
            continue
        if st.median(tilts) > 0.05:
            out.append(finding(
                "WARNING", f"the {side} sole is habitually off flat",
                f"median tilt {st.median(tilts):.3f} rad, max {max(tilts):.3f}. "
                f"Tilting a sole even 0.05 rad cuts lateral support margin from "
                f"about 88 mm to 14 mm.",
            ))


def check_leg_layer(rows, out):
    """Why did the legs not do what the human did?"""
    whys = [r.get("lb_why") for r in rows if r.get("lb_why")]
    if whys:
        counts: dict[str, int] = {}
        for w in whys:
            counts[w] = counts.get(w, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
        out.append(finding(
            "INFO", "what the lower body said it was doing",
            "; ".join(f"{pct(n, len(whys)):.0f}% {w!r}" for w, n in top),
        ))
    src = [r.get("lb_lift_source") for r in rows if r.get("lb_lift_source")]
    if src:
        none = sum(1 for v in src if v == "none")
        if pct(none, len(src)) > 40:
            out.append(finding(
                "CRITICAL", "the leg-lift signal was usually unavailable",
                f"lift_source was 'none' in {pct(none, len(src)):.1f}% of frames -- "
                f"neither both feet nor both knees were in shot. No software fix: "
                f"the subject has to step back until the knees are in frame.",
            ))
    modes = [r.get("lb_mode") for r in rows if r.get("lb_mode")]
    if modes:
        flips = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
        dur = span(rows)
        if dur and flips / dur > 2.0:
            out.append(finding(
                "WARNING", "the step sequencer is flapping",
                f"{flips} mode changes in {dur:.0f}s ({flips / dur:.1f}/s). Weight "
                f"transfers are aborting part-way; suspect a gate sitting exactly "
                f"where the data is (confidence, tilt, or the lift threshold).",
            ))
        singles = sum(1 for m in modes if m == "single")
        out.append(finding(
            "INFO", "leg-layer time budget",
            ", ".join(f"{m}={pct(sum(1 for x in modes if x == m), len(modes)):.0f}%"
                      for m in ("double", "load", "single"))
            + (f"  (single-support lift achieved: {singles > 0})"),
        ))
    gates = col(rows, "lb_gate")
    if gates and max(gates) < 0.05:
        out.append(finding(
            "CRITICAL", "the centre-of-mass gate never opened",
            "lb_gate stayed at 0, so a leg lift was never authorised. Either the "
            "weight transfer never completed or there is no CoM model loaded "
            "(check for a DEGRADED line at controller startup).",
        ))


def check_imu_zero(rows, out):
    """Is the tilt sensor even telling us which way is up?

    The most expensive bug found on this project: this NAO's InertialUnit reads
    roll = +1.618 rad (93 deg) while the robot stands still with its soles
    carrying all 50 N of its weight. Every tilt gate and the balance loop read
    that as falling over.
    """
    zero = col(rows, "imu_zero_roll")
    raw = col(rows, "imu_roll_raw")
    if not raw:
        return
    if not zero:
        out.append(finding(
            "CRITICAL", "the IMU tilt zero was never learned",
            f"raw roll sat at {st.median(raw):+.3f} rad and no zero was latched, so "
            f"tilt was reported as level all run and the balance loop and every "
            f"tilt gate stayed idle. Are the soles loaded? Calibration only "
            f"accepts samples while the foot sensors say the robot is standing.",
        ))
        return
    magnitude = abs(st.median(zero))
    if magnitude > 0.05:
        out.append(finding(
            "INFO", "this robot's InertialUnit is mounted rotated",
            f"learned tilt zero {st.median(zero):+.3f} rad "
            f"({math.degrees(magnitude):.0f} deg), and it is being corrected for. "
            f"Uncorrected it would displace the balance loop's CoM estimate by "
            f"about {0.18 * magnitude * 1000:.0f} mm and hold every tilt gate shut.",
        ))
    fsr = [a + b for a, b in zip(col(rows, "fsr_l"), col(rows, "fsr_r"), strict=False)]
    if fsr and st.median(fsr) < 20:
        out.append(finding(
            "CRITICAL", "the feet are not carrying the robot",
            f"median sole load {st.median(fsr):.1f} N against a body weight of "
            f"about 51 N. The robot is off its feet -- fallen, or held up by "
            f"something else. Every other finding below is a consequence.",
        ))


def check_falls(rows, out):
    """Did the robot go down, and did it get itself back up?"""
    h = col(rows, "head_height")
    reloads = col(rows, "reloads")
    if h:
        low = sum(1 for v in h if v < 0.25)
        if low:
            out.append(finding(
                "CRITICAL" if pct(low, len(h)) > 5 else "WARNING",
                "the robot spent time on the floor",
                f"head was under 0.25 m above the soles in {pct(low, len(h)):.1f}% of "
                f"frames (min {min(h):+.3f} m; standing is ~0.46, the deepest squat "
                f"0.41). Everything measured during those frames describes a fallen "
                f"robot, not a control problem -- segment them out with --from/--to.",
            ))
    if reloads and max(reloads) > 0:
        out.append(finding(
            "INFO", "automatic fall recoveries",
            f"the controller reset the simulation {int(max(reloads))} time(s) this "
            f"run. Each one is a fall worth explaining.",
        ))


def check_tilt(rows, out):
    """Did the robot actually wobble or go over?"""
    roll, pitch = col(rows, "imu_roll"), col(rows, "imu_pitch")
    if not roll or not pitch:
        return
    worst = [max(abs(a), abs(b)) for a, b in zip(roll, pitch, strict=False)]
    for limit, sev, label in ((0.40, "CRITICAL", "past the abort limit"),
                              (0.28, "WARNING", "past the lower-body stand-down limit"),
                              (0.15, "INFO", "past the clip-start ceiling")):
        share = pct(sum(1 for v in worst if v > limit), len(worst))
        if share > 1.0:
            out.append(finding(
                sev, f"torso tilt went {label}",
                f"|roll| or |pitch| exceeded {limit} rad in {share:.1f}% of frames "
                f"(max {max(worst):.3f}). Median tilt {st.median(worst):.3f}.",
            ))
            break


def check_head(rows, out):
    """The head is the most visible channel, and the easiest to get wrong."""
    for j in ("HeadYaw", "HeadPitch"):
        v = col(rows, f"{j}_cmd_rad")
        if not v:
            continue
        rng = max(v) - min(v)
        if rng < 0.05:
            out.append(finding(
                "WARNING", f"{j} barely moves",
                f"total range {rng:.3f} rad. Either the head landmarks are not "
                f"visible or the solve is returning nothing (it omits the channel "
                f"rather than guessing when the head line is unusable).",
            ))


def check_tracking_live(rows, out):
    stale = col(rows, "stale")
    if stale:
        share = pct(sum(1 for v in stale if v > 0.5), len(stale))
        if share > 25:
            out.append(finding(
                "WARNING", "pose input was stale for much of the run",
                f"{share:.1f}% of frames. The robot holds/stands down rather than "
                f"imitating; check the camera pipeline is running and the subject "
                f"is in frame.",
            ))


def check_heading(rows, out):
    err = col(rows, "yaw_error")
    if not err:
        return
    agree = pct(sum(1 for v in err if abs(v) < 0.05), len(err))
    med = st.median(err)
    if abs(med) > 0.25:
        out.append(finding(
            "WARNING", "the heading loop sits at a fixed offset",
            f"median yaw error {math.degrees(med):+.0f} deg, agreeing within 3 deg "
            f"in only {agree:.0f}% of frames. A one-sided median means the latched "
            f"reference is wrong, not that tracking is noisy.",
        ))


# ------------------------------------------------------------------ plumbing
def frame_dt(rows) -> float:
    t = col(rows, "sim_time_s")
    if len(t) < 3:
        return 0.0
    steps = [b - a for a, b in zip(t, t[1:], strict=False) if 0 < b - a < 1.0]
    return st.median(steps) if steps else 0.0


def span(rows) -> float:
    t = col(rows, "sim_time_s")
    return (t[-1] - t[0]) if len(t) > 1 else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="trajectory CSV (default: newest in logs/)")
    ap.add_argument("--from", dest="t0", type=float, help="start sim time (s)")
    ap.add_argument("--to", dest="t1", type=float, help="end sim time (s)")
    args = ap.parse_args(argv)

    path = args.log
    if path is None:
        candidates = glob.glob(os.path.join(REPO, "logs", "webots_joint_trajectory_*.csv"))
        if not candidates:
            print("No trajectory logs in logs/. Run the controller first.")
            return 2
        path = max(candidates, key=os.path.getmtime)

    rows = load(path)
    if not rows:
        print(f"{path} is empty.")
        return 2
    if args.t0 is not None or args.t1 is not None:
        lo = args.t0 if args.t0 is not None else -math.inf
        hi = args.t1 if args.t1 is not None else math.inf
        rows = [r for r in rows if lo <= (num(r, "sim_time_s") or 0.0) <= hi]
        if not rows:
            print("No frames in that time window.")
            return 2

    dt = frame_dt(rows)
    print("=" * 74)
    print(f"{os.path.basename(path)}")
    print(f"{len(rows)} frames | {span(rows):.1f}s of sim | step {dt * 1000:.0f}ms"
          f" | {'with' if 'support_margin_x' in rows[0] else 'WITHOUT'} controller diagnostics")
    print("=" * 74)

    out: list[tuple[int, str, str, str]] = []
    for check in (check_imu_zero, check_falls, check_tracking_live, check_legs_move, check_support,
                  check_sole_contact, check_leg_layer, check_tilt,
                  check_saturation, check_tracking, check_head, check_heading):
        try:
            check(rows, out)
        except Exception as exc:  # noqa: BLE001 - one bad check must not stop the rest
            out.append(finding("INFO", f"check {check.__name__} failed", str(exc)))

    out.sort(key=lambda f: f[0])
    if not out:
        print("\nNothing flagged.")
    for _, sev, title, detail in out:
        print(f"\n[{sev}] {title}")
        for line in wrap(detail, 70):
            print(f"    {line}")
    print()
    return 0


def wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    sys.exit(main())
