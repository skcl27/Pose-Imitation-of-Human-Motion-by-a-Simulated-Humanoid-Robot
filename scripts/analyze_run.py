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
# Mirrors lower_body.LowerBodyParams.sole_tilt_budget and the pelvis-shift clamps
# (com_shift_max_pitch/roll, BalanceParams.max_pitch/roll_corr). Kept as plain
# numbers so this script also reads logs written by older controllers.
SOLE_TILT_BUDGET = 0.05
PELVIS_PITCH_CLAMP = 0.30
PELVIS_ROLL_CLAMP = 0.25
FALL_HEAD_HEIGHT_M = 0.30


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
        over = pct(sum(1 for t in tilts if t > SOLE_TILT_BUDGET + 0.01), len(tilts))
        if st.median(tilts) > 0.05 or over > 5.0:
            out.append(finding(
                "CRITICAL" if over > 20.0 else "WARNING",
                f"the {side} sole is off flat",
                f"|HipRoll + AnkleRoll| exceeded the {SOLE_TILT_BUDGET:.2f} rad budget "
                f"in {over:.1f}% of frames (median {st.median(tilts):.3f}, max "
                f"{max(tilts):.3f}). A tilted sole contacts along one edge: the "
                f"support polygon collapses to that edge and the lateral margin "
                f"goes from ~88 mm to a few mm. Recorded before every lateral fall.",
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
    # The mounting rotation is a property of the ROBOT, so it must be one value
    # for the whole session. More than one means the zero was re-learned after a
    # reset -- and a re-learn only happens on a robot that has just fallen, which
    # is the one posture it must never be learned from. Taking a median across
    # the run averaged those together and reported the result as healthy, which
    # is how a zero 17.4 deg out went unnoticed through an entire session.
    distinct = sorted({round(z, 6) for z in zero})
    if len(distinct) > 1:
        spread = max(distinct) - min(distinct)
        out.append(finding(
            "CRITICAL", "the IMU tilt zero was re-learned mid-run",
            f"{len(distinct)} different zeros were latched this run "
            f"({', '.join(f'{z:+.4f}' for z in distinct[:5])}"
            f"{', ...' if len(distinct) > 5 else ''}), spread {spread:.4f} rad "
            f"({math.degrees(spread):.1f} deg). The mounting rotation cannot "
            f"change, so this is a re-calibration after a fall reset, which "
            f"learns 'upright' from a robot lying on the floor. That displaces "
            f"the CoM estimate by about {0.30 * math.sin(min(spread, 1.5)) * 1000:.0f} mm "
            f"and no later frame can recover from it.",
        ))

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
        low = sum(1 for v in h if v < FALL_HEAD_HEIGHT_M)
        if low:
            out.append(finding(
                "CRITICAL" if pct(low, len(h)) > 5 else "WARNING",
                "the robot spent time on the floor",
                f"head was under {FALL_HEAD_HEIGHT_M:.2f} m above the soles in "
                f"{pct(low, len(h)):.1f}% of "
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


def _pelvis_shift(rows) -> tuple[list[float], list[float]]:
    """Per-frame commanded pelvis shift (pitch, roll) reconstructed from the joints.

    The shift is hip +c / ankle -c on both legs, the crouch is -u on hip and ankle
    alike, and the antisymmetric imitation cancels across the legs, so the mean
    over both legs of (hip - ankle) / 2 is the pitch shift; the same on the roll
    channels (no crouch term) is the roll shift. Works on logs without the
    lb_ff_*/lb_fb_* columns.
    """
    pitch, roll = [], []
    for r in rows:
        vals = [num(r, f"{s}{j}_cmd_rad") for s in ("L", "R")
                for j in ("HipPitch", "AnklePitch", "HipRoll", "AnkleRoll")]
        if any(v is None for v in vals):
            continue
        lhp, lap, lhr, lar, rhp, rap, rhr, rar = vals
        pitch.append(0.5 * (0.5 * (lhp - lap) + 0.5 * (rhp - rap)))
        roll.append(0.5 * (0.5 * (lhr - lar) + 0.5 * (rhr - rar)))
    return pitch, roll


def check_shifter_saturation(rows, out):
    """Both CoM shifters at their clamps together: the signature of every fall on
    2026-09-03 (HipPitch +0.45 / AnklePitch -0.65 = 0.30 + 0.25 - 0.10 crouch)."""
    ff_p, fb_p = col(rows, "lb_ff_pitch"), col(rows, "lb_fb_pitch")
    if ff_p and fb_p:
        ff_r, fb_r = col(rows, "lb_ff_roll"), col(rows, "lb_fb_roll")
        tot_p = [a + b for a, b in zip(ff_p, fb_p, strict=False)]
        tot_r = [a + b for a, b in zip(ff_r, fb_r, strict=False)]
        source = "logged lb_ff_*/lb_fb_* columns"
    else:
        tot_p, tot_r = _pelvis_shift(rows)
        source = "reconstructed from the joint commands (no lb_ff_*/lb_fb_* columns)"
    if not tot_p:
        return
    sat_p = pct(sum(1 for v in tot_p if abs(v) >= 0.9 * PELVIS_PITCH_CLAMP), len(tot_p))
    sat_r = pct(sum(1 for v in tot_r if abs(v) >= 0.9 * PELVIS_ROLL_CLAMP), len(tot_r))
    over_p = pct(sum(1 for v in tot_p if abs(v) > PELVIS_PITCH_CLAMP + 0.02), len(tot_p))
    over_r = pct(sum(1 for v in tot_r if abs(v) > PELVIS_ROLL_CLAMP + 0.02), len(tot_r))
    if over_p > 1.0 or over_r > 1.0:
        out.append(finding(
            "CRITICAL", "the pelvis shift exceeds what ONE clamp allows",
            f"total pelvis shift beyond {PELVIS_PITCH_CLAMP:.2f} rad pitch in "
            f"{over_p:.1f}% of frames and beyond {PELVIS_ROLL_CLAMP:.2f} rad roll in "
            f"{over_r:.1f}% ({source}). Two CoM controllers are being summed on the "
            f"same joints; 0.1 rad is 16 mm of CoM travel, so this is the centre of "
            f"mass being driven outside the feet.",
        ))
    elif sat_p > 10.0 or sat_r > 10.0:
        out.append(finding(
            "WARNING", "the CoM shift sits at its clamp",
            f"pitch shift at >=90% of its clamp in {sat_p:.1f}% of frames, roll in "
            f"{sat_r:.1f}% ({source}). A saturated correction is a loop that has run "
            f"out of authority: the pose is beyond what standing can hold, or the "
            f"loop is pushing the wrong way.",
        ))


def check_tilt_sign_consistency(rows, out):
    """Two independent witnesses to which way is up must agree.

    The Gyro node is mounted unrotated (torso frame) while the InertialUnit is
    rolled 90 deg, so d(imu angle)/dt must correlate POSITIVELY with the gyro on
    each axis; and a balance loop with the right sign moves the pelvis AGAINST a
    growing tilt (forward tilt -> CoM back -> pitch shift decreasing). The single
    most expensive bug on this project -- an inverted TILT_PITCH_SIGN -- would have
    been caught on its first session by either half of this check.
    """
    def corr(a, b):
        n = len(a)
        if n < 50:
            return None
        ma, mb = sum(a) / n, sum(b) / n
        sa = math.sqrt(sum((x - ma) ** 2 for x in a))
        sb = math.sqrt(sum((y - mb) ** 2 for y in b))
        return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=False)) / (sa * sb) \
            if sa > 0 and sb > 0 else None

    for axis, angle_key, rate_key in (("pitch", "imu_pitch", "gyro_pitch_rate"),
                                      ("roll", "imu_roll", "gyro_roll_rate")):
        d, g = [], []
        for a, b in zip(rows, rows[1:], strict=False):
            t0, t1 = num(a, "sim_time_s"), num(b, "sim_time_s")
            v0, v1, rate = num(a, angle_key), num(b, angle_key), num(b, rate_key)
            if None in (t0, t1, v0, v1, rate) or not (0 < t1 - t0 < 0.05):
                continue
            if a.get("imu_zero_roll", "") == "" or abs(v1 - v0) > 0.5 or abs(rate) < 0.2:
                continue
            # Standing frames only: a tumbling robot wraps its Euler angles and
            # the small-angle relation between the two sensors no longer holds.
            head = num(b, "head_height")
            if abs(v1) > 0.5 or (head is not None and head < 0.40):
                continue
            d.append((v1 - v0) / (t1 - t0))
            g.append(rate)
        c = corr(d, g)
        if c is not None and c < 0.2:
            out.append(finding(
                "CRITICAL", f"the IMU and the gyro disagree about {axis}",
                f"corr(d imu_{axis}/dt, gyro) = {c:+.2f} over {len(d)} samples. On this "
                f"proto the gyro is unrotated and the IMU rolled 90 deg; they must "
                f"agree in sign, or the tilt fed to the balance loop is not what "
                f"the robot is doing.",
            ))

    # Does the pelvis shift move AGAINST a growing tilt?
    pitch = col(rows, "imu_pitch")
    shift_p, _ = _pelvis_shift(rows)
    if len(shift_p) == len(pitch) and len(pitch) > 100:
        with_, against = 0, 0
        for i in range(5, len(pitch)):
            p0, p1 = pitch[i - 5], pitch[i]
            if abs(p1) < 0.12 or abs(p1) > 0.6 or abs(p1) <= abs(p0):
                continue                      # only a clearly growing tilt
            dc = shift_p[i] - shift_p[i - 5]
            if abs(dc) < 1e-3:
                continue
            # Forward tilt (+) should be answered by moving the CoM back (dc < 0).
            if (dc > 0) == (p1 > 0):
                with_ += 1
            else:
                against += 1
        n = with_ + against
        if n >= 20 and with_ / n > 0.6:
            out.append(finding(
                "CRITICAL", "the balance correction moves WITH the tilt",
                f"while the pitch was growing, the commanded pelvis shift moved the "
                f"centre of mass in the direction of the tilt in {pct(with_, n):.0f}% "
                f"of {n} samples. That is positive feedback -- an inverted "
                f"TILT_PITCH_SIGN in balance.py, or a model frame that does not "
                f"match the robot.",
            ))


def check_lateral_rocking(rows, out):
    """Feet alternately unloading while 'standing' is a rocking mode the static CoM
    model cannot see; recorded right before the lateral falls."""
    standing = []
    for r in rows:
        h, l, rr = num(r, "head_height"), num(r, "fsr_l"), num(r, "fsr_r")
        if None in (h, l, rr) or h < 0.40 or (l + rr) < 30.0:
            continue
        standing.append((l, rr))
    if len(standing) < 100:
        return
    dur = len(standing) * (frame_dt(rows) or 0.02)
    one_foot = pct(sum(1 for l, rr in standing if min(l, rr) < 3.0), len(standing))
    side, flips = None, 0
    for l, rr in standing:
        share = l / (l + rr)
        cur = "L" if share > 0.8 else ("R" if share < 0.2 else None)
        if cur and side and cur != side:
            flips += 1
        if cur:
            side = cur
    rate = flips / dur if dur else 0.0
    if rate > 0.3 or one_foot > 5.0:
        out.append(finding(
            "WARNING", "the robot rocks from foot to foot while standing",
            f"the load flipped between the feet {flips} times in {dur:.0f}s "
            f"({rate:.2f}/s) and one foot carried under 3 N in {one_foot:.1f}% of "
            f"standing frames. A rocking robot is one whose leg commands change "
            f"faster than its feet can follow -- an unlimited antisymmetric "
            f"imitation, or a CoM shift moving faster than the motors.",
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
                  check_sole_contact, check_shifter_saturation, check_tilt_sign_consistency,
                  check_lateral_rocking, check_leg_layer, check_tilt,
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
