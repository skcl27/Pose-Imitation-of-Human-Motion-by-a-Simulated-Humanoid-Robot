# Webots NAO Pose Imitation Controller

Real-time control of the simulated **NAO (H25)** humanoid from human pose
tracking: arms, head, legs, and genuine locomotion — the robot squats when you
squat, lifts the leg you lift, walks across the floor when you walk, and turns
its whole body to face where you face.

---

## 1. Architecture

The Python pipeline is a generic *pose source*. Everything NAO-specific (joint
axes, signs, limits, balance, gait) lives here, on the robot side.

```
Python pipeline (src/)                    Webots controller (this folder)
──────────────────────                    ────────────────────────────────
MediaPipe 33 landmarks                    pose_imitation_controller.py
  ├─ raw landmarks ───────── UDP 8765 ──►    ├─ arms + head
  ├─ gait cues (cadence,                     │    nao_retarget.retarget_upper_body
  │   phase, body_yaw_rad)                   └─ legs: EXACTLY ONE of ↓
  └─ legacy joint angles                          1. locomotion  (motion clips)
     (fallback)                                   2. march engine (in place)
                                                  3. pose imitation (per-leg)
                                                  4. stand (balance only)
                                                        │
                                                   NaoPoseDriver
                                                   clamp → smooth → setPosition
```

**Exactly one layer commands the 12 leg joints on any given simulation step.**
Two at once means they fight each other and the robot falls; that single rule is
what the arbiter in `pose_imitation_controller._drive_legs` exists to enforce.

| Library | Responsibility | Webots-free? |
|---|---|---|
| [`nao_retarget.py`](../../libraries/nao_retarget.py) | landmarks → NAO angles (arms, head, and the closed-form per-leg solve) | ✅ |
| [`lower_body.py`](../../libraries/lower_body.py) | *may* the robot execute this leg pose? weight-shift / lift sequencer | ✅ |
| [`gait.py`](../../libraries/gait.py) | in-place march engine (gait command → leg motion) | ✅ |
| [`balance.py`](../../libraries/balance.py) | model-based CoM balance (FK + link masses + Fibonacci search) | ✅ |
| [`walk_motion.py`](../../libraries/walk_motion.py) | motion-clip discovery, yaw servo, locomotion planning | ✅ |
| [`pose_control_utils.py`](../../libraries/pose_control_utils.py) | `NaoPoseDriver`: limits, smoothing, velocity caps, logging | ✅ |
| `pose_imitation_controller.py` | Webots glue: sockets, devices, motion playback, arbitration | ❌ |

Every library is Webots-free on purpose, so all of the maths is unit-tested on a
dev machine that has no Webots installed (`pytest -q`).

---

## 2. What follows the human

### Upper body

| Human motion | NAO joints | How |
|---|---|---|
| Arm up / down | `ShoulderPitch` | vertical component of the upper-arm direction |
| Arm out sideways | `ShoulderRoll` | lateral component of the same direction |
| Elbow bend | `ElbowRoll` | angle between upper-arm and forearm |
| Head turn / nod | `HeadYaw`, `HeadPitch` | nose vs. shoulder midline |

NAO's 2-DOF shoulder is recovered from the single observed arm direction by
splitting it into a vertical part (pitch) and a lateral part (roll). Depth-free,
so it is robust for a frontal camera.

### Legs — the closed-form per-leg solve

For a thigh at hip roll `φ` and hip pitch `θ`, its direction in the torso frame is

```
R_x(φ) · R_y(θ) · (0,0,−1) = ( −sin θ , sin φ·cos θ , −cos φ·cos θ )
                                ^forward  ^lateral      ^vertical
```

A frontal camera observes the **lateral** and **vertical** components directly —
forward is the foreshortened, unobservable axis — which makes the system exactly
solvable:

```
d  = −up_obs        = cos φ·cos θ
φ  = atan2(lat_obs, d)                  ← abduction, fully observable
θ  = ± acos(d / cos φ)                  ← magnitude from foreshortening
```

The only ambiguity a single frontal view leaves is the **sign** of `θ` (thigh
forward or backward). That comes from the landmark depth `z` with a deadband and
a documented bias toward *forward* — human knee lifts are forward, and NAO's
`HipPitch` range (−88°…+27.7°) is mostly forward anyway.

The shank shares the hip roll and adds `KneePitch` about the same axis, so the
identical solve on knee→ankle yields `θ_hip + θ_knee`; the sole is then levelled
by `AnklePitch = −(θ_hip + θ_knee)` and `AnkleRoll = −φ`. Segment reference
lengths are learned from the stream by a peak-hold tracker normalized by torso
length, so there is **no calibration step** and the result is scale-invariant.

#### When the camera crops your legs

Lift detection needs no knowledge of where the floor is: whichever of your two
**feet** is lower defines the ground line, so the other foot's rise above it is
the lift. That needs both feet in frame — and standing close to a webcam usually
crops you at the shins, which made the lift signal read exactly zero however
high you lifted a leg. So the same relative trick falls back to the **knees**,
which are in frame whenever the hips are (it is also the signal the Python-side
gait detector uses, for the same reason). The status line reports which cue is
live as `lift-cue=feet|knees|none`.

With no ankle in view the knee *bend* is genuinely unobservable, so a lift then
shows as hip flexion — a raised straight leg rather than a folded knee. That is
the honest reading of what the camera can see. Segment lengths self-calibrate the
same way: if an ankle is never seen, the shank borrows the thigh (they are within
a few percent in both the human and NAO) rather than declaring the subject
uncalibrated and discarding the whole lower body.

#### How much of your pose gets through

Two very different things live in "the legs are at different angles", and gating
them the same way was wrong:

* **Mirror-symmetric** — both legs abducting outward (a wider stance), or both
  flexing equally (a squat). By symmetry these move the centre of mass *not at
  all*, and a wider stance makes the support polygon **bigger**. They are safer
  than standing, so they pass at **full authority, 1:1 with you**.
* **Antisymmetric** — both legs rolled the same way (a lean), or one leg forward
  and one back. These do move the CoM over the feet, so they stay limited
  (`asymmetric_gain`).

While a foot is genuinely off the ground the split is dropped: the swing leg is
unloaded and free to take your pose at whatever the safety gate allows, and the
stance leg stays near the balanced crouch because it is carrying the robot.

| You do | Robot does | Limited by |
|---|---|---|
| Squat | 1:1 to 40° hip / 80° knee | knee range, not balance — see below |
| Spread your legs | 1:1 to 22.8° per leg | **the ankle**, not the hip — see below |
| Lean sideways | ~35% of your lean | moves the CoM; gated on purpose |
| Split stance (one leg fwd) | ~35% | same |
| **Raise one leg** | full lift once the weight has transferred | the CoM model (§3) |
| Walk / march | walk clips, or the march engine | §4 |
| **Turn your body** | stepping turn (heading servo) | §5 |

**The squat cap is a joint-range limit, not a stability one.** NAO's thigh and
shank are within 3 mm of the same length, so the crouch posture
(`Hip = −d`, `Knee = +2d`, `Ankle = −d`) keeps the ankle under the hip — and the
CoM over the foot — at *any* depth, with the torso vertical and the soles flat
throughout. The real ceiling is the knee's own 121° range.

**A wide stance is capped by the ankle.** `HipRoll` reaches 45.3° but `AnkleRoll`
only 22.8°, and the ankle is what levels the sole against the hip's abduction.
Past 22.8° the sole cannot be kept flat and the robot ends up standing on the
inner edges of its feet, which tips it. So the usable stance width is set by the
*ankle* range — still about 2.5× the feet's resting separation.

**The squat is one degree of freedom, read off the solve.** Depth is the smallest
reading across two axes and both legs: hip vs. knee (bending at the *waist* also
flexes the hip while the knees stay straight, and NAO has no torso joint for
that), and left vs. right (a raised leg is deeply flexed at both, so averaging
the legs made lifting one knee also squat the robot — the straighter leg is the
one bearing the weight).

---

## 3. Lifting one foot: LOAD → SINGLE → UNLOAD

Raising a foot on a free-standing biped is three actions, not one. Skipping the
first two is why a raised leg used to produce no visible response at all.

```
    human raises a foot
            │
            ▼
   ┌──── LOAD ────┐   lean so the CoM moves over the STANCE foot.
   │              │   The lean sign is PROBED against balance.NaoCoMModel,
   │              │   not hard-coded — a wrong lean sign makes a balance
   │              │   loop tip faster, and probing makes that impossible.
   │              ▼
   │      stance_margin > 0 ?   ← forward kinematics + link masses,
   │              │               plus the foot force sensors when present
   │              ▼
   │  ┌─── SINGLE ───┐  the swing leg follows the human's leg, its
   │  │              │  authority scaled CONTINUOUSLY by that margin
   │  ▼              ▼
   └── UNLOAD ◄──── human lowers the foot / margin lost / torso tilts
            │
            ▼
     symmetric crouch (the proven no-fall baseline)
```

Both `shift` and `lift` are rate-limited blends in `[0, 1]`, so there are no
discrete jumps and no state that can get stuck: the controller can always ramp
back to the exact symmetric crouch.

The commanded posture is `crouch_posture(u)` — the squat whose
`Hip + Knee + Ankle = 0` keeps the torso vertical and the soles flat — **plus**
the human's per-leg *deviation from* that posture, authority-weighted. So the
symmetric part never leaves the proven-stable family, only the asymmetric detail
is gated, and a subject standing still produces a deviation of exactly zero.

Safety gates that stand the robot down: torso tilt past `tilt_abort_rad`,
lower-body landmark confidence below `conf_min`, and "both feet up" (a jump, or
bad tracking — never a step).

The **foot force sensors are a confirmation, not a veto**. They scale the lift
between `fsr_min_gain` and 1.0 as the stance foot's load share rises to
`fsr_load_frac`. They used to veto outright, which meant a sensor reading a
constant 50/50 — uncalibrated, or a proto whose soles barely redistribute —
forbade every step forever: a silent, permanent "raising my leg does nothing".
By the time this gate runs the CoM model has already agreed the weight is over
the stance foot, and the tilt abort is the real safety net. Likewise, without a
CoM model at all (no NumPy) the lift is hard-capped by `ungated_lift_cap` rather
than cancelled.

`margin_full` matters more than it looks: a completed weight transfer yields
about 0.015 m of stance margin, so setting it any higher silently caps the lift
below what you asked for and reads as "the leg only moves a little".

Tuning lives in `LowerBodyParams` in [`lower_body.py`](../../libraries/lower_body.py).

### Tuning for more pose fidelity

If you want the robot to follow you harder, these are the knobs, most useful
first — each trades stability margin for faithfulness:

| Knob | Raise it to… | Cost |
|---|---|---|
| `asymmetric_gain` (0.35) | follow leans and split stances more closely | moves the CoM with no single-foot polygon to verify it against |
| `max_crouch_u` (0.70) | squat deeper | approaches the knee's 121° limit |
| `max_abduction` (0.398) | spread wider | past the ankle's range the soles tilt onto their inner edges |
| `fsr_min_gain` (0.40) | trust the CoM model over the foot sensors | loses the load-transfer cross-check |
| `margin_min` (0.002) | start lifting sooner | starts unloading a foot with less margin |

Still **not** driven from the camera: `ElbowYaw` (forearm twist) and `WristYaw`
are held at their rest angles, so arm *rotation* is not imitated — only the arm's
direction and elbow bend. That is a known gap, not a fault.

---

## 4. Real locomotion

The robot **actually translates across the floor** by playing Webots' own
pre-balanced NAO `.motion` clips (`Forwards`, `TurnLeft60`, …). Those clips are
tuned by Cyberbotics for this exact robot; an online gait good enough to walk a
free-standing NAO is a research project in itself, and a Supervisor
base-teleport explodes the physics.

Two details make the difference between a walk and a stumble:

1. **Velocity caps must be lifted.** `Motion` playback works by calling
   `setPosition` on every joint each step. A motor still limited to ~25 % of its
   maximum velocity cannot reach those keyframes, so the pre-balanced gait
   arrives late at every foot placement and the robot topples.
   `NaoPoseDriver.release_to_motion()` raises the caps and suspends per-joint
   commanding; `reclaim_from_motion()` reseeds the smoothers from the position
   sensors so control returns without a jolt.
2. **Clips play to completion.** A clip boundary is a balanced double-support
   pose — the only safe place to hand control back. Clips are therefore never
   looped and never cut short (except by the tilt abort), which also makes clip
   length the latency of "stop walking". That is why the short
   `Forwards.motion` is preferred over `Forwards50.motion`.

If no clips are found on disk the controller says so in the log and falls back
to the **march engine** (`gait.py`), which tracks your cadence, phase and stop
but marches in place.

### Playback can never hold the body

Because a clip owns the *whole* body, anything that stops it from reporting
"over" would freeze the entire robot, not just the legs. Three guards make that
impossible:

| Guard | Effect |
|---|---|
| **Watchdog** (`MOTION_WATCHDOG_S`, and the clip's own `getDuration()` when Webots reports it) | Suspension is always time-bounded. On expiry the body is taken back and that clip is never used again. |
| **Failure backoff** (`MOTION_MAX_FAILURES`) | A clip that trips the watchdog, hits the tilt abort, or finishes with the robot tipped counts as a failure. After a few, clips are abandoned for the session and the controller says so — falling over repeatedly is worse than never walking. |
| **Settled-start check** (`MOTION_START_MAX_TILT_RAD`) | A clip is only started from an upright, calm robot. Starting one mid-wobble is how a walk becomes a fall. |

Likewise, a raised exception in a control step no longer ends the loop: it is
logged, the body is forced back under our control, and the loop carries on
(`MAX_CONSECUTIVE_ERRORS` bounds how long that can go on). Before this, one
transient error broke out of `run()` into cleanup, which zeroed every motor
velocity — a permanently dead robot from a single bad frame.

---

## 5. Turning: a heading servo, not a gesture

NAO has no torso-yaw joint, so "the human turned round" cannot be imitated by a
joint angle — the robot has to step round. And it cannot be done open-loop
("turned left → play one left-turn clip"), because clip and human turn by
different amounts and the error accumulates.

```
 human torso yaw  ──┐
 (gait_cues: atan2  │   desired = robot_yaw_at_latch
  of the shoulder   ├──►           + (human_yaw − human_yaw_at_latch)
  line's depth      │   error   = wrap_pi(desired − InertialUnit yaw)
  spread over its   │                    │
  lateral extent)   │                    ▼
 robot IMU yaw ─────┘        plan_action → turn_left / turn_right clip
```

Because the error is measured against the robot's **real** heading every step,
the rotation converges despite coarse clips and noisy tracking, and it works
while standing perfectly still. `plan_action` also refuses a clip that would
overshoot (a 60° clip is not fired at a 15° error), so the residual heading error
is bounded by half a clip. Aligning the heading takes priority over walking
forward: walking off along the wrong heading is much harder to undo.

While the stepping turn has not fired yet, a small capped `HipYawPitch` bias
gives immediate visual feedback. It is deliberately tiny — NAO's `HipYawPitch`
axis is canted 45°, so it splays the legs as well as yawing the pelvis — and it
is suppressed mid-step.

If the robot turns the *wrong way* for your camera setup, flip `TURN_SIGN` at the
top of the controller. (One sign knob, rather than a sign buried in the geometry:
with the pipeline's default mirrored selfie preview the robot mirrors the
on-screen figure, which is consistent with how the arms are mapped.)

---

## 6. Configuration

### `LEG_CONTROL` — the one knob that matters

At the top of `pose_imitation_controller.py`:

| Value | Behaviour |
|---|---|
| `"auto"` *(default)* | Full stack: motion clips for real walking/turning, march engine when no clips exist, per-leg pose imitation otherwise. |
| `"pose"` | Per-leg pose imitation only. Squat and single-leg lift work; the robot never leaves its spot. |
| `"engine"` | March engine + pose imitation, never the clips. Marches in place. |
| `"off"` | Legs held in the standing posture. Upper body only. |

### Other tunables

| Name | Meaning |
|---|---|
| `DRIVE_HEAD` | head follows the human head |
| `SWAP_SIDES` | mirror-image mapping (set `True` if left/right feels reversed) |
| `SMOOTHING_ALPHA` | EMA on arm/head targets (higher = snappier) |
| `VELOCITY_SCALE`, `LEG_VELOCITY_FACTOR` | fraction of hardware max velocity |
| `GAIT_SMOOTHING_ALPHA`, `GAIT_LEG_VELOCITY_FACTOR` | the same, for legs while stepping/marching |
| `ENABLE_BALANCE` | model-based CoM feedback |
| `WALK_TIER` | march engine tier (`"march"` / `"step"` / `"stand"`) |
| `TURN_SIGN` | flip the turn direction |
| `TILT_ABORT_RAD`, `TILT_RATE_LEAD_S` | fall detection (the gyro term predicts ahead) |
| `MOTION_WATCHDOG_S` | hard ceiling on how long a clip may own the body |
| `MOTION_MAX_FAILURES` | bad locomotion attempts before clips are abandoned |
| `MOTION_START_MAX_TILT_RAD` | how upright the robot must be to start a clip |
| `MAX_CONSECUTIVE_ERRORS` | failed control steps tolerated before giving up |
| `LOCOMOTION` | `LocomotionParams`: yaw gates, overshoot guard, walk thresholds |
| `MOTION_SEARCH_DIRS_EXTRA` | extra folders to search for `.motion` clips |

Deeper tuning: `LowerBodyParams` in `lower_body.py`, `GaitParams` in `gait.py`,
`BalanceParams` in `balance.py`.

Python side: `configs/default.yaml` (`walk.enabled` is the master switch for
streaming gait/yaw cues at all).

---

## 7. World requirements (do not skip)

`main/worlds/…​.wbt` **must** define the foot/floor contact pair:

```
WorldInfo {
  basicTimeStep 20
  contactProperties [
    ContactProperties {
      material2 "NAO foot material"
      coulombFriction [ 8 ]
      bounce 0
      bounceVelocity 0.003
    }
  ]
}
```

`Nao.proto` tags its soles with the contact material `"NAO foot material"`. With
no matching `ContactProperties`, the sole/floor pair silently falls back to
Webots' default contact (`coulombFriction 1`, bouncy) — the feet **slide and
jitter**, the robot cannot load one foot or take a step, and the leg controller
looks *frozen* because every leg command is absorbed by foot slip. This is the
single most common cause of "the legs do not respond".

`basicTimeStep` must be ≤ 20 ms. Webots' default of 32 ms is too coarse for NAO
leg control and destabilises the pre-balanced walk clips; the controller logs a
warning if it finds a larger value.

---

## 8. Communication protocol

UDP JSON on **port 8765**:

```json
{
  "timestamp_s": 1234567890.123,
  "frame_index": 45,
  "joint_angles_rad": { "LShoulderPitch": 0.5, "RElbowRoll": -1.1 },
  "keypoints": {
    "left_shoulder": [0.40, 0.40, -0.1, 0.99],
    "left_hip":      [0.46, 0.55, -0.1, 0.98],
    "left_knee":     [0.46, 0.73, -0.1, 0.97],
    "left_ankle":    [0.46, 0.91, -0.1, 0.95],
    "left_heel":     [0.45, 0.93, -0.1, 0.93]
  },
  "gait": {
    "state": "march", "cadence_hz": 0.95, "phase": 1.83, "swing_side": 1,
    "intensity": 0.7, "turn": 0.4, "conf": 0.98,
    "body_yaw_rad": 0.42, "yaw_conf": 0.99
  }
}
```

- **`keypoints`** *(preferred)* — MediaPipe landmarks `name → [x, y, z, visibility]`
  in normalized image coordinates (`x` right, `y` down, both 0–1). The controller
  retargets these itself. ~21 landmarks are streamed (head, shoulders, elbows,
  wrists, hips, knees, ankles, **heels and toes**) to keep packets small. The
  heels matter: the controller finds the "ground line" as the lower of the two
  feet, and averaging ankle with heel makes lift detection markedly steadier.
- **`joint_angles_rad`** *(fallback)* — used only when no `keypoints` are present.
- **`gait`** *(optional)* — cadence/phase/stop for the march engine, plus
  `body_yaw_rad` (**an angle**, so the controller can close a heading loop on it)
  and `yaw_conf`, which is independent of `conf` because the yaw needs only the
  shoulders and hips and stays usable when the legs leave the frame.

Additive and backward compatible: an older controller ignores fields it does not
know.

---

## 9. Sensors used

| Device | Used for |
|---|---|
| `inertial unit` | gravity direction for balance **and** the robot's true heading for the turn servo |
| `gyro` | tilt *rate*, as a lead term in fall detection |
| `accelerometer` | enabled for completeness |
| `LFsr`, `RFsr` | per-foot load, confirming a weight transfer before a lift |
| `<joint>S` | position sensors: achieved angles, stuck-motor detection, trajectory log |

NAO's foot sensors are 3-axis (`force-3d`) touch sensors, so their value comes
from **`getValues()`**, not `getValue()`. Reading them as scalars is why an
earlier version silently got no load information and the step gate never saw a
weight transfer. Both APIs are handled, so 1-axis protos work too.

---

## 10. Controllers

### `pose_imitation_controller.py` — **recommended**
The full stack described above. This is the one wired into the world file.

### `pose_imitation_controller_advanced.py`
Same `NaoPoseDriver` core, tuned for **inspecting motors**: smoother settings and
verbose per-joint diagnostics (commanded vs. measured, average error, stuck
flags). Deliberately no locomotion, so the robot stays on the spot and the
numbers stay interpretable — but the legs still do full per-leg pose imitation
through the same `LowerBodyController`, so what you measure here is what runs
there.

### Output: joint-trajectory log (FR-7 / US-3)
With `ENABLE_TRAJECTORY_LOG = True` (default), each run writes
`<project>/logs/webots_joint_trajectory_<epoch>.csv`: per simulation frame,
`wall_time_s, sim_time_s, frame_index` and, for every driven joint, a
`<joint>_cmd_rad` (commanded) and `<joint>_meas_rad` (achieved, from the position
sensor) column. This is what the evaluation step uses for per-joint MAE and
timing. Logging is fully defensive — any I/O error disables it without disturbing
the real-time loop. `logs/` is git-ignored.

---

## 11. Reading the controller log

### At startup — check this block first

```
====================================================================
NAO pose imitation controller ready
  timestep          : 20 ms
  motors / sensors  : 24 / 24  (12 leg joints)
  leg control       : auto
  arms + head       : ON
  leg pose imitation: ON (squat, single-leg lift)
  CoM balance       : ON
  march engine      : ON
  locomotion clips  : forward, turn_left, turn_right
  heading feedback  : ON (InertialUnit)
  foot force sensors: 2
====================================================================
```

Every "the robot does not move" report so far has had a cause that was visible
here — a missing device, a missing NumPy, no motion clips, a coarse timestep —
just buried further down the log. Anything reading `OFF` or `NONE FOUND` is
flagged inline with what it costs you.

### Per 100 frames

```
Frame 400 | sim 49.8 Hz | tracking | legs=pose | 8 joints applied
  legs: stepping (73% of the requested lift)
        mode=single stance=R shift=1.00 lift=0.73 gate=0.77 margin=+0.0145m crouch=0.10 lift-cue=feet
  heading error  -8 deg (servo latched)
```

- `legs=` — which layer is commanding: `pose`, `march:march`, `motion:forward`, `stand`
- `legs:` — **plain-language reason** for what the legs are (not) doing. "The legs
  are not moving" has half a dozen legitimate causes — nobody in frame, legs
  cropped, low confidence, the CoM not yet over the stance foot — and they are
  indistinguishable from a bug unless the controller names the one in play.
- `mode` — `double` / `load` / `single` (§3)
- `gate` — how much lift the CoM model currently permits (0 = do not unload)
- `margin` — signed distance of the CoM inside the stance foot; must be positive to step
- `lift-cue` — `feet` / `knees` / `none`: which landmarks the lift is read from
- `fsr_share` (in the reason text) — the stance foot's measured share of the load
- `heading error` — what the turn servo still wants to rotate

---

## 12. Troubleshooting

**The robot is completely frozen (nothing moves, not even the arms)**

Arms and head are driven independently of the legs, so *nothing* moving means
per-joint commanding is not reaching the motors at all. In order of likelihood:

1. **Read the startup block** (§11). If it never printed, the controller failed
   before its first step — look for a Python traceback in the Webots console. The
   usual cause is Webots' Python interpreter missing a package; set
   `Tools → Preferences → Python command` to the interpreter that has NumPy.
2. **Is a motion clip playing?** The status line shows `legs=motion:…` and
   `0 joints applied` while a clip owns the body. That is normal and bounded by
   the watchdog (§4); if you see it permanently, the watchdog message will say
   so and clips will be dropped.
3. **Is the simulation actually running?** Webots must be playing, not paused,
   and not in fast-forward-without-rendering if you are judging by eye.
4. **Are packets arriving?** The status line says `tracking` when frames are
   arriving and `STALE (holding)` when they are not. `STALE` plus a stationary
   robot means the pipeline is not reaching the controller — check the pipeline
   log for `Webots bridge sending to 127.0.0.1:8765` and that both processes are
   on the same machine.

**Legs do not move at all**
1. `WorldInfo.contactProperties` missing → §7. This is the usual cause.
2. `LEG_CONTROL = "off"` in the controller.
3. Log says `Lower-body pose imitation OFF (...)` → NumPy is missing from the
   Python interpreter Webots is using (`Tools → Preferences → Python command`).
4. Legs out of camera frame: watch `Visible: nn/33` in the pipeline HUD.

**Raised leg produces only a small response**
Read the `legs:` line — it names the limiting factor. `gate=0.00` means the CoM
model refuses to unload the foot: usually the robot has not finished leaning yet
(`shift < 0.75`). If it says *"stepping at reduced authority: foot sensors report
only 50% of the load…"*, the FSRs are not seeing the transfer; raise
`fsr_min_gain` toward 1.0 to trust the CoM model instead.

**Spreading my legs barely widens the stance**
It saturates at 22.8° per leg, which is the ankle's limit for keeping the soles
flat, not a safety gate (see §2). If it stops well short of that, check the
`legs:` reason — a one-sided spread is partly antisymmetric and therefore partly
gated.

**Raised leg produces nothing at all**
Check `lift-cue` in the log. `none` means neither your feet nor your knees are in
frame, so a leg lift is literally invisible — step back from the camera until at
least your knees are visible. `knees` means the feet are cropped: the lift is
detected, but it comes out as a raised straight leg rather than a folded knee
(§2), because the knee bend cannot be seen without an ankle.

**Robot marches instead of walking across the floor**
No `.motion` clips found. Check the startup log for `Locomotion clips found: …`.
Set `$WEBOTS_HOME`, or drop clips into
`main/controllers/pose_imitation_controller/motions/`.

**Robot turns the wrong way** → flip `TURN_SIGN`.

**Left/right mirrored** → set `SWAP_SIDES = True`.

**Robot falls while walking** → confirm `basicTimeStep` is 20 ms and the contact
properties are present; then lower `LOCOMOTION.walk_conf_min` so it walks less
eagerly, or use `LEG_CONTROL = "engine"` to keep it in place. After a few bad
attempts the controller abandons clips by itself and tells you.

**Robot not moving at all**
1. Is the simulation playing, and the controller running?
2. Pipeline log shows `Webots bridge sending to 127.0.0.1:8765`?
3. `netstat -uln | grep 8765` (Linux) — is the port bound?
4. Controller log shows `tracking`, not `STALE (holding)`?

**Jerky motion** → raise `SMOOTHING_ALPHA` for responsiveness or lower it for
smoothness; reduce camera resolution; run Webots without rendering (`--no-rendering`).

---

## 13. Tests

All the maths is Webots-free and unit-tested from the repo root:

```bash
pytest -q
```

| File | Covers |
|---|---|
| `tests/test_nao_retarget.py` | leg solve **round trip** (project known NAO angles → recover them), self-calibration, lift detection, mirrored roll signs, occlusion, garbage input |
| `tests/test_lower_body.py` | the step sequence, the model-probed lean direction, every safety gate, joint limits |
| `tests/test_walk_motion.py` | clip discovery, overshoot guard, hysteresis, yaw servo latch/relatch/wrapping |
| `tests/test_gait_cues.py` | cadence/phase/stop, and torso yaw sign, monotonicity and magnitude |
| `tests/test_gait.py`, `tests/test_balance.py` | march engine invariants, CoM model and Fibonacci search |
| `tests/test_controller_integration.py` | the real controller against a mocked Webots over real UDP: device setup, leg arbitration, motion hand-off, **freeze resistance** (a clip that never ends, a step that raises, repeated bad locomotion) and leg response with the feet cropped out of frame |
