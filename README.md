# Pose Imitation of Human Motion by a Simulated Humanoid Robot 

A Linux-first, real-time pipeline that:

1. Captures live video from a webcam (or replays a video file).
2. Runs **MediaPipe Pose** to detect a human and all **33 body landmarks**.
3. Draws the live skeleton on top of the camera feed in an OpenCV window.
4. Streams the landmarks (plus gait and body-yaw cues) to a **Webots** simulated
   NAO H25 over UDP.
5. The Webots controller retargets them to the full NAO pose and drives the robot
   while keeping it on its feet.

## What the robot does

| You | The robot |
|---|---|
| Move your arms | Follows both arms (shoulder pitch **and** roll, elbow) |
| Turn / nod your head | Follows head yaw and pitch |
| Squat | Squats — symmetric, statically balanced |
| **Raise one leg** | Shifts its weight onto the other foot, *then* raises the matching leg — only as far as its own centre-of-mass model says is safe |
| **Walk / march** | Walks across the floor for real (its world coordinates change), using Webots' pre-balanced NAO walk clips; marches in place if no clips are installed |
| **Turn your body** | Steps round to face the same way, closing the loop on the InertialUnit heading |

The lower body is the interesting part: a single camera cannot see whether NAO's
centre of mass is over a foot, so the camera only ever supplies the *desired* leg
pose and the robot's own forward-kinematics CoM model decides how much of it is
safe to execute. Details and the maths:
[`main/controllers/pose_imitation_controller/README.md`](main/controllers/pose_imitation_controller/README.md).

> Full requirements specification: [`docs/PRD.md`](docs/PRD.md)
> Complete install & run guide (for the target PC): [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)

---

## Quickstart (Ubuntu + Conda)

```bash
# 1. Activate the conda environment (create it first if needed):
#    conda create -n y313 python=3.11 -y
conda activate y313

# 2. Clone or pull
git clone https://github.com/tarikbilla/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.git
cd Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot

# 3. Install dependencies into the conda env
pip install -r requirements.txt

# 4. Camera-only demo (no Webots required) — shows live skeleton overlay:
python run.py --no-webots
```

Press **`q`** or **`ESC`** in the window to quit.

> Full setup guide: [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md)

---

## Repository Layout

```text
.
├── configs/default.yaml              # runtime configuration
├── docs/
│   ├── PRD.md                        # product requirements (incl. 33 landmarks)
│   └── RUN_INSTRUCTIONS.md           # full setup guide for target PC
├── main/                             # Webots project root
│   ├── worlds/…​.wbt                  # world + REQUIRED NAO foot/floor contact
│   ├── controllers/pose_imitation_controller/
│   └── libraries/                    # Webots-free (unit-tested) robot maths
│       ├── nao_retarget.py           # landmarks → NAO angles (incl. per-leg solve)
│       ├── lower_body.py             # weight-shift / single-leg-lift sequencer
│       ├── balance.py                # model-based CoM balance
│       ├── gait.py                   # in-place march engine
│       ├── walk_motion.py            # motion clips + heading (yaw) servo
│       └── pose_control_utils.py     # NaoPoseDriver: limits, smoothing, logging
├── scripts/setup_ubuntu.sh           # one-shot Ubuntu setup
├── src/
│   ├── perception/                   # video input + pose estimation + visualizer
│   ├── retargeting/                  # keypoints → joint angles
│   ├── utils/                        # config, fps controller, smoother, csv logger
│   ├── pipeline.py                   # end-to-end orchestrator
│   ├── run.py                        # CLI entrypoint
│   ├── types.py                      # dataclasses
│   └── webots_bridge.py              # UDP bridge to Webots controller
├── tests/                            # pytest unit tests
├── Makefile                          # make setup | run | demo | test | lint
├── requirements.txt
└── run.py                            # `python run.py`
```

## CLI Flags

```text
python run.py [--config configs/default.yaml]
              [--source 0|path/to/video.mp4]
              [--no-webots]            # skip UDP send (pure perception demo)
              [--no-display]           # headless, no OpenCV window
              [--max-frames N]
              [--log-level INFO|DEBUG|WARNING|ERROR]
```

## Make Targets

| Target | Description |
|---|---|
| `make setup`    | Provision venv + apt deps (Ubuntu). |
| `make run`      | Full pipeline (camera + Webots bridge + window). |
| `make demo`     | Camera + window only (no Webots needed). |
| `make headless` | 100 frames, no window — CI smoke test. |
| `make test`     | Run pytest. |
| `make lint`     | Run ruff. |
| `make format`   | Black + ruff --fix. |

## Tests

```bash
pytest -q
```

All the robot-side maths lives in `main/libraries/` and imports **no** Webots
module, so it is fully unit-tested on a machine without Webots installed.

## Webots setup in one line

Open `main/worlds/Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt`,
press ▶, then run `python run.py`.

Two world settings are **required**, and both are already in the committed world
file — if you build your own world, copy them or the legs will not work:

```
WorldInfo {
  basicTimeStep 20                     # Webots' default 32 ms is too coarse for NAO
  contactProperties [
    ContactProperties {                # Nao.proto tags its soles "NAO foot material";
      material2 "NAO foot material"    # without this pair the feet slide and the
      coulombFriction [ 8 ]            # robot cannot load a foot or take a step
      bounce 0
      bounceVelocity 0.003
    }
  ]
}
```

See [`docs/RUN_INSTRUCTIONS.md`](docs/RUN_INSTRUCTIONS.md) for Webots configuration and troubleshooting.
