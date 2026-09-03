"""Launch Webots alongside the pose pipeline (PRD FR-9 / acceptance criterion 1).

The PRD asks for the whole demo to come up from a single command. Until now
``run.py`` started only the Python half, so every run needed Webots opened by
hand first and the two halves could silently disagree about which world was
loaded.

Deliberately conservative about what it starts: if a Webots process is already
running, this attaches to it rather than opening a second instance that would
bind nothing, load its own copy of the world, and leave two NAOs fighting over
UDP 8765.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = (
    REPO_ROOT / "main" / "worlds"
    / "Pose-Imitation-of-Human-Motion-by-a-Simulated-Humanoid-Robot.wbt"
)

# Checked in order after $WEBOTS_HOME and $PATH. Mirrors the install locations
# walk_motion.default_motion_search_dirs already knows about, so a machine where
# the motion clips are found is a machine where the binary is found.
FALLBACK_BINARIES = (
    "/usr/local/webots/webots",
    "/usr/share/webots/webots",
    "/opt/webots/webots",
    "/snap/bin/webots",
)


class WebotsLaunchError(RuntimeError):
    """Raised when Webots is requested but cannot be started."""


def find_webots() -> str | None:
    """Absolute path to the Webots executable, or None if it isn't installed."""
    home = os.environ.get("WEBOTS_HOME")
    if home:
        candidate = Path(home) / "webots"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    found = shutil.which("webots")
    if found:
        return found
    for path in FALLBACK_BINARIES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# Process names an actual running Webots can have. ``webots-bin`` is the one
# that matters in practice: /snap/bin/webots is a symlink to /usr/bin/snap,
# which execs .../usr/share/webots/bin/webots-bin -- so matching only "webots"
# finds nothing on a snap install and cheerfully opens a second simulator.
WEBOTS_PROCESS_NAMES = ("webots", "webots-bin")


def is_webots_running() -> bool:
    """True if some Webots process is already up.

    Matches on exact process NAME (``pgrep -x``) rather than the full command
    line: ``pgrep -f webots`` would match this pipeline's own
    ``run.py --launch-webots`` argv and report success every single time.

    A process check rather than a port check, because the controller binds UDP
    8765 only once the simulation is playing -- a freshly-opened-but-paused
    Webots would look absent and get launched a second time.
    """
    for name in WEBOTS_PROCESS_NAMES:
        try:
            result = subprocess.run(
                ["pgrep", "-x", name], capture_output=True, check=False, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0:
            continue
        for pid in result.stdout.decode("utf-8", "replace").split():
            if _is_live(pid):
                return True
    return False


def _is_live(pid: str) -> bool:
    """True unless ``pid`` is a zombie.

    ``pgrep`` happily reports processes that have exited but not yet been reaped.
    A Webots that was killed while its launcher was still sleeping leaves exactly
    such a ``<defunct>`` entry, and treating it as "already running" makes this
    module refuse to start Webots ever again -- it attaches to a corpse, reports
    success, and the caller then streams pose frames at nothing.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            # Field 3 is the state code, but comm (field 2) may contain spaces
            # and brackets, so split after the closing parenthesis.
            state = fh.read().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        # No procfs (macOS) or the process vanished mid-check: assume it is real
        # rather than launching a second simulator on a maybe.
        return True
    return state != "Z"


class WebotsProcess:
    """A Webots instance started by us, stopped again on exit.

    Instances we merely attached to are never terminated -- closing a simulation
    the user had already set up would be a nasty surprise.
    """

    def __init__(self, proc: subprocess.Popen | None) -> None:
        self._proc = proc

    @property
    def owned(self) -> bool:
        return self._proc is not None

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info("Closing Webots (pid %d) ...", self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Webots did not exit in %.0fs; killing it.", timeout)
            self._proc.kill()


def launch(
    world: Path | str | None = None,
    *,
    minimize: bool = False,
    startup_delay_s: float = 4.0,
    mode: str = "realtime",
) -> WebotsProcess:
    """Start Webots on ``world`` (or attach to a running instance).

    ``startup_delay_s`` gives the simulation time to load the world and bind the
    controller's socket before the pipeline starts streaming; UDP is
    connectionless, so frames sent earlier are silently dropped rather than
    queued, and the robot would simply sit still for the first few seconds.

    ``mode`` is passed to Webots as ``--mode``. It defaults to ``realtime``
    rather than to Webots' own default because Webots restores whatever run
    state the world was last saved in -- and a world saved while paused opens
    paused, so the controller blocks in its first ``robot.step()``, never binds
    its socket, and the whole demo looks like a networking failure.
    """
    if is_webots_running():
        logger.info("Webots is already running; attaching to it.")
        return WebotsProcess(None)

    binary = find_webots()
    if binary is None:
        raise WebotsLaunchError(
            "Webots not found. Install it (R2023b or newer -- see docs/"
            "RUN_INSTRUCTIONS.md), put it on $PATH or set $WEBOTS_HOME, or run "
            "with --no-launch-webots and open the world yourself."
        )

    world_path = Path(world) if world is not None else DEFAULT_WORLD
    if not world_path.is_file():
        raise WebotsLaunchError(f"Webots world not found: {world_path}")

    argv = [binary]
    if minimize:
        argv.append("--minimize")
    if mode:
        argv.append(f"--mode={mode}")
    argv.append(str(world_path))

    logger.info("Launching Webots: %s", " ".join(argv))
    try:
        proc = subprocess.Popen(argv)
    except OSError as exc:
        raise WebotsLaunchError(f"Could not start Webots ({binary}): {exc}") from exc

    # Fail fast on an immediate crash (bad world file, no display) instead of
    # streaming into the void for the rest of the run.
    time.sleep(min(startup_delay_s, 1.0))
    if proc.poll() is not None:
        raise WebotsLaunchError(
            f"Webots exited immediately (code {proc.returncode}). Check that a "
            f"display is available and that {world_path.name} loads."
        )
    remaining = startup_delay_s - 1.0
    if remaining > 0:
        logger.info("Waiting %.1fs for the world to load ...", remaining)
        time.sleep(remaining)
    return WebotsProcess(proc)
