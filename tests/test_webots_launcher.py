"""Tests for the Webots launcher (src/webots_launcher.py).

Nothing here actually starts Webots: the behaviour worth protecting is the
decision-making around it -- attach vs. launch, and failing with an actionable
message instead of streaming pose frames at a simulator that was never started.
"""
from __future__ import annotations

import subprocess

import pytest

from src import webots_launcher as wl


class _FakeProc:
    def __init__(self, poll_result=None) -> None:
        self.pid = 4242
        self._poll = poll_result
        self.terminated = False
        self.killed = False
        self.returncode = poll_result

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0

    def kill(self):
        self.killed = True
        self._poll = -9

    def wait(self, timeout=None):
        return 0


def test_the_project_world_exists_where_the_launcher_expects_it() -> None:
    """The default world path is baked in; a rename must not go unnoticed."""
    assert wl.DEFAULT_WORLD.is_file(), f"missing world: {wl.DEFAULT_WORLD}"


def test_an_already_running_webots_is_attached_to_not_duplicated(monkeypatch) -> None:
    """Two instances would both load the world and fight over UDP 8765."""
    monkeypatch.setattr(wl, "is_webots_running", lambda: True)
    monkeypatch.setattr(wl, "find_webots", lambda: pytest.fail("should not launch"))
    handle = wl.launch()
    assert handle.owned is False
    handle.stop()   # must be a no-op -- we did not start it


def test_a_missing_webots_fails_with_an_actionable_message(monkeypatch) -> None:
    monkeypatch.setattr(wl, "is_webots_running", lambda: False)
    monkeypatch.setattr(wl, "find_webots", lambda: None)
    with pytest.raises(wl.WebotsLaunchError, match="WEBOTS_HOME|not found"):
        wl.launch()


def test_a_missing_world_file_is_reported_before_launching(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(wl, "is_webots_running", lambda: False)
    monkeypatch.setattr(wl, "find_webots", lambda: "/usr/bin/true")
    with pytest.raises(wl.WebotsLaunchError, match="world not found"):
        wl.launch(world=tmp_path / "nope.wbt")


def test_an_immediate_crash_is_reported_not_ignored(monkeypatch) -> None:
    """Otherwise the pipeline streams into the void for the whole run."""
    monkeypatch.setattr(wl, "is_webots_running", lambda: False)
    monkeypatch.setattr(wl, "find_webots", lambda: "/usr/bin/true")
    monkeypatch.setattr(wl.subprocess, "Popen", lambda argv: _FakeProc(poll_result=1))
    monkeypatch.setattr(wl.time, "sleep", lambda _s: None)
    with pytest.raises(wl.WebotsLaunchError, match="exited immediately"):
        wl.launch()


def test_a_launched_instance_is_owned_and_stopped(monkeypatch) -> None:
    proc = _FakeProc(poll_result=None)
    monkeypatch.setattr(wl, "is_webots_running", lambda: False)
    monkeypatch.setattr(wl, "find_webots", lambda: "/usr/bin/true")
    monkeypatch.setattr(wl.subprocess, "Popen", lambda argv: proc)
    monkeypatch.setattr(wl.time, "sleep", lambda _s: None)
    handle = wl.launch()
    assert handle.owned is True
    handle.stop()
    assert proc.terminated and not proc.killed


def test_a_hung_instance_is_killed(monkeypatch) -> None:
    proc = _FakeProc(poll_result=None)

    def _hang(timeout=None):
        raise subprocess.TimeoutExpired(cmd="webots", timeout=timeout or 0)

    proc.wait = _hang
    proc.terminate = lambda: None      # refuses to go down politely
    handle = wl.WebotsProcess(proc)
    handle.stop(timeout=0.01)
    assert proc.killed


def test_webots_home_is_preferred_over_path(monkeypatch, tmp_path) -> None:
    binary = tmp_path / "webots"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("WEBOTS_HOME", str(tmp_path))
    monkeypatch.setattr(wl.shutil, "which", lambda _n: "/somewhere/else/webots")
    assert wl.find_webots() == str(binary)


def test_a_snap_webots_is_detected(monkeypatch) -> None:
    """/snap/bin/webots is a symlink to snap itself; the process that actually
    runs is `webots-bin`, so matching only "webots" misses it entirely and a
    second simulator gets opened on top of the user's."""
    seen = []

    class _Result:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = b"38861\n" if rc == 0 else b""

    def fake_run(argv, **kwargs):
        seen.append(argv[-1])
        return _Result(0 if argv[-1] == "webots-bin" else 1)

    monkeypatch.setattr(wl.subprocess, "run", fake_run)
    monkeypatch.setattr(wl, "_is_live", lambda pid: True)
    assert wl.is_webots_running() is True
    assert "webots-bin" in seen


def test_the_running_check_never_matches_our_own_command_line(monkeypatch) -> None:
    """`pgrep -f webots` would match `run.py --launch-webots` and always say
    yes; the check must be on the process NAME."""
    used = []

    class _Result:
        returncode = 1
        stdout = b""

    def fake_run(argv, **kwargs):
        used.append(argv)
        return _Result()

    monkeypatch.setattr(wl.subprocess, "run", fake_run)
    assert wl.is_webots_running() is False
    assert all("-x" in argv and "-f" not in argv for argv in used)


def test_a_zombie_webots_does_not_count_as_running(monkeypatch) -> None:
    """pgrep reports un-reaped processes. A Webots killed while its launcher was
    still alive leaves a <defunct> entry, and treating that as "already running"
    means the launcher attaches to a corpse and the caller streams pose frames at
    nothing -- forever, because the zombie never goes away on its own."""
    class _Result:
        returncode = 0
        stdout = b"36537\n"

    monkeypatch.setattr(wl.subprocess, "run", lambda argv, **kw: _Result())
    monkeypatch.setattr(wl, "_is_live", lambda pid: False)
    assert wl.is_webots_running() is False


def test_a_live_process_does_count_as_running(monkeypatch) -> None:
    class _Result:
        returncode = 0
        stdout = b"4242\n"

    monkeypatch.setattr(wl.subprocess, "run", lambda argv, **kw: _Result())
    monkeypatch.setattr(wl, "_is_live", lambda pid: True)
    assert wl.is_webots_running() is True


def test_is_live_reads_the_state_field_past_a_bracketed_comm(tmp_path) -> None:
    """/proc/<pid>/stat's comm field is bracketed and may contain spaces, so the
    state must be located from the LAST ')' -- splitting on whitespace picks up
    the wrong field for a process whose name contains one."""
    assert wl._is_live("1") in (True, False)          # real pid 1, must not raise
    assert wl._is_live("2147483646") is True          # absent -> assume real


def test_is_live_parses_a_zombie_and_a_sleeper(monkeypatch, tmp_path) -> None:
    import builtins
    real_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path).endswith("/stat"):
            pid = str(path).split("/")[2]
            state = "Z" if pid == "111" else "S"
            import io
            return io.StringIO(f"{pid} (webots bin) {state} 1 1 0 0 -1 0 0")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    assert wl._is_live("111") is False
    assert wl._is_live("222") is True
