"""Tests for MeTRAbs model resolution (src/perception/metrabs_model.py).

These cover the path resolution only -- no download, no TensorFlow, no GPU --
because that is where the pipeline actually broke: the configured
``DEFAULT_MODEL_URL`` could not be loaded at all, since MeTRAbs' release zips
nest the SavedModel one directory deep and ``tfhub.load`` only looks at the
extraction root.
"""
from __future__ import annotations

import pytest

from src.perception.metrabs_model import (
    MetrabsUnavailableError,
    _saved_model_root,
    resolve_model_path,
)


def _make_saved_model(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "saved_model.pb").write_bytes(b"")
    (directory / "variables").mkdir(exist_ok=True)
    return directory


def test_a_flat_savedmodel_resolves_to_itself(tmp_path) -> None:
    root = _make_saved_model(tmp_path / "model")
    assert _saved_model_root(root) == root


def test_a_nested_savedmodel_is_descended_into(tmp_path) -> None:
    """The actual MeTRAbs layout: metrabs_eff2s_y4/metrabs_eff2s_y4/..."""
    outer = tmp_path / "metrabs_eff2s_y4"
    inner = _make_saved_model(outer / "metrabs_eff2s_y4")
    assert _saved_model_root(outer) == inner


def test_several_levels_of_nesting_still_resolve(tmp_path) -> None:
    inner = _make_saved_model(tmp_path / "a" / "b" / "c")
    assert _saved_model_root(tmp_path / "a") == inner


def test_a_directory_with_no_savedmodel_says_so(tmp_path) -> None:
    """A truncated download must fail with an actionable message rather than
    TF-Hub's 'does not appear to be a valid module'."""
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    with pytest.raises(MetrabsUnavailableError, match="No saved_model.pb"):
        _saved_model_root(tmp_path)


def test_a_local_directory_is_used_as_is(tmp_path) -> None:
    root = _make_saved_model(tmp_path / "model")
    assert resolve_model_path(str(root)) == str(root)


def test_an_already_cached_url_is_not_re_downloaded(tmp_path) -> None:
    """Resolution must be offline once the cache is populated -- this test
    would hang for a 320 MB download if it were not."""
    cached = _make_saved_model(tmp_path / "metrabs_eff2s_y4")
    url = "https://example.invalid/data/metrabs/metrabs_eff2s_y4.zip"
    assert resolve_model_path(url, cache_dir=tmp_path) == str(cached)


def test_a_non_zip_handle_is_passed_through_to_tfhub() -> None:
    handle = "https://tfhub.dev/some/model/1"
    assert resolve_model_path(handle) == handle


# --------------------------------------------------------------- GPU placement
# "TensorFlow can SEE a GPU" and "TensorFlow will RUN ON one" are different
# claims. The second is the one that matters: a CUDA/cuDNN mismatch leaves the
# card enumerated but silently places every op on CPU, which is the "why is it
# so slow" failure this guard exists to catch. No TensorFlow is imported here --
# a fake stands in, because the behaviour under test is the decision, not TF.
class _FakeDevice:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTensor:
    def __init__(self, device: str) -> None:
        self.device = device


class _FakeConfigExperimental:
    def __init__(self, owner) -> None:
        self._owner = owner

    def set_memory_growth(self, gpu, enabled) -> None:
        self._owner.growth_set.append((gpu.name, enabled))


class _FakeConfig:
    def __init__(self, owner) -> None:
        self._owner = owner
        self.experimental = _FakeConfigExperimental(owner)

    def list_physical_devices(self, kind):
        return self._owner.gpus if kind == "GPU" else []


class _FakeTF:
    def __init__(self, gpus, placed_on="/job:localhost/replica:0/task:0/device:GPU:0",
                 raise_on_op=None) -> None:
        self.gpus = [_FakeDevice(n) for n in gpus]
        self.growth_set: list = []
        self._placed_on = placed_on
        self._raise_on_op = raise_on_op
        self.config = _FakeConfig(self)
        self.linalg = self

    def device(self, _name):
        class _Ctx:
            def __enter__(self_inner): return None
            def __exit__(self_inner, *a): return False
        return _Ctx()

    def ones(self, _shape):
        return object()

    def matmul(self, _a, _b):
        if self._raise_on_op is not None:
            raise self._raise_on_op
        return _FakeTensor(self._placed_on)


def _patch_tf(monkeypatch, fake):
    import src.perception.metrabs_model as mm
    monkeypatch.setattr(mm, "require_tensorflow", lambda: fake)
    return mm


def test_no_visible_gpu_is_refused(monkeypatch) -> None:
    mm = _patch_tf(monkeypatch, _FakeTF(gpus=[]))
    with pytest.raises(MetrabsUnavailableError, match="No GPU visible"):
        mm.require_gpu()


def test_a_working_gpu_is_accepted_and_memory_growth_is_requested(monkeypatch) -> None:
    """Growth matters: TF's default reserves the whole card (measured 22.6 GB of
    the 3090 Ti), starving every other process on the machine."""
    fake = _FakeTF(gpus=["/physical_device:GPU:0"])
    mm = _patch_tf(monkeypatch, fake)
    mm.require_gpu()
    assert fake.growth_set == [("/physical_device:GPU:0", True)]


def test_an_op_placed_on_cpu_is_refused(monkeypatch) -> None:
    """The silent-CPU case: the card is enumerated, but ops land on CPU."""
    fake = _FakeTF(
        gpus=["/physical_device:GPU:0"],
        placed_on="/job:localhost/replica:0/task:0/device:CPU:0",
    )
    mm = _patch_tf(monkeypatch, fake)
    with pytest.raises(MetrabsUnavailableError, match="placed a test op"):
        mm.require_gpu()


def test_an_op_that_throws_is_refused(monkeypatch) -> None:
    """The usual cause is a CUDA/cuDNN build that disagrees with the driver."""
    fake = _FakeTF(
        gpus=["/physical_device:GPU:0"],
        raise_on_op=RuntimeError("cuDNN failed to initialize"),
    )
    mm = _patch_tf(monkeypatch, fake)
    with pytest.raises(MetrabsUnavailableError, match="could not run an op"):
        mm.require_gpu()


def test_memory_growth_survives_an_already_initialised_device(monkeypatch) -> None:
    """Growth can only be set before first use; afterwards TF raises and the
    allocator is already fixed. That is not a reason to refuse to run."""
    fake = _FakeTF(gpus=["/physical_device:GPU:0"])

    def _boom(gpu, enabled):
        raise RuntimeError("Physical devices cannot be modified after being initialized")

    fake.config.experimental.set_memory_growth = _boom
    mm = _patch_tf(monkeypatch, fake)
    mm.require_gpu()          # must not raise
