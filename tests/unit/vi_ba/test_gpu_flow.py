"""Tests for ``vi_ba.gpu_flow``'s stream bookkeeping. The module imports without
CUDA, so the reuse decision is testable on any host."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from specialist.vi_ba import gpu_flow


@pytest.fixture(autouse=True)
def _clean_streams() -> Iterator[None]:
  for state in (gpu_flow._claimed, gpu_flow._bad, gpu_flow._engines):
    state.clear()
  gpu_flow._stream.__dict__.clear()
  yield
  for state in (gpu_flow._claimed, gpu_flow._bad, gpu_flow._engines):
    state.clear()
  gpu_flow._stream.__dict__.clear()


def _frames(h: int = 64, w: int = 64) -> list[np.ndarray]:
  return [np.zeros((h, w), dtype=np.uint8) for _ in range(2)]


def _pretend_warm(engine: Any, shape: tuple[int, int], fwd: int, bwd: int) -> None:
  gpu_flow._engines.append(engine)
  gpu_flow._stream.engine = engine
  gpu_flow._stream.key = (shape, fwd, bwd)


def test_a_captures_second_lens_reuses_the_stream_its_first_lens_warmed() -> None:
  """Each stream holds gigabytes of page-locked host staging, so a capture's
  two lenses share one."""
  engine = object()
  _pretend_warm(engine, (64, 64), 10, 2)
  assert gpu_flow.prime(_frames(), 10, 2) is engine


def _refuse_to_build(monkeypatch: pytest.MonkeyPatch) -> list[int]:
  """Make warming a new stream observable without a GPU: the backend import is
  the first thing a fresh stream needs."""
  attempts: list[int] = []

  def _no_backend() -> tuple[Any, Any]:
    attempts.append(1)
    raise RuntimeError("no backend in this test")

  monkeypatch.setattr(gpu_flow, "_import_backend", _no_backend)
  return attempts


def test_a_different_frame_shape_does_not_reuse_the_warm_stream(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The staging buffers are cut to one shape, so a stream warmed for another
  cannot serve this lens."""
  attempts = _refuse_to_build(monkeypatch)
  _pretend_warm(object(), (64, 64), 10, 2)
  with pytest.raises(RuntimeError, match="no backend"):
    gpu_flow.prime(_frames(128, 128), 10, 2)
  assert attempts, "the warm stream was reused for a different shape"


def test_a_stream_already_shut_down_is_not_handed_out_again(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``shutdown`` empties the registry, so a thread still holding a reference
  warms a new stream on its next call."""
  attempts = _refuse_to_build(monkeypatch)
  _pretend_warm(object(), (64, 64), 10, 2)
  gpu_flow._engines.clear()
  with pytest.raises(RuntimeError, match="no backend"):
    gpu_flow.prime(_frames(), 10, 2)
  assert attempts, "a stream that was shut down was handed out again"


def _fake_torch(free_gb: float, total_gb: float) -> Any:
  class _Cuda:
    @staticmethod
    def is_available() -> bool:
      return True

    @staticmethod
    def device_count() -> int:
      return 1

    @staticmethod
    def mem_get_info(i: int) -> tuple[int, int]:
      return int(free_gb * 2**30), int(total_gb * 2**30)

  class _Torch:
    cuda = _Cuda()

  return _Torch()


def test_claimed_batch_is_sized_from_total_memory(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A busy card still gets the batch its hardware affords: the size comes
  from total memory, not the free snapshot."""
  fake = _fake_torch(free_gb=9.0, total_gb=24.0)
  monkeypatch.setattr(gpu_flow, "_import_backend", lambda: (fake, None))
  idx, batch = gpu_flow._claim_device()
  assert (idx, batch) == (
    0,
    int((24.0 - gpu_flow._RESERVE_GB) / gpu_flow._GB_PER_PAIR),
  )


def test_claim_requires_the_minimum_free_memory(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A nearly-full card is not claimable even though its total could host a
  batch."""
  fake = _fake_torch(free_gb=gpu_flow._MIN_FREE_GB - 0.5, total_gb=24.0)
  monkeypatch.setattr(gpu_flow, "_import_backend", lambda: (fake, None))
  with pytest.raises(RuntimeError, match="enough free memory"):
    gpu_flow._claim_device()


def test_device_batch_floors_at_zero_below_the_reserve() -> None:
  """A card smaller than the reserve hosts nothing."""
  assert gpu_flow._device_batch(total_gb=gpu_flow._RESERVE_GB) == 0
  assert gpu_flow._device_batch(total_gb=0.0) == 0


def test_device_batch_caps_at_the_maximum() -> None:
  """The per-stream batch is bounded regardless of card size."""
  assert gpu_flow._device_batch(total_gb=1024.0) == gpu_flow._MAX_BATCH
