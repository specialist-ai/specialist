"""Reading a layer back: statics, lenses, grippers and anchors."""

from pathlib import Path

import pytest
import rerun as rr
from rerun.experimental import Chunk, send_chunks
from specialist_corpus import schema
from specialist_corpus.read import Recording, lens_from_row


def _lens_chunk() -> Chunk:
  """One stored lens entity: intrinsics in a 2880-wide stitched frame."""
  return Chunk.from_columns(
    schema.lens("cam00", "front"),
    indexes=[],
    columns=[
      *rr.Transform3D.columns(
        translation=[[0.0, 0.0, 0.0]], quaternion=[[0.0, 0.0, 0.0, 1.0]]
      ),
      *rr.AnyValues.columns(
        fx=[1000.0], fy=[1000.0], cx=[720.0], cy=[720.0],
        k1=[0.0], k2=[0.0], k3=[0.0], p1=[0.0], p2=[0.0],
        xi=[1.9], frame_width=[2880.0],
      ),
    ],
  )  # fmt: skip


def _write(rrd: Path, chunks: list[Chunk]) -> None:
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks(chunks, recording=stream)


def test_a_static_written_twice_resolves_to_one_value(tmp_path: Path) -> None:
  def calibration(fx: float) -> Chunk:
    return Chunk.from_columns(
      schema.lens("cam00", "front"),
      indexes=[],
      columns=rr.AnyValues.columns(fx=[fx]),
    )

  older = calibration(1.0)
  newer = calibration(2.0)
  rrd = tmp_path / "base.rrd"
  _write(rrd, [newer, older])
  row = Recording(rrd).row(schema.lens("cam00", "front"))
  assert list(row["fx"]) == [1.0]


def test_a_lens_reads_back_at_the_decode_resolution(tmp_path: Path) -> None:
  rrd = tmp_path / "base.rrd"
  with rr.RecordingStream("test", recording_id="s") as stream:
    stream.save(str(rrd))
    send_chunks([_lens_chunk()], recording=stream)
  row = Recording(rrd).row(schema.lens("cam00", "front"))
  lens = lens_from_row(row, 1440, "front")
  assert lens.params.fx == pytest.approx(1000.0)
  assert lens.body_from_lens.shape == (4, 4)
