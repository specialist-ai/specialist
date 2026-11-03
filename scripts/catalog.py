"""Catalog a corpus's state layer as a Rerun dataset.

The catalog serves corpus inspection and training, and a training dataset
holds the state layer alone: ``state_<cfg>/<session>.rrd`` registers as the
segments themselves, with the state dashboard as the default and both hands'
frames side by side as the segment-table card. The gripper models under
``<root>/assets`` register as the dataset's assets, which every segment
shows. Every other layer is opened per session, straight from its files:

    uv run rerun <root>/blueprints/specialist.rbl <root>/base/<id>.rrd \
        <root>/estimate/<id>.rrd <root>/viz/<id>.rrd \
        --asset <root>/assets/gripper-<urdf-sha>-<slot>.rrd

Usage:
    uv run python scripts/catalog.py corpus/state_s12_r100 \
        --catalog-url rerun+http://host:port
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from specialist_corpus.read import Recording
from specialist_corpus.schema import (
  APPLICATION_ID,
  HANDS,
  POSE_TRANSLATION,
  STATE,
  state_image,
)

from specialist.record.asset import register_assets
from specialist.vi_ba.blueprint import state_blueprint, state_card_blueprint


@dataclass
class Args:
  """Catalog registration of one state configuration."""

  # One configuration's directory, e.g. corpus/state_s12_r100. The directory
  # name is the configuration, and the blueprints land beside it.
  state: tyro.conf.Positional[Path]
  # The catalog server, e.g. rerun+http://host:port.
  catalog_url: str
  # Dataset to register into, created if absent; the corpus directory's name
  # unless given.
  dataset: str | None = None


def state_files(state: Path) -> list[Path]:
  """One configuration's files. The state directories write one set of names,
  so a dataset takes exactly one directory."""
  if state.name != STATE and not state.name.startswith(f"{STATE}_"):
    raise SystemExit(f"{state} is not a state layer directory")
  files = sorted(state.glob("*.rrd"))
  if not files:
    present = sorted(
      d.name
      for d in state.parent.iterdir()
      if d.is_dir()
      and (d.name == STATE or d.name.startswith(f"{STATE}_"))
      and any(d.glob("*.rrd"))
    )
    raise SystemExit(f"{state} holds no .rrd; state layers present: {present}")
  return files


def frame_width(rrd: Path) -> int:
  """The width of the state layer's images: the offset the right
  hand's frames are placed at."""
  placed = Recording(rrd).row(state_image(HANDS[1])).get(POSE_TRANSLATION)
  if placed is None:
    raise SystemExit(f"{rrd} places no {state_image(HANDS[1])} frames")
  return int(placed[0][0])


def check_one_configuration(dataset: DatasetEntry, state: str) -> None:
  """Refuse a dataset holding any layer but ``state``: each configuration
  registers as the layer of its name and writes the same names, so two would
  resolve silently."""
  held = {layer.name for layer in dataset.info().layers}
  if held - {state}:
    raise SystemExit(
      f"{sorted(held)} is already on the dataset; a dataset holds one state "
      "configuration and no other layer"
    )


def register_all(dataset: DatasetEntry, files: list[Path], state: str) -> None:
  """Register every file as a segment; a state layer is regenerated freely,
  so a duplicate always REPLACEs."""
  by_uri = {p.resolve().as_uri(): p for p in files}
  handle = dataset.register(
    list(by_uri),
    layer_name=[state] * len(by_uri),
    on_duplicate=OnDuplicateSegmentLayer.REPLACE,
  )
  failures: Counter[str] = Counter()
  for result in handle.iter_results(timeout_secs=1800):
    if result.is_error:
      failures[result.uri] += 1
      print(f"FAILED {by_uri[result.uri].name}: {result.error}")
  print(
    f"registered {len(files) - sum(failures.values())} of {len(files)} {state} recording(s)"
  )
  if failures:
    raise SystemExit(
      f"{sum(failures.values())} registration(s) failed; no blueprint installed"
    )


def main(args: Args) -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  files = state_files(args.state)
  name = args.dataset or args.state.resolve().parent.name
  client = CatalogClient(args.catalog_url)
  if name in client.dataset_names():
    dataset = client.get_dataset(name)
  else:
    dataset = client.create_dataset(name)
  check_one_configuration(dataset, args.state.name)
  register_all(dataset, files, args.state.name)
  register_assets(dataset, args.state.parent)

  dashboard = state_blueprint()
  blueprints_dir = args.state.parent / "blueprints"
  blueprints_dir.mkdir(exist_ok=True)
  dashboard.save(APPLICATION_ID, str(blueprints_dir / "state.rbl"))
  print(f"wrote {(blueprints_dir / 'state.rbl').resolve()}")
  # The catalog rewrites every segment's application id to the dataset id, and
  # a default blueprint applies only where the application id matches.
  with tempfile.TemporaryDirectory() as tmp:
    for kind, blueprint, table in (
      ("dashboard", dashboard, False),
      # The card holds the two frames, which is what tells one session from
      # another.
      ("card", state_card_blueprint(frame_width(files[0])), True),
    ):
      rbl = Path(tmp) / f"{name}-{kind}.rbl"
      blueprint.save(str(dataset.id), str(rbl))
      dataset.register_blueprint(
        rbl.resolve().as_uri(), set_default=True, segment_table=table
      )
  print(f"installed the default + segment-table blueprints on {name!r}")


if __name__ == "__main__":
  main(tyro.cli(Args))
