"""Register a corpus of session records on a Rerun catalog dataset.

The corpus root is laid out layer-first — ``base/<session>.rrd`` plus one
directory per additional layer (``track/``, ``mocap/``, …) — so the root alone
says everything: every layer directory registers under its own name. A layer
whose recording id matches no base segment is refused (it would never attach).
The standard run blueprint installs as the dataset default so every session
opens with the same dashboard.

Usage:
    uv run python scripts/register.py --catalog-url rerun+http://host:port \
        --dataset gripper_corpus --root corpus
"""

from __future__ import annotations

import functools
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.experimental import RrdReader

from specialist.record.read import check_frame_alignment
from specialist.record.session import SessionRecording
from specialist.vi_ba.viz import blueprint_for_names, segment_card_blueprint


@dataclass
class Args:
  """Catalog registration of a layer-first corpus root."""

  # The catalog server, e.g. rerun+http://host:port.
  catalog_url: str
  # Dataset to register into; created if it does not exist.
  dataset: str
  # Corpus root: base/<session>.rrd plus one directory per additional layer.
  root: Path
  # Session ids of re-converted bases; their segments register with REPLACE so
  # the byte-range index is rebuilt over the new file.
  replace_base: tuple[str, ...] = ()


def recording_id(rrd: Path) -> str:
  """The single recording id carried by one record file."""
  entries = RrdReader(str(rrd)).recordings()
  if len(entries) != 1:
    raise ValueError(f"{rrd} holds {len(entries)} recordings; expected exactly 1")
  return str(entries[0].recording_id)


def corpus_layers(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
  """Discover ``(base_files, {layer_name: files})`` from a layer-first root.
  Every directory holding rrds is a layer named after itself; ``base`` defines
  the segments the others attach to."""
  bases = sorted((root / "base").glob("*.rrd"))
  if not bases:
    raise SystemExit(f"{root / 'base'} holds no .rrd; a corpus needs its base layer")
  layers = {
    d.name: files
    for d in sorted(root.iterdir())
    if d.is_dir() and d.name != "base" and (files := sorted(d.glob("*.rrd")))
  }
  return bases, layers


def check_layers_attach(base_ids: set[str], layers: dict[Path, str]) -> None:
  """Refuse layers whose recording id matches no base segment."""
  orphans = {p: rid for p, rid in layers.items() if rid not in base_ids}
  if orphans:
    lines = ", ".join(f"{p.name} ({rid})" for p, rid in orphans.items())
    raise ValueError(f"layers match no base segment: {lines}")


_recording = functools.cache(SessionRecording)


def register_batch(
  dataset: DatasetEntry,
  layer_of: dict[Path, str],
  on_duplicate: OnDuplicateSegmentLayer,
  failures: list[str],
) -> tuple[set[str], Counter[str]]:
  """Register one batch, drain its results, and return the accepted segment ids
  alongside a per-layer failure count."""
  if not layer_of:
    return set(), Counter()
  layer_by_uri = {p.resolve().as_uri(): layer for p, layer in layer_of.items()}
  handle = dataset.register(
    list(layer_by_uri),
    layer_name=list(layer_by_uri.values()),
    on_duplicate=on_duplicate,
  )
  accepted: set[str] = set()
  failed: Counter[str] = Counter()
  for result in handle.iter_results(timeout_secs=1800):
    if result.is_error:
      layer = layer_by_uri[result.uri]
      failed[layer] += 1
      failures.append(f"{layer}: {result.uri}: {result.error}")
      print(f"FAILED {layer}: {result.uri}: {result.error}")
    else:
      accepted.add(str(result.segment_id))
  return accepted, failed


def main(args: Args) -> None:
  base_files, layer_files = corpus_layers(args.root)
  bases = {p: recording_id(p) for p in base_files}

  client = CatalogClient(args.catalog_url)
  if args.dataset in client.dataset_names():
    dataset = client.get_dataset(args.dataset)
    known = {str(s) for s in dataset.segment_ids()}
  else:
    dataset = client.create_dataset(args.dataset)
    known = set()
  layers = {p: recording_id(p) for files in layer_files.values() for p in files}
  # Only the track layer shares the base's frame clock; a GT layer samples on
  # its own instrument's clock and has nothing to align.
  base_by_id = {rid: p for p, rid in bases.items()}
  for path in layer_files.get("track", []):
    base = base_by_id.get(layers[path])
    if base is None:
      print(
        f"WARNING: {path.name} attaches to already-registered base "
        f"{layers[path]}; frame alignment NOT checked (restore that base "
        "under the root to verify)"
      )
      continue
    check_frame_alignment(_recording(base).frame_times, path)

  unknown = set(args.replace_base) - set(bases.values())
  if unknown:
    raise SystemExit(
      f"--replace-base names no base under {args.root}: {sorted(unknown)}"
    )

  # Bases are immutable, so duplicates SKIP; REPLACE rebuilds the catalog's
  # byte-range index.
  failures: list[str] = []
  replace = set(args.replace_base)
  keep = {p: "base" for p, rid in bases.items() if rid not in replace}
  redo = {p: "base" for p, rid in bases.items() if rid in replace}
  accepted, _ = register_batch(dataset, keep, OnDuplicateSegmentLayer.SKIP, failures)
  redone, _ = register_batch(dataset, redo, OnDuplicateSegmentLayer.REPLACE, failures)
  accepted |= redone
  print(f"registered {len(bases) - len(failures)} of {len(bases)} base recording(s)")
  check_layers_attach(known | accepted, layers)
  # `layer_name` takes one name per URI.
  layer_of = {p: name for name, files in layer_files.items() for p in files}
  _, failed = register_batch(
    dataset, layer_of, OnDuplicateSegmentLayer.REPLACE, failures
  )
  for name, files in layer_files.items():
    print(f"registered {len(files) - failed[name]} {name} layer(s)")

  names = sorted({n for p in bases for n in _recording(p).tracker_names})
  # The catalog rewrites every segment's application id to the dataset id, and
  # a default blueprint applies only where the application id matches.
  blueprint_app_id = str(dataset.id)
  with tempfile.TemporaryDirectory() as tmp:
    rbl = Path(tmp) / f"{args.dataset}.rbl"
    blueprint_for_names(names).save(blueprint_app_id, str(rbl))
    dataset.register_blueprint(rbl.resolve().as_uri(), set_default=True)
    # The segment table renders every session at once, so its blueprint never
    # decodes the video streams.
    cards = Path(tmp) / f"{args.dataset}-cards.rbl"
    segment_card_blueprint(names).save(blueprint_app_id, str(cards))
    dataset.register_blueprint(
      cards.resolve().as_uri(), set_default=True, segment_table=True
    )
  print(f"installed the default + segment-table blueprints on {args.dataset!r}")
  if failures:
    raise SystemExit(f"{len(failures)} registration(s) failed")


if __name__ == "__main__":
  main(tyro.cli(Args))
