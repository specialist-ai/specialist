"""Register a corpus of session records on a Rerun catalog dataset.

The corpus root is laid out layer-first — ``base/<session>.rrd`` plus one
directory per additional layer (``estimate/``, ``viz/``, ``mocap/``, …) — and
every layer directory registers under its own name. A layer whose recording id
matches no base segment is refused, and so is one that writes a name another
layer already owns. The standard run blueprint installs as the dataset default;
the same layout is saved to ``<root>/blueprints/specialist.rbl`` for files
opened directly. ``--state`` builds a training dataset instead: the named
state layer as the segments themselves and nothing else, with the state
dashboard as the default and both hands' frames side by side as its
segment-table card.

Usage:
    uv run python scripts/register.py --catalog-url rerun+http://host:port \
        --dataset gripper_corpus --root corpus
"""

from __future__ import annotations

import functools
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.experimental import RrdReader
from specialist_corpus.read import Recording, check_frame_alignment
from specialist_corpus.schema import (
  APPLICATION_ID,
  BASE,
  ESTIMATE,
  ESTIMATE_PROPERTIES,
  HANDS,
  POSE_TRANSLATION,
  STATE,
  anchor_id_of,
  anchor_ids_in,
  layer_dir,
  layer_rrd,
  slots_in,
  state_image,
)

from specialist.record.session import SessionRecording
from specialist.vi_ba.blueprint import (
  blueprint_for_slots,
  segment_card_blueprint,
  state_blueprint,
  state_card_blueprint,
)


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
  # Registers this state layer, e.g. state_s12_i3, as the segments themselves
  # and nothing else. A corpus holds one directory per configuration and a
  # dataset takes one of them.
  state: str | None = None


def recording_id(rrd: Path) -> str:
  """The single recording id carried by one record file."""
  entries = RrdReader(str(rrd)).recordings()
  if len(entries) != 1:
    raise ValueError(f"{rrd} holds {len(entries)} recordings; expected exactly 1")
  return str(entries[0].recording_id)


def anchor_tag_sizes(rrds: list[Path]) -> set[float]:
  """The tag width the anchors were solved against, as the estimate layers
  record it."""
  sizes = set()
  for rrd in rrds:
    for chunk in RrdReader(str(rrd)).stream():
      if anchor_id_of(str(chunk.entity_path)) is None:
        continue
      batch = chunk.to_record_batch()
      if "tag_size" in batch.schema.names:
        sizes |= {v for row in batch.column("tag_size").to_pylist() if row for v in row}
  return sizes


def corpus_layers(
  root: Path, state: str | None = None
) -> tuple[list[Path], dict[str, list[Path]]]:
  """Discover ``(base_files, {layer_name: files})`` from a layer-first root.
  Every directory holding rrds is a layer named after itself; ``base`` defines
  the segments the others attach to; ``state`` selects a state directory for a
  training dataset instead. The state directories write one set of names, so
  at most one of them is selected; the others are skipped."""
  bases = sorted(layer_dir(root, BASE).glob("*.rrd"))
  if not bases and state is None:
    raise SystemExit(f"{root / 'base'} holds no .rrd; a corpus needs its base layer")
  layers = {
    d.name: files
    for d in sorted(root.iterdir())
    if d.is_dir() and d.name != "base" and (files := sorted(d.glob("*.rrd")))
  }
  states = [n for n in layers if n == STATE or n.startswith(f"{STATE}_")]
  if state is not None and state not in states:
    raise SystemExit(f"{root / state} holds no .rrd; state layers present: {states}")
  skipped = [name for name in states if name != state]
  if skipped:
    print(f"state layers not registered: {skipped}; --state <one> registers it")
  for name in skipped:
    del layers[name]
  return bases, layers


def frame_width(rrd: Path) -> int:
  """The width of the state layer's images: the offset the right
  hand's frames are placed at."""
  placed = Recording(rrd).row(state_image(HANDS[1])).get(POSE_TRANSLATION)
  if placed is None:
    raise SystemExit(f"{rrd} places no {state_image(HANDS[1])} frames")
  return int(placed[0][0])


def written_names(rrd: Path) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
  """Every ``(entity, component)`` and ``(property group, key)`` one file
  writes."""
  columns = RrdReader(str(rrd)).store().schema().component_columns()
  entities = {(c.entity_path, c.component) for c in columns if not c.is_property}
  properties = {(c.entity_path, c.component) for c in columns if c.is_property}
  return entities, properties


def check_state_current(files: list[Path], root: Path) -> None:
  """Refuse a state layer whose estimate is not the estimate beside it."""
  stale = []
  for rrd in files:
    estimate = layer_rrd(root, ESTIMATE, rrd.stem)
    if not estimate.exists():
      print(f"[warn] {rrd.name}: no estimate beside it; staleness NOT checked")
      continue
    written = Recording(rrd).property(STATE).get("estimated_at", [None])[0]
    account = Recording(estimate)
    stamped = {**account.row(ESTIMATE), **account.property(ESTIMATE_PROPERTIES)}
    solved = stamped.get("estimated_at", [None])[0]
    if solved is None:
      print(f"[warn] {estimate.name}: records no estimated_at; staleness NOT checked")
      continue
    if written != solved:
      stale.append(rrd.stem)
  if stale:
    raise SystemExit(f"state layers older than their estimate: {stale}")


def check_one_configuration(dataset: DatasetEntry, state: str) -> None:
  """Refuse a dataset that already holds another state configuration: every
  configuration writes the same names, so two would resolve silently."""
  table = dataset.segment_table().to_arrow_table()
  column = f"property:{STATE}:config"
  if column not in table.schema.names:
    return
  held = {
    v
    for row in table.column(column).to_pylist()
    if row
    for v in (row if isinstance(row, list) else [row])
    if v
  }
  if held - {state}:
    raise SystemExit(
      f"{sorted(held)} is already on the dataset; one dataset takes one "
      "state configuration"
    )


def check_no_collisions(files_by_session: dict[str, list[Path]]) -> None:
  """Refuse a session whose layers write the same name twice.

  A name written by two layers resolves silently: two writers of one
  ``(entity, component)`` resolve by write time, two writers of one property
  key by registration order.
  """
  clashes = []
  for session, files in sorted(files_by_session.items()):
    if len(files) < 2:
      continue
    seen: dict[tuple[str, str], Path] = {}
    seen_properties: dict[tuple[str, str], Path] = {}
    for path in files:
      entities, properties = written_names(path)
      for owned, names in ((seen, entities), (seen_properties, properties)):
        for name in sorted(names):
          if name in owned:
            clashes.append(
              f"{session}: {name[0]}:{name[1]} written by both "
              f"{owned[name].parent.name}/ and {path.parent.name}/"
            )
          else:
            owned[name] = path
  if clashes:
    raise SystemExit("layers collide:\n  " + "\n  ".join(clashes))


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
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  base_files, layer_files = corpus_layers(args.root, args.state)
  # --state builds the training dataset: the state layer as the segments and
  # nothing else, whatever the root holds beside it.
  if args.state:
    check_state_current(layer_files[args.state], args.root)
    segment_layer, segment_files = args.state, layer_files[args.state]
    layer_files = {}
  else:
    segment_layer, segment_files = BASE, base_files
  segments = {p: recording_id(p) for p in segment_files}

  client = CatalogClient(args.catalog_url)
  if args.dataset in client.dataset_names():
    dataset = client.get_dataset(args.dataset)
    known = {str(s) for s in dataset.segment_ids()}
  else:
    dataset = client.create_dataset(args.dataset)
    known = set()
  if args.state:
    check_one_configuration(dataset, args.state)
  layers = {p: recording_id(p) for files in layer_files.values() for p in files}
  by_session: dict[str, list[Path]] = {}
  for path, rid in {**segments, **layers}.items():
    by_session.setdefault(rid, []).append(path)
  check_no_collisions(by_session)
  # Only the estimate layer shares the base's frame clock.
  base_by_id = (
    dict(zip(segments.values(), segments, strict=True)) if segment_layer == BASE else {}
  )
  for path in layer_files.get(ESTIMATE, []):
    base = base_by_id.get(layers[path])
    if base is None:
      print(
        f"[warn] {path.name}: no base under the root; frame alignment NOT "
        "checked (restore the base to verify)"
      )
      continue
    check_frame_alignment(_recording(base).frame_times, path)

  if args.replace_base and segment_layer != BASE:
    raise SystemExit(
      "--replace-base names base segments; a state dataset replaces every segment"
    )
  unknown = set(args.replace_base) - set(segments.values())
  if unknown:
    raise SystemExit(
      f"--replace-base names no base under {args.root}: {sorted(unknown)}"
    )

  # Bases are immutable, so duplicates SKIP; REPLACE rebuilds the catalog's
  # byte-range index. A state layer standing as the segment is regenerated
  # freely, so it always REPLACEs.
  failures: list[str] = []
  replace = set(args.replace_base) if segment_layer == BASE else set(segments.values())
  keep = {p: segment_layer for p, rid in segments.items() if rid not in replace}
  redo = {p: segment_layer for p, rid in segments.items() if rid in replace}
  accepted, kept_bad = register_batch(
    dataset, keep, OnDuplicateSegmentLayer.SKIP, failures
  )
  redone, redone_bad = register_batch(
    dataset, redo, OnDuplicateSegmentLayer.REPLACE, failures
  )
  accepted |= redone
  bad = sum(kept_bad.values()) + sum(redone_bad.values())
  print(
    f"registered {len(segments) - bad} of {len(segments)} {segment_layer} recording(s)"
  )
  if failures:
    raise SystemExit(f"{len(failures)} registration(s) failed; no blueprint installed")
  check_layers_attach(known | accepted, layers)
  # `layer_name` takes one name per URI.
  layer_of = {p: name for name, files in layer_files.items() for p in files}
  _, failed = register_batch(
    dataset, layer_of, OnDuplicateSegmentLayer.REPLACE, failures
  )
  for name, files in layer_files.items():
    print(f"registered {len(files) - failed[name]} {name} layer(s)")
  if failures:
    raise SystemExit(f"{len(failures)} registration(s) failed; no blueprint installed")

  # The catalog rewrites every segment's application id to the dataset id, and
  # a default blueprint applies only where the application id matches.
  blueprint_app_id = str(dataset.id)
  blueprints_dir = args.root / "blueprints"
  blueprints_dir.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory() as tmp:
    if args.state:
      dashboard = state_blueprint()
      dashboard.save(APPLICATION_ID, str(blueprints_dir / "state.rbl"))
      print(f"wrote {(blueprints_dir / 'state.rbl').resolve()}")
      # The card holds the two frames, which is what tells one session from
      # another.
      cards = state_card_blueprint(frame_width(segment_files[0]))
    else:
      paths = [c.entity_path for c in dataset.schema().component_columns()]
      slots = slots_in(paths)
      if not slots:
        raise SystemExit("the dataset holds no camera slots; no blueprint installed")
      anchors = anchor_ids_in(paths)
      if not anchors:
        print("  [warn] the dataset holds no world anchors; the blueprint draws none")
      sizes = anchor_tag_sizes(layer_files.get(ESTIMATE, []))
      if len(sizes) > 1:
        raise SystemExit(
          f"the corpus solved anchors at {sorted(sizes)} m; no blueprint"
        )
      if anchors and not sizes:
        print("  [warn] the estimate layers record no anchor tag size; drawing 0.1 m")
      tag_size = sizes.pop() if sizes else 0.1
      dashboard = blueprint_for_slots(
        slots, anchor_tag_ids=anchors, anchor_tag_size=tag_size
      )
      # One .rbl under the recordings' app id, for session files opened directly.
      corpus_rbl = blueprints_dir / "specialist.rbl"
      dashboard.save(APPLICATION_ID, str(corpus_rbl))
      print(
        f"wrote {corpus_rbl.resolve()}: slots {slots}, "
        f"anchor tags {anchors} at {tag_size} m"
      )
      # The segment table renders every session at once, so its card never
      # decodes the video streams.
      cards = segment_card_blueprint(slots)
    for name, blueprint, table in (
      ("dashboard", dashboard, False),
      ("card", cards, True),
    ):
      if blueprint is None:
        continue
      rbl = Path(tmp) / f"{args.dataset}-{name}.rbl"
      blueprint.save(blueprint_app_id, str(rbl))
      dataset.register_blueprint(
        rbl.resolve().as_uri(), set_default=True, segment_table=table
      )
  print(f"installed the default + segment-table blueprints on {args.dataset!r}")


if __name__ == "__main__":
  main(tyro.cli(Args))
