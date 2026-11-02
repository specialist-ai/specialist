"""The gripper model as dataset assets: one static recording per camera slot,
shared by every segment of a dataset and named by the URDF's digest."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import rerun as rr
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, OptimizationProfile
from rerun.urdf import UrdfTree
from specialist_corpus import schema

__all__ = ["register_assets", "write_gripper_asset"]


def _gripper_chunks(urdf_path: Path, slot: str) -> list[Chunk]:
  """One slot's model: the visual meshes, the fixed joints' edges, and the
  frames of the root and of each revolute joint's link. The revolute edges
  are the solve's to write."""
  tree = UrdfTree.from_file_path(
    str(urdf_path), entity_path_prefix=slot, frame_prefix=f"{slot}_"
  )
  gripper = f"/{schema.gripper(slot)}"
  # The loader escapes slashes in a prefix, so the tree is built at a
  # one-segment root and moved.
  model_root = f"/{slot}/{tree.name}"
  chunks = [
    chunk.with_entity_path(gripper + chunk.entity_path.removeprefix(model_root))
    for chunk in tree.stream(include_joint_transforms=False)
    .drop(content=f"{model_root}/collision_geometries/**")
    .to_chunks()
  ]
  fixed = [joint.name for joint in tree.joints() if joint.joint_type == "fixed"]
  edges = tree.compute_joint_transform_batches(
    pa.array([fixed]), pa.array([[0.0] * len(fixed)])
  ).to_pylist()[0]
  chunks.append(
    Chunk.from_columns(
      f"/{schema.gripper_part(slot, 'tf_static')}",
      indexes=[],
      columns=rr.Transform3D.columns(
        translation=[edge["translation"] for edge in edges],
        quaternion=[edge["quaternion"] for edge in edges],
        parent_frame=[edge["parent_frame"] for edge in edges],
        child_frame=[edge["child_frame"] for edge in edges],
      ),
    )
  )
  frames = {gripper: tree.root_link().name} | {
    f"/{schema.gripper_part(slot, joint.name)}": tree.get_joint_child(joint).name
    for joint in tree.joints()
    if joint.joint_type == "revolute"
  }
  chunks += [
    Chunk.from_columns(
      entity,
      indexes=[],
      columns=rr.CoordinateFrame.columns(frame=[schema.link_frame(slot, link)]),
    )
    for entity, link in frames.items()
  ]
  return chunks


def write_gripper_asset(urdf_path: Path, slot: str, corpus: Path) -> Path:
  """One slot's asset for this URDF under ``corpus/assets``, written once."""
  digest = sha256(urdf_path.read_bytes()).hexdigest()[:12]
  path = schema.asset_rrd(corpus, f"gripper-{digest}-{slot}")
  if path.exists():
    return path
  path.parent.mkdir(parents=True, exist_ok=True)
  store = LazyChunkStream.from_iter(_gripper_chunks(urdf_path, slot)).collect(
    optimize=OptimizationProfile.OBJECT_STORE
  )
  # Concurrent solves write the same asset, so each writes its own file and renames.
  written = path.with_suffix(f".{os.getpid()}")
  with rr.RecordingStream(
    schema.APPLICATION_ID, recording_id=path.stem, send_properties=False
  ) as stream:
    stream.set_sinks(rr.FileSink(written))
    rr.send_chunks(store, recording=stream)
  written.replace(path)
  return path


def register_assets(dataset: DatasetEntry, corpus: Path) -> None:
  """The models under ``corpus/assets`` are the dataset's assets; one the
  corpus no longer holds is unregistered."""
  files = sorted((corpus / schema.ASSETS).glob("*.rrd"))
  for stale in set(dataset.assets()) - {f.stem for f in files}:
    dataset.unregister_asset(stale)
  for f in files:
    dataset.register_asset(f.resolve().as_uri())
  print(f"registered {len(files)} asset(s): {[f.stem for f in files]}", flush=True)
