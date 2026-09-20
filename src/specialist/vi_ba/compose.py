"""Composition of per-tracker subgraphs into one joint VI-BA factor graph.

``TrackerKeys`` allocates a collision-free symbol namespace per tracker.
``assemble_joint_graph`` is the pure "trackers + anchors → one gtsam graph"
step; solving that graph is ``specialist.vi_ba.reference_solve``.

Vocabulary, held throughout the solve side: a **detection** is frontend
output, one tag in one frame (``TagDetection``); a **sighting** is a
``(observer, keyframe, lens, detection)`` tuple selected to become a factor
(``CrossSighting``, ``anchor_sightings``); an **observation** is a pixel
measurement of a landmark at a keyframe (the inner ``{kf: pixel}`` of a
track).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import gtsam
import numpy as np
import numpy.typing as npt
from specialist_camera import MeiParams
from specialist_corpus.fiducials import tag_corners_local

from specialist.vi_ba.anchor_detection import (
  TagDetection,
  estimate_camera_to_tag_pose,
  tag_center_bearing,
)
from specialist.vi_ba.build import (
  LensChannel,
  LensSetup,
  add_mei_anchor_tag_factor,
  add_mei_seated_base_tag_factor,
  add_mei_seated_cross_finger_factor,
  add_mei_seated_cross_tag_factor,
  add_mei_seated_finger_factor,
  build_vi_graph,
  huber_pixel_noise,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.symbols import (
  ANGLE_CHAR,
  BIAS_CHAR,
  LANDMARK_CHAR,
  MAX_KEYFRAMES_PER_TRACKER,
  POSE_CHAR,
  VELOCITY_CHAR,
)
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

Float64Array = npt.NDArray[np.float64]

# One lens's surviving observations, ``{track_id: {keyframe: pixel}}``.
LandmarkTracks = dict[int, dict[int, Float64Array]]
# Per tracker name, the landmark map its front lens carries.
LandmarkKeep = dict[str, LandmarkTracks]
__all__ = [
  "CrossSighting",
  "LandmarkKeep",
  "LandmarkTracks",
  "TrackerKeys",
  "align_to_shared_world",
  "anchor_sightings",
  "anchored_keyframes",
  "assemble_joint_graph",
  "cross_sightings",
  "points_from_values",
  "seat_evidence",
]

_BASE_TAG_CHAR = "G"
_SEAT_CHAR = "E"
_ANCHOR_CHAR = "A"

# Finger joint-angle recovery: per-corner reprojection σ (px) and the soft
# equal-and-opposite symmetry prior σ (rad) on θ_left + θ_right.
_FINGER_REPROJ_SIGMA = 1.5
_FINGER_SYMMETRY_SIGMA = 0.05
# Corner reprojection σ (px) and the PnP-seeded prior (rotation rad,
# translation m) on the base-tag pose variable.
_BASE_TAG_REPROJ_SIGMA = 1.5
_BASE_TAG_PRIOR_SIGMA_ROT = 1.0
_BASE_TAG_PRIOR_SIGMA_TRANS = 0.3
# Cross-tracker sighting σ (px), weighted apart from the per-arm base-tag
# factors.
_CROSS_REPROJ_SIGMA = 3.0
# Anchor corner noise growth past the lens's calibrated radius, in px per unit
# of fractional overshoot; the floor is ``AnchorCfg.corner_reproj_sigma``.
_SIGMA_PER_RADIUS = 40.0
# The datum holds yaw and position; the horizontal rotations are left to the IMU.
_DATUM_SIGMA = 1e-4
_DATUM_FREE_SIGMA = 1e3
# Rear-seat rotation prior (rad), loose enough for the measured ~9 deg seat errors.
_SEAT_PRIOR_SIGMA_ROT = 0.2

_MAX_LANDMARKS_PER_TRACKER = 1_000_000


class TrackerKeys:
  """Each tracker gets a unique slice of the integer-key space, indexed by
  ``MAX_KEYFRAMES_PER_TRACKER`` for poses/velocities/biases and
  ``_MAX_LANDMARKS_PER_TRACKER`` for its landmarks. Anchors are shared by every
  tracker, so they key off the tag id alone."""

  def __init__(
    self,
    tracker_names: Sequence[str],
  ) -> None:
    if len(set(tracker_names)) != len(tracker_names):
      raise ValueError(f"tracker names must be unique: {list(tracker_names)}")
    self._index = {name: i for i, name in enumerate(tracker_names)}

  def tracker_index(self, name: str) -> int:
    if name not in self._index:
      raise KeyError(f"unknown tracker {name!r}")
    return self._index[name]

  def _kf_key(self, char: str, name: str, kf: int) -> int:
    """Collision-free key for a per-keyframe variable in one tracker's slice."""
    return int(
      gtsam.symbol(char, self.tracker_index(name) * MAX_KEYFRAMES_PER_TRACKER + kf)
    )

  def pose_key(self, name: str, kf: int) -> int:
    return self._kf_key(POSE_CHAR, name, kf)

  def velocity_key(self, name: str, kf: int) -> int:
    return self._kf_key(VELOCITY_CHAR, name, kf)

  def bias_key(self, name: str, kf: int) -> int:
    return self._kf_key(BIAS_CHAR, name, kf)

  def front_landmark_key(self, name: str, track_id: int) -> int:
    """Key for a landmark in one tracker's landmark slot. Track ids are handed
    out monotonically and never renumbered, so a long capture can outgrow its
    slot; the bound is checked."""
    if not 0 <= track_id < _MAX_LANDMARKS_PER_TRACKER:
      raise ValueError(
        f"track id {track_id} does not fit the {_MAX_LANDMARKS_PER_TRACKER}-wide "
        f"landmark slot for {name!r}; its key would collide with another slot"
      )
    return int(
      gtsam.symbol(
        LANDMARK_CHAR, self.tracker_index(name) * _MAX_LANDMARKS_PER_TRACKER + track_id
      )
    )

  def theta_key(self, name: str, finger_idx: int, kf: int) -> int:
    """Key for a finger's revolute joint angle (scalar) at one keyframe.
    ``finger_idx`` 0 = left, 1 = right; collision-free per tracker and finger."""
    return self._slot_key(ANGLE_CHAR, name, finger_idx, MAX_KEYFRAMES_PER_TRACKER, kf)

  def _slot_key(self, char: str, name: str, half: int, width: int, index: int) -> int:
    """One integer key in the tracker's ``half`` slice of a ``width``-wide slot."""
    return int(
      gtsam.symbol(char, (self.tracker_index(name) * 2 + half) * width + index)
    )

  def anchor_key(self, tag_id: int) -> int:
    """Key for a wall tag's world pose, shared by every tracker that sights it."""
    return int(gtsam.symbol(_ANCHOR_CHAR, tag_id))

  def base_tag_key(self, name: str) -> int:
    """Key for a tracker's base-tag pose in its body frame (one static ``Pose3``
    shared across all keyframes; the base + finger tags reproject through it)."""
    return int(gtsam.symbol(_BASE_TAG_CHAR, self.tracker_index(name)))

  def seat_key(self, name: str) -> int:
    """Key for a tracker's rear-lens seat rotation (one static ``Rot3``; the
    translation is baked): every rear-lens tag factor reprojects through it."""
    return int(gtsam.symbol(_SEAT_CHAR, self.tracker_index(name)))


def _bfs_world_alignments(
  anchor_locals_by_tracker: dict[str, dict[int, gtsam.Pose3]],
  sightings_by_tracker: dict[str, dict[int, int]],
  anchor_tag_ids: tuple[int, ...],
) -> tuple[dict[str, gtsam.Pose3], dict[int, gtsam.Pose3]]:
  """Per-tracker world-alignment transforms via BFS over the tracker-anchor
  co-visibility graph, rooted at the reference anchor (first id). Each anchor is
  placed by the tracker that sighted it most among those the same wave aligned.
  Returns ``({tracker: align}, {tag_id: world_T_anchor})``; raises when a
  tracker cannot reach the shared world."""
  ref_tag_id = anchor_tag_ids[0]
  ref = next(
    (
      locals_[ref_tag_id]
      for locals_ in anchor_locals_by_tracker.values()
      if ref_tag_id in locals_
    ),
    None,
  )
  if ref is None:
    raise RuntimeError("no tracker observes the reference anchor; cannot share a world")
  world_T_anchor: dict[int, gtsam.Pose3] = {ref_tag_id: ref}
  aligns: dict[str, gtsam.Pose3] = {}
  pending = list(anchor_locals_by_tracker)
  while pending:
    aligned_now: list[str] = []
    for name in pending:
      locals_ = anchor_locals_by_tracker[name]
      via = next(
        (a for a in anchor_tag_ids if a in world_T_anchor and a in locals_), None
      )
      if via is None:
        continue
      aligns[name] = world_T_anchor[via].compose(locals_[via].inverse())
      aligned_now.append(name)
    if not aligned_now:
      raise RuntimeError(
        f"trackers {pending} sight no anchor with an established world pose "
        f"(established anchors: {sorted(world_T_anchor)}); the tracker-anchor "
        f"co-visibility graph is disconnected"
      )
    for tag_id in {t for n in aligned_now for t in anchor_locals_by_tracker[n]}:
      if tag_id in world_T_anchor:
        continue
      best = max(
        (n for n in aligned_now if tag_id in anchor_locals_by_tracker[n]),
        key=lambda n: sightings_by_tracker[n].get(tag_id, 0),
      )
      world_T_anchor[tag_id] = aligns[best].compose(
        anchor_locals_by_tracker[best][tag_id]
      )
    pending = [n for n in pending if n not in aligned_now]
  return aligns, world_T_anchor


def align_to_shared_world(
  trackers: list[PreparedTracker], anchor_tag_ids: tuple[int, ...]
) -> dict[int, gtsam.Pose3]:
  """Transform every tracker's body poses, velocities, and landmarks into the
  shared world, connecting trackers through any co-sighted anchor. The world
  keeps the reference tracker's gravity-aligned vertical and is placed on the
  reference anchor: the tag sits at the origin with its x axis projected to
  horizontal along world x. Returns every sighted anchor's world pose, the
  secondaries each seeded from the tracker that sighted it most."""
  locals_by_tracker = {t.name: t.anchor_locals_by_tag for t in trackers}
  sightings_by_tracker = {
    t.name: {tag_id: len(obs) for tag_id, obs in t.anchor_obs_by_tag.items()}
    for t in trackers
  }
  aligns, world_T_anchor = _bfs_world_alignments(
    locals_by_tracker, sightings_by_tracker, anchor_tag_ids
  )
  ref = world_T_anchor[anchor_tag_ids[0]]
  x_axis = ref.rotation().matrix()[:, 0]
  yaw = gtsam.Rot3.Rz(-float(np.arctan2(x_axis[1], x_axis[0])))
  canon = gtsam.Pose3(yaw, yaw.rotate(gtsam.Point3(*-np.asarray(ref.translation()))))
  aligns = {name: canon.compose(align) for name, align in aligns.items()}
  world_T_anchor = {
    tag_id: canon.compose(pose) for tag_id, pose in world_T_anchor.items()
  }
  for t in trackers:
    align = aligns[t.name]
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    for ch in t.lenses:
      ch.points = {tid: align_R @ p + align_t for tid, p in ch.points.items()}
  return world_T_anchor


@dataclass(frozen=True)
class CrossSighting:
  """One observer-lens sighting of another tracker's tag, for a cross factor."""

  observer: str
  owner: str
  keyframe: int
  lens: LensSetup
  corners: Float64Array  # (4, 2) detected corner pixels
  bearing: Float64Array  # (3,) lens-frame tag direction
  finger: int | None  # None = base tag; 0 = left finger, 1 = right finger


def cross_sightings(
  trackers: list[PreparedTracker],
  base_tag_size: float,
) -> list[CrossSighting]:
  """Every sighting of one tracker's tags (base + fingers) in another tracker's
  rear-lens frames, feeding the cross-tracker corner-reprojection factors.
  Captures are audio-synced, so observer and owner share the keyframe index. A
  PnP solve gates degenerate / behind-camera base-tag detections; the finger
  factors rely on the reprojection factor's own cheirality guard."""
  for a, b in itertools.combinations(trackers, 2):
    n = min(a.n_keyframes, b.n_keyframes)
    if not np.array_equal(a.keyframe_rows[:n], b.keyframe_rows[:n]):
      raise ValueError(
        f"keyframe grids of {a.name!r} and {b.name!r} disagree over their "
        f"shared span; cross-tracker sightings would pair the wrong instants"
      )
  sightings: list[CrossSighting] = []
  for owner in trackers:
    # The owner tags an observer may see: the base tag, plus each finger tag
    # that carries URDF joint kinematics.
    targets: dict[int, int | None] = {owner.tags.base: None}
    fingers = zip(FINGER_JOINTS, owner.tags.fingers, strict=True)
    for idx, (name, finger_id) in enumerate(fingers):
      if finger_id > 0 and name in owner.finger_reproj:
        targets[finger_id] = idx
    for observer in trackers:
      if observer is owner:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      ch = observer.rear
      for kf in range(n):
        for det in ch.tags[kf]:
          tag_id = int(det.tag_id)
          if tag_id not in targets:
            continue
          finger = targets[tag_id]
          if finger is None and (
            estimate_camera_to_tag_pose(det.corners, base_tag_size, ch.setup.params)
            is None
          ):
            continue  # base-tag PnP gate: drop degenerate / behind-camera
          sightings.append(
            CrossSighting(
              observer=observer.name,
              owner=owner.name,
              keyframe=kf,
              lens=ch.setup,
              corners=det.corners,
              bearing=tag_center_bearing(det, ch.setup.params),
              finger=finger,
            )
          )
  return sightings


def _finger_symmetry_factor(
  key_l: int, key_r: int, noise: gtsam.noiseModel.Base
) -> gtsam.LinearContainerFactor:
  """Soft URDF mimic: residual = θ_left + θ_right, the fingers moving
  equal-and-opposite. The residual is linear, so a container around a
  unit-Jacobian linear factor is exact at every linearization point."""
  jacobian = gtsam.JacobianFactor(
    key_l, np.array([[1.0]]), key_r, np.array([[1.0]]), np.zeros(1), noise
  )
  lin_point = gtsam.Values()
  lin_point.insert(key_l, 0.0)
  lin_point.insert(key_r, 0.0)
  return gtsam.LinearContainerFactor(jacobian, lin_point)


def anchor_corner_sigma(
  max_radius: float, calibrated_radius: float, base_sigma: float
) -> float:
  """Corner-reprojection sigma (px) for a sighting whose furthest corner sits
  at ``max_radius`` from the principal point: the noise floor and the growth
  past the lens's calibrated radius, added in quadrature."""
  over = max(0.0, max_radius / calibrated_radius - 1.0)
  return float(np.hypot(base_sigma, _SIGMA_PER_RADIUS * over))


def sighting_max_radius(corners: Float64Array, params: MeiParams) -> float:
  """How far the sighting's furthest corner sits from the principal point."""
  pixels = np.asarray(corners, dtype=np.float64)
  return float(np.hypot(pixels[:, 0] - params.cx, pixels[:, 1] - params.cy).max())


def corner_noise(
  corners: Float64Array, setup: LensSetup, base_sigma: float
) -> gtsam.noiseModel.Base:
  """The robust noise model one tag sighting's four corner factors carry, priced
  by the radius the sighting sits at."""
  return huber_pixel_noise(
    anchor_corner_sigma(
      sighting_max_radius(corners, setup.params), setup.calibrated_radius, base_sigma
    )
  )


def anchor_datum_factor(
  anchor_key: int, world_T_anchor: gtsam.Pose3
) -> gtsam.PriorFactorPose3:
  """The graph's whole datum: the anchor's position and its heading about
  gravity, saying nothing about the two horizontal rotations."""
  rotation = world_T_anchor.rotation().matrix()
  up = np.array([0.0, 0.0, 1.0])
  free = np.eye(3) - np.outer(up, up)
  world_cov = _DATUM_FREE_SIGMA**2 * free + _DATUM_SIGMA**2 * np.outer(up, up)
  cov = np.zeros((6, 6))
  cov[:3, :3] = rotation.T @ world_cov @ rotation
  cov[3:, 3:] = _DATUM_SIGMA**2 * np.eye(3)
  return gtsam.PriorFactorPose3(
    anchor_key, world_T_anchor, gtsam.noiseModel.Gaussian.Covariance(cov)
  )


def anchor_sightings(
  tracker: PreparedTracker,
  anchor_tag_id: int,
) -> list[tuple[LensChannel, int, TagDetection]]:
  """Per-keyframe anchor-tag detections in both lenses."""
  return [
    (ch, kf, det)
    for ch in tracker.lenses
    for kf, dets in enumerate(ch.tags)
    for det in dets
    if int(det.tag_id) == anchor_tag_id
  ]


def anchored_keyframes(tracker: PreparedTracker) -> set[int]:
  """Keyframes with a PnP sighting of any configured anchor; the smoothness
  priors treat these as anchored."""
  return set().union(*tracker.anchor_obs_by_tag.values())


def _add_corner_factors_for_anchor(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  base_sigma: float,
  tag_id: int,
  tag_size: float,
  graph: gtsam.NonlinearFactorGraph,
) -> int:
  """Corner-reprojection factors for one anchor, every sighting by any tracker
  and lens, with the detection-center ray as the cheirality bearing and a noise
  model scaled to each sighting's corner radius; rear sightings reproject through
  the tracker's solved seat. Returns the sighting count."""
  corners = tag_corners_local(tag_size)
  anchor_key = keys.anchor_key(tag_id)
  n_sightings = 0
  for t in trackers:
    for ch, kf, det in anchor_sightings(t, tag_id):
      n_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      bearing = tag_center_bearing(det, ch.setup.params)
      sighting_noise = corner_noise(det.corners, ch.setup, base_sigma)
      seat_key = keys.seat_key(t.name) if ch.side == "rear" else None
      for corner, pixel in zip(corners, det.corners, strict=True):
        add_mei_anchor_tag_factor(
          graph,
          pose_key,
          anchor_key,
          pixel,
          sighting_noise,
          ch.setup,
          corner,
          bearing,
          seat_key=seat_key,
        )
  return n_sightings


def _add_vi_subgraphs(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """One VI subgraph (poses, velocities, biases, landmarks, IMU factors,
  visual reprojection) per tracker, grounded by the anchors."""
  for t in trackers:
    build_vi_graph(
      body_poses=t.body_poses,
      velocities=t.velocities,
      biases=t.biases,
      preintegrated=t.preintegrated,
      front=t.front,
      sigma=args.frontend.reproj_sigma,
      velocity_prior_sigma=args.solver.velocity_prior_sigma,
      graph=graph,
      values=values,
      pose_key_fn=partial(keys.pose_key, t.name),
      velocity_key_fn=partial(keys.velocity_key, t.name),
      bias_key_fn=partial(keys.bias_key, t.name),
      front_landmark_key_fn=partial(keys.front_landmark_key, t.name),
    )


def _add_anchor_factors(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  anchor_world: dict[int, gtsam.Pose3],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """Every anchor as one rigid pose variable seeded from the initializer, with
  the graph's single datum on the reference and the sightings holding the
  rest."""
  base_sigma = args.anchor.corner_reproj_sigma
  reference = int(args.anchor.enabled_tag_ids[0])
  for tag_id, world_T_tag in anchor_world.items():
    key = keys.anchor_key(tag_id)
    values.insert(key, world_T_tag)
    n = _add_corner_factors_for_anchor(
      trackers, keys, base_sigma, tag_id, args.anchor.tag_size, graph
    )
    print(f"[graph] anchor {tag_id} corner reprojection: {n} sightings")
  for tag_id in args.anchor.enabled_tag_ids:
    if int(tag_id) in anchor_world:
      continue
    rear = sum(
      1
      for t in trackers
      for ch, _kf, _det in anchor_sightings(t, int(tag_id))
      if ch.side == "rear"
    )
    print(
      f"  [warn] [graph] anchor {tag_id} has no world pose and carries no "
      f"factors: prep resolves an anchor in the front lens alone, and this one "
      f"was sighted there by no tracker ({rear} rear sightings unused)"
    )
  graph.add(anchor_datum_factor(keys.anchor_key(reference), anchor_world[reference]))


def _add_seat_variables(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """One rear-seat rotation variable per tracker, seeded from the factory
  block."""
  noise = gtsam.noiseModel.Isotropic.Sigma(3, _SEAT_PRIOR_SIGMA_ROT)
  for t in trackers:
    key = keys.seat_key(t.name)
    rot = t.rear.setup.extrinsic.rotation()
    values.insert(key, rot)
    graph.add(gtsam.PriorFactorRot3(key, rot, noise))


def seat_evidence(
  tracker: PreparedTracker,
  solved_anchor_ids: Iterable[int],
  cross_obs: list[CrossSighting],
) -> tuple[int, int]:
  """The sightings that can measure a tracker's seat: rear-lens anchor corners
  on anchors the graph solved, and the cross-tracker corners it observed. A
  cross factor is seated on its observer alone."""
  anchors = sum(
    1
    for tag_id in solved_anchor_ids
    for ch, _kf, _det in anchor_sightings(tracker, int(tag_id))
    if ch.side == "rear"
  )
  cross = sum(1 for s in cross_obs if s.observer == tracker.name)
  return anchors, cross


def _add_gripper_factors(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> tuple[int, int]:
  """Base + finger tag factors. The base-tag pose in the body frame
  (body_T_base_tag) is one static BA variable that the base tag (directly) and
  the finger tags (through their URDF joints) jointly pin, seeded by a loose
  prior from PnP. Each finger's scalar joint angle θ co-optimizes with it, tied
  equal-and-opposite by a soft prior, so at a keyframe with only one finger
  visible the other's θ is inferred."""
  base_tag_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([_BASE_TAG_PRIOR_SIGMA_ROT] * 3 + [_BASE_TAG_PRIOR_SIGMA_TRANS] * 3)
  )
  sym_noise = gtsam.noiseModel.Isotropic.Sigma(1, _FINGER_SYMMETRY_SIGMA)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  finger_factor_count = 0
  base_tag_factor_count = 0
  for t in trackers:
    reprojs = {
      idx: reproj
      for idx, joint in enumerate(FINGER_JOINTS)
      if (reproj := t.finger_reproj.get(joint)) is not None
    }
    bt_key = keys.base_tag_key(t.name)
    values.insert(bt_key, t.body_T_base_tag)
    graph.add(gtsam.PriorFactorPose3(bt_key, t.body_T_base_tag, base_tag_prior_noise))
    # Base-tag self-reprojection: the base tag's four metric corners reproject
    # through body_T_base_tag and the seat into this tracker's own rear lens.
    for dets in t.rear.tags:
      det = next((d for d in dets if int(d.tag_id) == t.tags.base), None)
      if det is None:
        continue
      bearing = tag_center_bearing(det, t.rear.setup.params)
      base_tag_noise = corner_noise(det.corners, t.rear.setup, _BASE_TAG_REPROJ_SIGMA)
      for j in range(4):
        add_mei_seated_base_tag_factor(
          graph,
          bt_key,
          keys.seat_key(t.name),
          det.corners[j],
          base_tag_noise,
          t.rear.setup,
          base_tag_corners[j],
          bearing,
        )
        base_tag_factor_count += 1
    both_fingers = len(reprojs) == 2
    observed_kf = sorted({kf for r in reprojs.values() for kf in r.observations})
    for kf in observed_kf:
      seen_seed: dict[int, float] = {}
      for finger_idx, reproj in reprojs.items():
        obs = reproj.observations.get(kf)
        if obs is None:
          continue
        tkey = keys.theta_key(t.name, finger_idx, kf)
        values.insert(tkey, float(obs.theta_seed))
        finger_noise = corner_noise(obs.pixels, t.rear.setup, _FINGER_REPROJ_SIGMA)
        for j in range(4):
          add_mei_seated_finger_factor(
            graph,
            tkey,
            bt_key,
            keys.seat_key(t.name),
            obs.pixels[j],
            finger_noise,
            t.rear.setup,
            reproj.base_T_joint,
            reproj.corners_in_pivot[j],
            obs.obs_bearing,
          )
          finger_factor_count += 1
        seen_seed[finger_idx] = float(obs.theta_seed)
      if both_fingers:
        for finger_idx in (0, 1):
          if finger_idx not in seen_seed:  # unseen finger: inferred from the prior only
            values.insert(
              keys.theta_key(t.name, finger_idx, kf), -next(iter(seen_seed.values()))
            )
        graph.add(
          _finger_symmetry_factor(
            keys.theta_key(t.name, 0, kf), keys.theta_key(t.name, 1, kf), sym_noise
          )
        )
  return base_tag_factor_count, finger_factor_count


def _add_cross_tracker_factors(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> tuple[int, int]:
  """Cross-tracker corner reprojection: when observer O sees owner W's base tag,
  its corners reproject through O's pose, W's pose, W's shared base-tag variable
  and O's seat, coupling the two trajectories. Seeing W's finger tag adds W's
  joint angle θ, which must already carry a variable at that keyframe."""
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  tracker_by_name = {t.name: t for t in trackers}
  base_count = finger_count = 0
  for sighting in cross_obs:
    if not values.exists(keys.base_tag_key(sighting.owner)):
      continue
    obs_key = keys.pose_key(sighting.observer, sighting.keyframe)
    owner_key = keys.pose_key(sighting.owner, sighting.keyframe)
    owner_bt_key = keys.base_tag_key(sighting.owner)
    seat_key = keys.seat_key(sighting.observer)
    base_tag_noise = corner_noise(sighting.corners, sighting.lens, _CROSS_REPROJ_SIGMA)
    if sighting.finger is None:
      for j in range(4):
        add_mei_seated_cross_tag_factor(
          graph,
          obs_key,
          owner_key,
          owner_bt_key,
          seat_key,
          sighting.corners[j],
          base_tag_noise,
          sighting.lens,
          base_tag_corners[j],
          sighting.bearing,
        )
        base_count += 1
    elif values.exists(
      keys.theta_key(sighting.owner, sighting.finger, sighting.keyframe)
    ):
      finger_name = FINGER_JOINTS[sighting.finger]
      reproj = tracker_by_name[sighting.owner].finger_reproj[finger_name]
      theta_key = keys.theta_key(sighting.owner, sighting.finger, sighting.keyframe)
      for j in range(4):
        add_mei_seated_cross_finger_factor(
          graph,
          obs_key,
          owner_key,
          owner_bt_key,
          theta_key,
          seat_key,
          sighting.corners[j],
          base_tag_noise,
          sighting.lens,
          reproj.base_T_joint,
          reproj.corners_in_pivot[j],
          sighting.bearing,
        )
        finger_count += 1
  return base_count, finger_count


def _add_smoothness_priors(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  graph: gtsam.NonlinearFactorGraph,
) -> tuple[int, int]:
  """Smoothness priors across stretches where both endpoints are unanchored.
  Each such pair gets a Pose3 BetweenFactor whose measurement keeps the prep
  relative rotation with zero translation, plus a BetweenFactorVector on the
  prep velocity delta. Returns the pose and velocity factor counts."""
  smooth_sigmas = np.array([np.deg2rad(2.0)] * 3 + [0.10] * 3, dtype=np.float64)
  smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(smooth_sigmas)
  vel_smooth_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.3, 0.3, 0.3], dtype=np.float64)
  )
  smooth_count = 0
  vel_smooth_count = 0
  zero_t = gtsam.Point3(0.0, 0.0, 0.0)
  for t in trackers:
    anchored_kfs = anchored_keyframes(t)
    for k in range(1, t.n_keyframes):
      # A boundary pair (one endpoint anchored) would pull the unanchored pose
      # toward the anchor-pinned one.
      if (k - 1) in anchored_kfs or k in anchored_kfs:
        continue
      # The prep-stage relative rotation is IMU-propagated through the
      # VI-aligned prep poses.
      prep_rel = t.body_poses[k - 1].inverse().compose(t.body_poses[k])
      smooth_mean = gtsam.Pose3(prep_rel.rotation(), zero_t)
      graph.add(
        gtsam.BetweenFactorPose3(
          keys.pose_key(t.name, k - 1),
          keys.pose_key(t.name, k),
          smooth_mean,
          smooth_noise,
        )
      )
      smooth_count += 1
      vel_delta = np.asarray(t.velocities[k] - t.velocities[k - 1], dtype=np.float64)
      graph.add(
        gtsam.BetweenFactorVector(
          keys.velocity_key(t.name, k - 1),
          keys.velocity_key(t.name, k),
          vel_delta,
          vel_smooth_noise,
        )
      )
      vel_smooth_count += 1
  return smooth_count, vel_smooth_count


def assemble_joint_graph(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  cross_obs: list[CrossSighting],
  anchor_world: dict[int, gtsam.Pose3],
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Build the joint factor graph: per-tracker VI subgraphs, anchor pose
  variables with their corner reprojection, gripper base/finger factors,
  cross-tracker reprojection, and smoothness priors on unanchored tails."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  _add_vi_subgraphs(trackers, keys, args, graph, values)
  _add_seat_variables(trackers, keys, graph, values)
  _add_anchor_factors(trackers, keys, args, anchor_world, graph, values)
  base_tags, fingers = _add_gripper_factors(trackers, keys, args, graph, values)
  cross_base, cross_finger = _add_cross_tracker_factors(
    trackers, keys, args, cross_obs, graph, values
  )
  poses_smoothed, velocities_smoothed = _add_smoothness_priors(trackers, keys, graph)
  if base_tags or fingers:
    print(
      f"[graph] base-tag reprojection: {base_tags} factors; "
      f"finger joint-angle reprojection: {fingers} factors"
    )
  if cross_base:
    print(f"[graph] cross-tracker reprojection: {cross_base} factors")
  if cross_finger:
    print(f"[graph] finger-tag cross-tracker reprojection: {cross_finger} factors")
  if poses_smoothed:
    print(f"[graph] unanchored-stretch pose smoothness priors: {poses_smoothed}")
  if velocities_smoothed:
    print(
      f"[graph] unanchored-stretch velocity smoothness priors: {velocities_smoothed}"
    )
  for t in trackers:
    anchors, cross = seat_evidence(t, anchor_world, cross_obs)
    print(f"[graph] {t.name} rear-seat evidence: {anchors} anchor, {cross} cross")
  return graph, values


def points_from_values(
  values: gtsam.Values,
  key_fn: Callable[[str, int], int],
  name: str,
  tids: Iterable[int],
) -> dict[int, Float64Array]:
  """Pull the solved 3-D landmark points out of ``values`` for ``tids``."""
  return {
    tid: np.asarray(values.atPoint3(key_fn(name, tid)), dtype=np.float64)
    for tid in tids
  }
