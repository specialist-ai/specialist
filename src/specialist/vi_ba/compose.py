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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import gtsam
import numpy as np
import numpy.typing as npt

from specialist.vi_ba.anchor_detection import (
  TagDetection,
  estimate_camera_to_tag_pose,
  tag_center_bearing,
  tag_corners_in_frame,
  tag_corners_local,
  tag_half_angle,
)
from specialist.vi_ba.build import (
  LensChannel,
  LensSetup,
  add_mei_base_tag_factor,
  add_mei_cross_finger_factor,
  add_mei_cross_tag_factor,
  add_mei_factor,
  add_mei_finger_factor,
  add_mei_world_point_factor,
  build_vi_graph,
  huber_pixel_noise,
)
from specialist.vi_ba.configs import Args
from specialist.vi_ba.single_tracker import PreparedTracker
from specialist.vi_ba.symbols import LANDMARK_CHAR
from specialist.vi_ba.urdf_joints import FINGER_JOINTS

Float64Array = npt.NDArray[np.float64]

# One lens's surviving observations, ``{track_id: {keyframe: pixel}}``.
LandmarkTracks = dict[int, dict[int, Float64Array]]
# One lens's triangulated points, ``{track_id: world_point}``.
LandmarkPoints = dict[int, Float64Array]
# Per tracker name, the ``(front, rear)`` pair each lens map comes in.
LandmarkKeep = dict[str, tuple[LandmarkTracks, LandmarkTracks]]
__all__ = [
  "CrossSighting",
  "LandmarkKeep",
  "LandmarkPoints",
  "LandmarkTracks",
  "TrackerKeys",
  "align_to_shared_world",
  "anchor_sightings",
  "anchored_keyframes",
  "assemble_joint_graph",
  "cross_sightings",
  "points_from_values",
]

_BASE_TAG_CHAR = "G"
_POSE_CHAR = "p"
_VEL_CHAR = "v"
_BIAS_CHAR = "b"
_ANGLE_CHAR = "a"
_SEC_CORNER_CHAR = "n"

# Finger joint-angle recovery: per-corner reprojection σ (px) and the soft
# equal-and-opposite symmetry prior σ (rad) on θ_left + θ_right.
_FINGER_REPROJ_SIGMA = 1.5
_FINGER_SYMMETRY_SIGMA = 0.05
# The base-tag pose is a BA variable, pinned by reprojecting its corners (σ px)
# and by a loose PnP-seeded prior (rotation rad / translation m).
_BASE_TAG_REPROJ_SIGMA = 1.5
_BASE_TAG_PRIOR_SIGMA_ROT = 0.3
_BASE_TAG_PRIOR_SIGMA_TRANS = 0.05
# Cross-tracker sighting σ (px), weighted apart from the per-arm base-tag
# factors.
_CROSS_REPROJ_SIGMA = 3.0

_MAX_KEYFRAMES_PER_TRACKER = 100_000
_MAX_LANDMARKS_PER_TRACKER = 1_000_000


class TrackerKeys:
  """Each tracker gets a unique slice of the integer-key space, indexed by
  ``_MAX_KEYFRAMES_PER_TRACKER`` for poses/velocities/biases and
  ``_MAX_LANDMARKS_PER_TRACKER`` for each lens's landmarks. Anchor tag corners
  are constants baked into unary factors, not variables, so they hold no
  keys."""

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
      gtsam.symbol(char, self.tracker_index(name) * _MAX_KEYFRAMES_PER_TRACKER + kf)
    )

  def pose_key(self, name: str, kf: int) -> int:
    return self._kf_key(_POSE_CHAR, name, kf)

  def velocity_key(self, name: str, kf: int) -> int:
    return self._kf_key(_VEL_CHAR, name, kf)

  def bias_key(self, name: str, kf: int) -> int:
    return self._kf_key(_BIAS_CHAR, name, kf)

  def _lm_key(self, name: str, half: int, track_id: int) -> int:
    """Key for a landmark in one tracker's front (``half=0``) or rear
    (``half=1``) lens slot.

    Track ids are handed out monotonically by the frontend and never renumbered,
    so a long capture that keeps redetecting can outgrow its slot. A key that
    lands in the next slot is inserted once and then shared, fusing two
    unrelated points into one variable, so the bound is checked rather than
    assumed.
    """
    if not 0 <= track_id < _MAX_LANDMARKS_PER_TRACKER:
      raise ValueError(
        f"track id {track_id} does not fit the {_MAX_LANDMARKS_PER_TRACKER}-wide "
        f"landmark slot for {name!r}; its key would collide with another slot"
      )
    return self._slot_key(
      LANDMARK_CHAR, name, half, _MAX_LANDMARKS_PER_TRACKER, track_id
    )

  def front_landmark_key(self, name: str, track_id: int) -> int:
    return self._lm_key(name, 0, track_id)

  def rear_landmark_key(self, name: str, track_id: int) -> int:
    return self._lm_key(name, 1, track_id)

  def theta_key(self, name: str, finger_idx: int, kf: int) -> int:
    """Key for a finger's revolute joint angle (scalar) at one keyframe.
    ``finger_idx`` 0 = left, 1 = right; collision-free per tracker and finger."""
    return self._slot_key(_ANGLE_CHAR, name, finger_idx, _MAX_KEYFRAMES_PER_TRACKER, kf)

  def _slot_key(self, char: str, name: str, half: int, width: int, index: int) -> int:
    """One integer key in the tracker's ``half`` slice of a ``width``-wide slot."""
    return int(
      gtsam.symbol(char, (self.tracker_index(name) * 2 + half) * width + index)
    )

  def secondary_corner_key(self, tag_id: int, corner: int) -> int:
    """Key for one freed secondary-anchor corner; tags own disjoint 4-slots."""
    return int(gtsam.symbol(_SEC_CORNER_CHAR, tag_id * 4 + corner))

  def base_tag_key(self, name: str) -> int:
    """Key for a tracker's base-tag pose in its body frame (one static ``Pose3``
    shared across all keyframes; the base + finger tags reproject through it)."""
    return int(gtsam.symbol(_BASE_TAG_CHAR, self.tracker_index(name)))


def _bfs_world_alignments(
  anchor_locals_by_tracker: dict[str, dict[int, gtsam.Pose3]],
  anchor_tag_ids: tuple[int, ...],
) -> tuple[dict[str, gtsam.Pose3], dict[int, gtsam.Pose3]]:
  """Per-tracker world-alignment transforms via BFS over the tracker-anchor
  co-visibility graph, rooted at the reference anchor (first id). Returns
  ``({tracker: align}, {tag_id: world_T_anchor})``; raises when a tracker
  cannot reach the shared world."""
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
      align = world_T_anchor[via].compose(locals_[via].inverse())
      aligns[name] = align
      for tag_id, local in locals_.items():
        if tag_id not in world_T_anchor:
          world_T_anchor[tag_id] = align.compose(local)
      aligned_now.append(name)
    if not aligned_now:
      raise RuntimeError(
        f"trackers {pending} sight no anchor with an established world pose "
        f"(established anchors: {sorted(world_T_anchor)}); the tracker-anchor "
        f"co-visibility graph is disconnected"
      )
    pending = [n for n in pending if n not in aligned_now]
  return aligns, world_T_anchor


def align_to_shared_world(
  trackers: list[PreparedTracker], anchor_tag_ids: tuple[int, ...]
) -> gtsam.Pose3:
  """Transform every tracker's body poses, velocities, and landmarks into the
  shared world rooted at the reference anchor, connecting trackers through any
  co-sighted anchor. Returns the reference anchor's shared-world pose."""
  locals_by_tracker = {t.name: t.anchor_locals_by_tag for t in trackers}
  aligns, world_T_anchor = _bfs_world_alignments(locals_by_tracker, anchor_tag_ids)
  for t in trackers:
    align = aligns[t.name]
    align_R = align.rotation().matrix()
    align_t = np.asarray(align.translation(), dtype=np.float64)
    t.body_poses = [align.compose(p) for p in t.body_poses]
    t.velocities = [align_R @ v for v in t.velocities]
    for ch in t.lenses:
      ch.points = {tid: align_R @ p + align_t for tid, p in ch.points.items()}
  return world_T_anchor[anchor_tag_ids[0]]


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
  base_tag_ids: list[int],
  base_tag_size: float,
) -> list[CrossSighting]:
  """Every sighting of one tracker's tags (base + fingers) in another tracker's
  frames, feeding the cross-tracker corner-reprojection factors. Captures are
  audio-synced, so observer and owner share the keyframe index. A PnP solve gates
  degenerate / behind-camera base-tag detections; the finger factors rely on the
  reprojection factor's own cheirality guard."""
  sightings: list[CrossSighting] = []
  for owner_idx, owner in enumerate(trackers):
    # The owner tags an observer may see: the base tag, plus each finger tag
    # that carries URDF joint kinematics.
    targets: dict[int, int | None] = {base_tag_ids[owner_idx]: None}
    finger_ids = (owner.left_finger_tag_id, owner.right_finger_tag_id)
    for idx, (name, finger_id) in enumerate(
      zip(FINGER_JOINTS, finger_ids, strict=True)
    ):
      if finger_id > 0 and name in owner.finger_reproj:
        targets[finger_id] = idx
    for observer in trackers:
      if observer is owner:
        continue
      n = min(observer.n_keyframes, owner.n_keyframes)
      for ch in observer.lenses:
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


def anchor_sightings(
  tracker: PreparedTracker,
  anchor_tag_id: int,
  max_half_angle: float,
) -> list[tuple[LensChannel, int, TagDetection]]:
  """Per ``(lens channel, keyframe)`` anchor-tag detection within
  ``max_half_angle`` of the optical axis. The corner reprojection factors and
  the viz lines both consume this, so they cover exactly the same keyframes."""
  return [
    (ch, kf, det)
    for ch in tracker.lenses
    for kf, dets in enumerate(ch.tags)
    for det in dets
    if int(det.tag_id) == anchor_tag_id
    and tag_half_angle(det, ch.setup.params) <= max_half_angle
  ]


def _sighting_bearing(
  obs: dict[int, dict[str, gtsam.Pose3]],
  kf: int,
  side: str,
  det: TagDetection,
  lens: LensSetup,
) -> Float64Array:
  """Cheirality bearing: the PnP tag position when present, else the
  detection-center ray."""
  cam_T_tag = obs.get(kf, {}).get(side)
  if cam_T_tag is None:
    return tag_center_bearing(det, lens.params)
  bearing = np.asarray(cam_T_tag.translation(), dtype=np.float64)
  return bearing / np.linalg.norm(bearing)


def anchored_keyframes(tracker: PreparedTracker) -> set[int]:
  """Keyframes with a PnP sighting of any configured anchor; the smoothness
  priors treat these as anchored."""
  return set().union(*tracker.anchor_obs_by_tag.values())


def _add_corner_factors_for_anchor(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  corner_noise: gtsam.noiseModel.Base,
  tag_id: int,
  corners_world: Float64Array,
  pnp_bearings: bool,
  corner_keys: list[int] | None,
  max_half_angle: float,
  graph: gtsam.NonlinearFactorGraph,
) -> int:
  """Corner-reprojection factors for one anchor, every sighting by any tracker
  and lens. ``pnp_bearings`` uses the PnP tag position as the cheirality
  bearing (the reference anchor), else the detection-center ray.
  ``corner_keys`` makes the corners graph variables seeded at ``corners_world``
  instead of constants pinned there. Returns the sighting count."""
  n_sightings = 0
  for t in trackers:
    obs = t.anchor_obs_by_tag.get(tag_id, {}) if pnp_bearings else {}
    for ch, kf, det in anchor_sightings(t, tag_id, max_half_angle):
      n_sightings += 1
      pose_key = keys.pose_key(t.name, kf)
      bearing = _sighting_bearing(obs, kf, ch.side, det, ch.setup)
      for j, pixel in enumerate(det.corners):
        if corner_keys is not None:
          add_mei_factor(
            graph, pose_key, corner_keys[j], pixel, corner_noise, ch.setup, bearing
          )
        else:
          add_mei_world_point_factor(
            graph, pose_key, pixel, corner_noise, ch.setup, corners_world[j], bearing
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
  visual reprojection) per tracker."""
  for t in trackers:
    build_vi_graph(
      body_poses=t.body_poses,
      velocities=t.velocities,
      biases=t.biases,
      preintegrated=t.preintegrated,
      front=t.front,
      rear=t.rear,
      sigma=args.frontend.reproj_sigma,
      velocity_prior_sigma=args.solver.velocity_prior_sigma,
      graph=graph,
      values=values,
      pose_key_fn=partial(keys.pose_key, t.name),
      velocity_key_fn=partial(keys.velocity_key, t.name),
      bias_key_fn=partial(keys.bias_key, t.name),
      front_landmark_key_fn=partial(keys.front_landmark_key, t.name),
      rear_landmark_key_fn=partial(keys.rear_landmark_key, t.name),
    )


def _add_all_anchor_corner_factors(
  trackers: list[PreparedTracker],
  keys: TrackerKeys,
  args: Args,
  anchor_corners_world: Float64Array,
  secondary_anchor_world: dict[int, gtsam.Pose3],
  graph: gtsam.NonlinearFactorGraph,
  values: gtsam.Values,
) -> None:
  """Corner-reprojection factors for the reference anchor (PnP bearings) and
  every secondary anchor (detection-center bearings), pinning absolute scale
  and the shared world gauge.

  These counts are the reference's own sighting totals."""
  corner_noise = huber_pixel_noise(args.anchor.corner_reproj_sigma)
  n_ref = _add_corner_factors_for_anchor(
    trackers,
    keys,
    corner_noise,
    int(args.anchor.enabled_tag_ids[0]),
    anchor_corners_world,
    pnp_bearings=True,
    corner_keys=None,
    max_half_angle=args.anchor.max_half_angle,
    graph=graph,
  )
  print(f"[graph] anchor corner reprojection: {n_ref} sightings")
  sec_prior = gtsam.noiseModel.Isotropic.Sigma(
    3, args.anchor.secondary_corner_prior_sigma
  )
  for tag_id, world_T_tag in secondary_anchor_world.items():
    seed_corners = tag_corners_in_frame(world_T_tag, args.anchor.tag_size)
    corner_keys: list[int] | None = None
    if args.anchor.free_secondary:
      # The secondary's corners are graph variables held near the seed by a
      # loose prior.
      corner_keys = [keys.secondary_corner_key(tag_id, j) for j in range(4)]
      for key, xyz in zip(corner_keys, seed_corners, strict=True):
        values.insert(key, xyz)
        graph.add(gtsam.PriorFactorPoint3(key, xyz, sec_prior))
    n_sec = _add_corner_factors_for_anchor(
      trackers,
      keys,
      corner_noise,
      tag_id,
      seed_corners,
      pnp_bearings=False,
      corner_keys=corner_keys,
      max_half_angle=args.anchor.max_half_angle,
      graph=graph,
    )
    print(f"[graph] secondary anchor {tag_id} corner reprojection: {n_sec} sightings")


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
  finger_noise = huber_pixel_noise(_FINGER_REPROJ_SIGMA)
  base_tag_noise = huber_pixel_noise(_BASE_TAG_REPROJ_SIGMA)
  base_tag_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([_BASE_TAG_PRIOR_SIGMA_ROT] * 3 + [_BASE_TAG_PRIOR_SIGMA_TRANS] * 3)
  )
  sym_noise = gtsam.noiseModel.Isotropic.Sigma(1, _FINGER_SYMMETRY_SIGMA)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  finger_factor_count = 0
  base_tag_factor_count = 0
  for t, base_tag_id in zip(trackers, args.base_tag_ids, strict=True):
    reprojs = {
      idx: reproj
      for idx, joint in enumerate(FINGER_JOINTS)
      if (reproj := t.finger_reproj.get(joint)) is not None
    }
    if t.body_T_base_tag is None or (not reprojs and base_tag_id <= 0):
      continue
    bt_key = keys.base_tag_key(t.name)
    values.insert(bt_key, t.body_T_base_tag)
    graph.add(gtsam.PriorFactorPose3(bt_key, t.body_T_base_tag, base_tag_prior_noise))
    # Base-tag self-reprojection: the base tag's four metric corners reproject
    # through body_T_base_tag (unary) into this tracker's own rear lens.
    if base_tag_id > 0:
      for dets in t.rear.tags:
        det = next((d for d in dets if int(d.tag_id) == base_tag_id), None)
        if det is None:
          continue
        bearing = tag_center_bearing(det, t.rear.setup.params)
        for j in range(4):
          add_mei_base_tag_factor(
            graph,
            bt_key,
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
        for j in range(4):
          add_mei_finger_factor(
            graph,
            tkey,
            bt_key,
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
  its corners reproject through O's pose, W's pose, and W's shared base-tag
  variable (a ternary factor), coupling the two trajectories. Seeing W's finger
  tag adds W's joint angle θ (a quaternary factor), so the cross view also
  constrains the angle; that needs W's θ variable at the keyframe (from its own
  observation or the symmetry inference)."""
  base_tag_noise = huber_pixel_noise(_CROSS_REPROJ_SIGMA)
  base_tag_corners = tag_corners_local(args.gripper.tag_size)
  tracker_by_name = {t.name: t for t in trackers}
  base_count = finger_count = 0
  for sighting in cross_obs:
    if not values.exists(keys.base_tag_key(sighting.owner)):
      continue
    obs_key = keys.pose_key(sighting.observer, sighting.keyframe)
    owner_key = keys.pose_key(sighting.owner, sighting.keyframe)
    owner_bt_key = keys.base_tag_key(sighting.owner)
    if sighting.finger is None:
      for j in range(4):
        add_mei_cross_tag_factor(
          graph,
          obs_key,
          owner_key,
          owner_bt_key,
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
        add_mei_cross_finger_factor(
          graph,
          obs_key,
          owner_key,
          owner_bt_key,
          theta_key,
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
  anchor_corners_world: Float64Array,
  secondary_anchor_world: dict[int, gtsam.Pose3],
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
  """Build the joint factor graph: per-tracker VI subgraphs, anchor corner
  reprojection (reference + pinned secondaries), gripper base/finger factors,
  cross-tracker reprojection, and smoothness priors on unanchored tails."""
  graph = gtsam.NonlinearFactorGraph()
  values = gtsam.Values()
  _add_vi_subgraphs(trackers, keys, args, graph, values)
  _add_all_anchor_corner_factors(
    trackers, keys, args, anchor_corners_world, secondary_anchor_world, graph, values
  )
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
