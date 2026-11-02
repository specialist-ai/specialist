"""The standard dashboards, built from slots alone."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from specialist_corpus import schema as record_schema

__all__ = [
  "TRAJECTORY_TRAIL_DURATION",
  "blueprint_for_slots",
  "state_blueprint",
  "state_card_blueprint",
]

# Seconds the fading trajectory ribbon trails the cursor; the drawn segments
# and the visible-time window must agree on it.
TRAJECTORY_TRAIL_DURATION = 1.0

# The per-keyframe series worth a plot, as the entity each camera carries it on.
_PLOTTED: tuple[tuple[str, Callable[[str], str]], ...] = (
  ("n_keypoints", lambda slot: record_schema.fit(slot, "n_keypoints")),
  ("rmse", lambda slot: record_schema.fit(slot, "rmse")),
  ("gyro_bias", lambda slot: record_schema.imu_bias(slot, "gyro")),
  ("accel_bias", lambda slot: record_schema.imu_bias(slot, "accel")),
  ("opening", lambda slot: record_schema.gripper_part(slot, "opening")),
)


# The rate the keyframe grid steps at: ~0.10 s per bucket at any capture rate.
KEYFRAME_FPS = 10.0


def _panels() -> tuple[rrb.BlueprintPanel, rrb.SelectionPanel, rrb.TimePanel]:
  """The chrome every dashboard hides, and the clock they open on."""
  return (
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(
      state=rrb.PanelState.Collapsed,
      timeline=record_schema.TIMELINE,
      fps=KEYFRAME_FPS,
    ),
  )


_SPIN_SPEED = 0.25

_TAG_COLOR = (255, 210, 0)

BODY_AXIS = 0.05
_LENS_AXIS = 0.03
TAG_AXIS = 0.02


def _anchor_square(tag_size: float) -> np.ndarray:
  """The tag's outline in its own frame, closed."""
  half = tag_size / 2.0
  return np.array(
    [
      [+half, -half, 0.0],
      [+half, +half, 0.0],
      [-half, +half, 0.0],
      [-half, -half, 0.0],
      [+half, -half, 0.0],
    ],
    dtype=np.float64,
  )


def _triad_overrides(
  slots: list[str], anchor_tag_ids: Iterable[int], tag_size: float
) -> dict[rr.encodings.EntityPath | str, Any]:
  """A triad on every frame the solve exports. An override names every
  visualizer its entity keeps."""
  square = _anchor_square(tag_size)
  axes: dict[rr.encodings.EntityPath | str, Any] = {
    f"/{record_schema.anchor(int(tag_id))}": [
      rr.TransformAxes3D(tag_size / 2.0),
      rr.LineStrips3D(strips=[square], colors=[_TAG_COLOR], radii=[0.003]),
    ]
    for tag_id in anchor_tag_ids
  }
  for slot in slots:
    axes[f"/{record_schema.camera(slot)}"] = rr.TransformAxes3D(BODY_AXIS)
    axes[f"/{record_schema.gripper(slot)}"] = rr.TransformAxes3D(BODY_AXIS)
    axes[f"/{record_schema.base_tag(slot)}"] = rr.TransformAxes3D(TAG_AXIS)
    for side in record_schema.SIDES:
      axes[f"/{record_schema.lens(slot, side)}"] = rr.TransformAxes3D(_LENS_AXIS)
    axes[f"/{record_schema.lens_seat(slot, 'rear')}"] = rr.TransformAxes3D(_LENS_AXIS)
    for finger in ("left_finger", "right_finger"):
      axes[f"/{record_schema.gripper_part(slot, finger)}"] = rr.TransformAxes3D(
        TAG_AXIS
      )
      axes[f"/{record_schema.joint_tag(slot, finger)}"] = rr.TransformAxes3D(TAG_AXIS)
  return axes


def _trail_overrides(
  slots: list[str],
) -> dict[rr.encodings.EntityPath | str, rrb.VisibleTimeRanges]:
  """The window every view that draws a trajectory ribbon has to set."""
  return {f"/{record_schema.trajectory(s)}": _trail_window() for s in slots}


_HIDDEN = (
  (record_schema.detection, record_schema.KEYPOINTS_DETECTION),
  (record_schema.detection, record_schema.REJECTED_KEYPOINTS_DETECTION),
  (record_schema.overlay, record_schema.LANDMARK_REPROJECTION_OVERLAY),
)


def _hidden_overlay_overrides(
  slot: str, lens: str
) -> dict[rr.encodings.EntityPath | str, rrb.EntityBehavior]:
  """The dense pixel layers start hidden. Only the lens that builds landmarks
  has them."""
  if lens != "front":
    return {}
  return {
    f"/{path(slot, lens, kind)}": rrb.EntityBehavior(visible=False)
    for path, kind in _HIDDEN
  }


def _without_pixels(slots: list[str]) -> list[str]:
  """What a 3D view drops: each lens's frames and everything measured or drawn
  in them, and the hands' images, but never a lens itself.
  Excluding the lens subtree wholesale takes the lens with it."""
  return [
    f"- /{record_schema.lens(slot, lens)}/{tail}"
    for slot in slots
    for lens in record_schema.SIDES
    for tail in ("video", "video/**", "overlay/**", "detection/**")
  ] + [f"- /{record_schema.state_image(hand)}" for hand in record_schema.HANDS]


def _scene(
  slots: list[str],
  anchor_tag_ids: Iterable[int] = (),
  anchor_tag_size: float = 0.1,
  overrides: dict[rr.encodings.EntityPath | str, Any] | None = None,
) -> rrb.Spatial3DView:
  """The 3D view every dashboard shares: a triad on each solved frame, the
  anchors' outlines and the trajectory ribbons; the landmark clouds start
  hidden and no lens's pixels enter. ``overrides`` add to the standard ones.

  Every entity is named in full: a Rerun content pattern is a literal path
  optionally ending in ``/**``, and a wildcard in the middle matches nothing.
  """
  hidden = rrb.EntityBehavior(visible=False)
  all_overrides: dict[rr.encodings.EntityPath | str, Any] = {**_trail_overrides(slots)}
  all_overrides.update(_triad_overrides(slots, anchor_tag_ids, anchor_tag_size))
  all_overrides.update(
    {
      f"/{record_schema.landmark(s, lens)}": hidden
      for s in slots
      for lens in record_schema.SIDES
    }
  )
  all_overrides.update(overrides or {})
  return rrb.Spatial3DView(
    name="3D",
    origin="/",
    contents=["+ /**", *_without_pixels(slots)],
    overrides=all_overrides,
    eye_controls=rrb.EyeControls3D(spin_speed=_SPIN_SPEED),
  )


def blueprint_for_slots(
  slots: list[str],
  anchor_tag_ids: Iterable[int] = (),
  anchor_tag_size: float = 0.1,
) -> rrb.Blueprint:
  """Multi-camera blueprint: one image-pair column per camera, and one plot per
  diagnostic series with every camera's on it."""
  image_columns = [
    rrb.Vertical(
      *(
        rrb.Spatial2DView(
          name=f"{slot}/{lens}",
          origin=f"/{record_schema.lens(slot, lens)}",
          overrides=_hidden_overlay_overrides(slot, lens),
        )
        for lens in record_schema.SIDES
      ),
      name=slot,
    )
    for slot in slots
  ]
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Tabs(
        _scene(slots, anchor_tag_ids, anchor_tag_size),
        rrb.Grid(
          *(
            rrb.TimeSeriesView(
              name=metric,
              origin="/",
              contents=[f"+ /{entity(slot)}" for slot in slots],
            )
            for metric, entity in _PLOTTED
          ),
          name="plots",
        ),
      ),
      rrb.Horizontal(*image_columns) if len(slots) > 1 else image_columns[0],
      column_shares=[2, len(slots)],
    ),
    *_panels(),
  )


_STATE_PLOTTED: tuple[tuple[str, Callable[[str], str]], ...] = (
  ("opening", record_schema.state_opening),
  ("lin_vel", record_schema.state_lin_vel),
  ("ang_vel", record_schema.state_ang_vel),
)


def state_blueprint() -> rrb.Blueprint:
  """The state dashboard: the two hands' images stacked beside a 3D view with
  a triad on each hand's base-tag frame, and a plot per plotted state
  series."""
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Tabs(
        _scene([], overrides=_state_triads()),
        rrb.Grid(
          *(
            rrb.TimeSeriesView(
              name=metric,
              origin="/",
              contents=[f"+ /{entity(hand)}" for hand in record_schema.HANDS],
            )
            for metric, entity in _STATE_PLOTTED
          ),
          name="plots",
        ),
      ),
      _hands_column(),
      column_shares=[3, 1],
    ),
    *_panels(),
  )


def _state_triads() -> dict[rr.encodings.EntityPath | str, Any]:
  return {
    f"/{record_schema.state_pose(hand)}": rr.TransformAxes3D(BODY_AXIS)
    for hand in record_schema.HANDS
  }


def _hands_column() -> rrb.Vertical:
  return rrb.Vertical(
    *(
      rrb.Spatial2DView(name=hand, origin=f"/{record_schema.state_image(hand)}")
      for hand in record_schema.HANDS
    ),
    name="state",
  )


def state_card_blueprint(frame_width: int) -> rrb.TableBlueprint:
  """Segment-table cards of a state dataset: each session's id over both
  hands' images, side by side in the plane the layer places them in.

  The bounds are the pair itself. A view left to fit its own contents anchors
  them against one edge of whatever region it settles on.
  """
  w = float(frame_width)
  pair = rrb.Spatial2DView(
    name="state",
    origin=f"/{record_schema.STATE}",
    contents=[f"+ /{record_schema.state_image(h)}" for h in record_schema.HANDS],
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 2.0 * w), y_range=(0.0, w)),
  )
  preview = rrb.table.Column("recording link", column_type=rrb.table.PreviewCell(pair))
  return rrb.TableBlueprint(
    card_layout=rrb.table.CardLayout(
      title="rerun_segment_id", link="recording link", fields=[preview]
    )
  )


def _trail_window() -> rrb.VisibleTimeRanges:
  """The stretch of time a trajectory ribbon is drawn over. A view that does not
  set it draws a single step."""
  return rrb.VisibleTimeRanges(
    [
      rrb.VisibleTimeRange(
        record_schema.TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-TRAJECTORY_TRAIL_DURATION),
        end=rrb.TimeRangeBoundary.cursor_relative(),
      )
    ]
  )
