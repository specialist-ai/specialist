"""The standard dashboards, built from slots alone."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from specialist_dataset import schema

__all__ = [
  "TRAJECTORY_TRAIL_DURATION",
  "blueprint_for_slots",
  "segment_card_blueprint",
  "state_blueprint",
  "state_card_blueprint",
]

# Seconds the trajectory ribbon trails the cursor; segments and time window share it.
TRAJECTORY_TRAIL_DURATION = 1.0

# The per-keyframe series worth a plot, as the entity each camera carries it on.
_PLOTTED: tuple[tuple[str, Callable[[str], str]], ...] = (
  ("n_keypoints", lambda slot: schema.fit(slot, "n_keypoints")),
  ("rmse", lambda slot: schema.fit(slot, "rmse")),
  ("gyro_bias", lambda slot: schema.imu_bias(slot, "gyro")),
  ("accel_bias", lambda slot: schema.imu_bias(slot, "accel")),
  ("opening", lambda slot: schema.gripper_part(slot, "opening")),
)


# Keyframe-grid rate (Hz) of the time panel, the same at any capture rate.
_KEYFRAME_FPS = 10.0


def _panels() -> tuple[rrb.BlueprintPanel, rrb.SelectionPanel, rrb.TimePanel]:
  """The chrome every dashboard hides, and the clock they open on."""
  return (
    rrb.BlueprintPanel(state=rrb.PanelState.Collapsed),
    rrb.SelectionPanel(state=rrb.PanelState.Collapsed),
    rrb.TimePanel(
      state=rrb.PanelState.Collapsed,
      timeline=schema.TIMELINE,
      fps=_KEYFRAME_FPS,
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
) -> dict[rr.datatypes.EntityPath | str, Any]:
  """A triad per exported frame; an override lists all visualizers its entity keeps."""
  square = _anchor_square(tag_size)
  axes: dict[rr.datatypes.EntityPath | str, Any] = {
    f"/{schema.anchor(int(tag_id))}": [
      rr.TransformAxes3D(tag_size / 2.0),
      rr.LineStrips3D(strips=[square], colors=[_TAG_COLOR], radii=[0.003]),
    ]
    for tag_id in anchor_tag_ids
  }
  for slot in slots:
    axes[f"/{schema.camera(slot)}"] = rr.TransformAxes3D(BODY_AXIS)
    axes[f"/{schema.gripper(slot)}"] = rr.TransformAxes3D(BODY_AXIS)
    axes[f"/{schema.base_tag(slot)}"] = rr.TransformAxes3D(TAG_AXIS)
    for side in schema.SIDES:
      axes[f"/{schema.lens(slot, side)}"] = rr.TransformAxes3D(_LENS_AXIS)
    axes[f"/{schema.lens_seat(slot, 'rear')}"] = rr.TransformAxes3D(_LENS_AXIS)
    for finger in ("left_finger", "right_finger"):
      axes[f"/{schema.gripper_part(slot, finger)}"] = rr.TransformAxes3D(TAG_AXIS)
      axes[f"/{schema.joint_tag(slot, finger)}"] = rr.TransformAxes3D(TAG_AXIS)
  return axes


def _trail_overrides(
  slots: list[str],
) -> dict[rr.datatypes.EntityPath | str, rrb.VisibleTimeRanges]:
  """The window every view that draws a trajectory ribbon has to set."""
  return {f"/{schema.trajectory(s)}": _trail_window() for s in slots}


_HIDDEN = (
  (schema.detection, schema.KEYPOINTS_DETECTION),
  (schema.detection, schema.REJECTED_KEYPOINTS_DETECTION),
  (schema.overlay, schema.LANDMARK_REPROJECTION_OVERLAY),
)


def _hidden_overlay_overrides(
  slot: str, lens: str
) -> dict[rr.datatypes.EntityPath | str, rrb.EntityBehavior]:
  """Hide the front lens's dense pixel layers at start; other lenses have none."""
  if lens != "front":
    return {}
  return {
    f"/{path(slot, lens, kind)}": rrb.EntityBehavior(visible=False)
    for path, kind in _HIDDEN
  }


def _without_pixels(slots: list[str]) -> list[str]:
  """3D-view exclusions: lens pixel subtrees and hands' images, never a lens."""
  return [
    f"- /{schema.lens(slot, lens)}/{tail}"
    for slot in slots
    for lens in schema.SIDES
    for tail in ("video", "video/**", "overlay/**", "detection/**")
  ] + [f"- /{schema.state_image(hand)}" for hand in schema.HANDS]


def _scene(
  slots: list[str],
  anchor_tag_ids: Iterable[int] = (),
  anchor_tag_size: float = 0.1,
  overrides: dict[rr.datatypes.EntityPath | str, Any] | None = None,
) -> rrb.Spatial3DView:
  """The shared 3D view of triads, anchors and ribbons; ``overrides`` add to these."""
  hidden = rrb.EntityBehavior(visible=False)
  all_overrides: dict[rr.datatypes.EntityPath | str, Any] = {**_trail_overrides(slots)}
  all_overrides.update(_triad_overrides(slots, anchor_tag_ids, anchor_tag_size))
  all_overrides.update(
    {f"/{schema.landmark(s, lens)}": hidden for s in slots for lens in schema.SIDES}
  )
  all_overrides.update(overrides or {})
  # Rerun matches a wildcard only as a trailing /**, so every path is literal.
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
  """A 3D view and per-series plots beside one image column per camera."""
  image_columns = [
    rrb.Vertical(
      *(
        rrb.Spatial2DView(
          name=f"{slot}/{lens}",
          origin=f"/{schema.lens(slot, lens)}",
          overrides=_hidden_overlay_overrides(slot, lens),
        )
        for lens in schema.SIDES
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
  ("opening", schema.state_opening),
  ("lin_vel", schema.state_lin_vel),
  ("ang_vel", schema.state_ang_vel),
)


def state_blueprint(
  slots: Sequence[str] = (),
  anchor_tag_ids: Iterable[int] = (),
  anchor_tag_size: float = 0.1,
) -> rrb.Blueprint:
  """Both hands' images beside a 3D view of hand triads and per-series state plots."""
  return rrb.Blueprint(
    rrb.Horizontal(
      rrb.Tabs(
        _scene(list(slots), anchor_tag_ids, anchor_tag_size, _state_triads()),
        rrb.Grid(
          *(
            rrb.TimeSeriesView(
              name=metric,
              origin="/",
              contents=[f"+ /{entity(hand)}" for hand in schema.HANDS],
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


def _state_triads() -> dict[rr.datatypes.EntityPath | str, Any]:
  return {
    f"/{schema.state_pose(hand)}": rr.TransformAxes3D(BODY_AXIS)
    for hand in schema.HANDS
  }


def _hands_column() -> rrb.Vertical:
  return rrb.Vertical(
    *(
      rrb.Spatial2DView(name=hand, origin=f"/{schema.state_image(hand)}")
      for hand in schema.HANDS
    ),
    name="state",
  )


def segment_card_blueprint(slots: list[str]) -> rrb.Blueprint:
  """Segment-table card whose 3D view drops pixels, landmark clouds and factor lines."""
  contents = ["+ /**", *_without_pixels(slots)]
  for slot in slots:
    for lens in schema.SIDES:
      contents.append(f"- /{schema.landmark(slot, lens)}/**")
    for kind in ("anchor", "cross_camera"):
      contents.append(f"- /{schema.factor(slot, kind)}/**")
  return rrb.Blueprint(
    rrb.Spatial3DView(
      name="3D", origin="/", contents=contents, overrides=_trail_overrides(slots)
    ),
    *_panels(),
  )


def state_card_blueprint(frame_width: int) -> rrb.Blueprint:
  """Segment-table card of a state catalog dataset: both hands' images side by side."""
  w = float(frame_width)
  return rrb.Blueprint(
    rrb.Spatial2DView(
      name="state",
      origin=f"/{schema.STATE}",
      contents=[f"+ /{schema.state_image(h)}" for h in schema.HANDS],
      visual_bounds=rrb.VisualBounds2D(x_range=(0.0, 2.0 * w), y_range=(0.0, w)),
    ),
    *_panels(),
  )


def _trail_window() -> rrb.VisibleTimeRanges:
  """The time window a trajectory ribbon draws; a view without it draws one step."""
  return rrb.VisibleTimeRanges(
    [
      rrb.VisibleTimeRange(
        schema.TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-TRAJECTORY_TRAIL_DURATION),
        end=rrb.TimeRangeBoundary.cursor_relative(),
      )
    ]
  )
