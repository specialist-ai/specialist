"""Run one synthetic scenario: optimize, write .rrd, write JSON summary.

Example usage::

  uv run python examples/run_synthetic.py
  uv run python examples/run_synthetic.py --scenario.seed 7 --scenario.duration-s 20
  uv run python examples/run_synthetic.py --spawn-viewer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from specialist.config import ScenarioConfig
from specialist.eval.crb import crb_trajectory_bounds
from specialist.eval.metrics import absolute_trajectory_error, relative_pose_error
from specialist.runner import run_scenario
from specialist.viz import log_scenario_run


@dataclass
class RunArgs:
  """CLI arguments for the synthetic runner."""

  scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
  output_rrd: Path = Path("logs/specialist_run.rrd")
  output_json: Path = Path("logs/specialist_run.json")
  spawn_viewer: bool = False


def _summary(args: RunArgs) -> dict[str, object]:
  run = run_scenario(args.scenario)
  keys_per_tracker = {
    name: [run.key_map.tracker_pose_key(name, i) for i in range(run.n_keyframes)]
    for name in args.scenario.tracker_names
  }
  crb = crb_trajectory_bounds(run.graph, run.ground_truth_values, keys_per_tracker)

  per_tracker: dict[str, dict[str, float]] = {}
  for name in args.scenario.tracker_names:
    ate = absolute_trajectory_error(
      run.estimated_keyframes[name], run.ground_truth_keyframes[name]
    )
    per_tracker[name] = {
      "ate_translation": ate.translation_rmse,
      "ate_rotation": ate.rotation_rmse,
      "crb_translation": crb[name].translation_rmse,
      "crb_rotation": crb[name].rotation_rmse,
    }

  rpe_pairs: dict[str, dict[str, float]] = {}
  names = list(args.scenario.tracker_names)
  for i in range(len(names)):
    for j in range(i + 1, len(names)):
      a, b = names[i], names[j]
      rpe = relative_pose_error(
        run.estimated_keyframes[a],
        run.estimated_keyframes[b],
        run.ground_truth_keyframes[a],
        run.ground_truth_keyframes[b],
      )
      rpe_pairs[f"{a}_vs_{b}"] = {
        "rpe_translation": rpe.translation_rmse,
        "rpe_rotation": rpe.rotation_rmse,
      }

  log_scenario_run(run, output_path=args.output_rrd, spawn=args.spawn_viewer)

  return {
    "seed": args.scenario.seed,
    "duration": args.scenario.duration,
    "rate": args.scenario.rate,
    "tracker_names": list(args.scenario.tracker_names),
    "n_keyframes": run.n_keyframes,
    "n_odometry_factors": len(run.odometry),
    "n_anchor_observations": len(run.anchor_observations),
    "n_cross_tracker_observations": len(run.cross_tracker_observations),
    "optimization": {
      "iterations": run.optimization.iterations,
      "initial_error": run.optimization.initial_error,
      "final_error": run.optimization.final_error,
    },
    "per_tracker": per_tracker,
    "rpe": rpe_pairs,
    "output_rrd": str(args.output_rrd),
  }


def main(args: RunArgs) -> None:
  args.output_rrd.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  summary = _summary(args)
  args.output_json.write_text(json.dumps(summary, indent=2))
  print(f"Wrote {args.output_rrd} and {args.output_json}")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main(tyro.cli(RunArgs))
