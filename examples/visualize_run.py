"""Open a saved .rrd file in the Rerun viewer.

Example::

  uv run python examples/visualize_run.py specialist_run.rrd
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import rerun as rr
import tyro


def main(path: Annotated[Path, tyro.conf.Positional]) -> None:
  """``path``: the .rrd file saved by run_synthetic.py."""
  if not path.exists():
    raise FileNotFoundError(f"no such .rrd file: {path}")
  rr.init("specialist", spawn=True)
  rr.log_file_from_path(path)


if __name__ == "__main__":
  tyro.cli(main)
