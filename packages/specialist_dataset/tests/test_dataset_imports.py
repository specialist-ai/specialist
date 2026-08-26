"""Reading a dataset does not import the solve's stack."""

from __future__ import annotations

import subprocess
import sys

_FORBIDDEN = ("gtsam", "cv2", "torch", "av", "scipy", "specialist")


def test_importing_the_dataset_leaves_the_solve_behind() -> None:
  probe = (
    "import sys, specialist_dataset;"
    f"print([m for m in {_FORBIDDEN!r} if m in sys.modules])"
  )
  found = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=True
  )
  assert found.stdout.strip() == "[]", found.stdout
