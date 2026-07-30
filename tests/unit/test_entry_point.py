"""Import coverage for the CLI entry point.

``scripts/track.py`` lives outside the package, so no other test imports it and
a syntax or import error there survives the suite and only surfaces once a run
has already paid for decode and frontend.
"""

from __future__ import annotations

import importlib


def test_entry_point_imports() -> None:
  track = importlib.import_module("scripts.track")
  assert callable(track.main)
