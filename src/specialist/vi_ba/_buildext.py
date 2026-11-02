"""Compile the native MEI reprojection factor extension (``_mei_factor_cpp``).

The gtsam wheel ships ``libgtsam`` but no C++ headers, so this shallow-clones the
matching gtsam source (pinned to the wheel's version) once into a gitignored
build cache for its headers, then compiles ``cpp/mei_factor.cpp`` against them +
the vendored generated headers in ``cpp/gtsam_generated/``. Boost headers are
fetched into the same build cache at the release the gtsam wheel bundles, so the
only system prerequisites are a C++17 compiler and git.

Invoked at install time by ``hatch_build.py`` and re-runnable with ``uv run
build-mei-factor``, which needs a source checkout: the header cache and the C++
source both live in the repo. Idempotent and platform-aware (macOS arm64/x86,
Linux). pybind11 must be the 2.x series (internals v5) so the extension's type
registry matches the gtsam wheel.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

GTSAM_TAG = "4.3a0"
# The release the gtsam wheel bundles: other headers corrupt the heap at runtime.
BOOST_TAG = "1.87.0"


def _repo_root(start: Path) -> Path:
  for d in (start, *start.parents):
    if (d / "pyproject.toml").exists():
      return d
  raise SystemExit("could not locate repo root (pyproject.toml)")


def _run(cmd: list[str]) -> None:
  print("  $", " ".join(str(c) for c in cmd))
  subprocess.run(cmd, check=True)


def _gtsam_paths() -> tuple[Path, Path]:
  """Return ``(gtsam_package_dir, libgtsam_path)`` from the installed wheel."""
  import gtsam

  pkg = Path(gtsam.__file__).resolve().parent
  if sys.platform == "darwin":
    libs = sorted((pkg / ".dylibs").glob("libgtsam.*.dylib"))
  else:
    libs = sorted((pkg.parent / "gtsam.libs").glob("libgtsam*.so*")) or sorted(
      pkg.glob("libgtsam*.so*")
    )
  if not libs:
    raise SystemExit("could not locate libgtsam shared library next to the wheel")
  return pkg, libs[0]


def _ensure_headers(gtsam_src: Path) -> None:
  """Shallow-clone gtsam at the pinned tag for its C++ headers."""
  if (gtsam_src / "gtsam" / "geometry" / "Pose3.h").exists():
    return
  gtsam_src.parent.mkdir(parents=True, exist_ok=True)
  print(f"cloning gtsam {GTSAM_TAG} headers into {gtsam_src} ...")
  _run(
    [
      "git",
      "clone",
      "--depth",
      "1",
      "--branch",
      GTSAM_TAG,
      "--single-branch",
      "https://github.com/borglab/gtsam.git",
      str(gtsam_src),
    ]
  )


def _ensure_boost(boost_src: Path) -> None:
  """Fetch the pinned Boost release headers; ``boost/`` is all the build needs."""
  if (boost_src / "boost" / "concept_check.hpp").exists():
    return
  import io
  import tarfile
  import urllib.request

  plain = BOOST_TAG.replace(".", "_")
  url = f"https://archives.boost.io/release/{BOOST_TAG}/source/boost_{plain}.tar.gz"
  print(f"fetching boost {BOOST_TAG} headers into {boost_src} ...")
  boost_src.mkdir(parents=True, exist_ok=True)
  with urllib.request.urlopen(url) as resp:
    data = io.BytesIO(resp.read())
  prefix = f"boost_{plain}/boost/"
  with tarfile.open(fileobj=data, mode="r:gz") as tar:
    members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
    for m in members:
      m.name = m.name[len(f"boost_{plain}/") :]
    tar.extractall(boost_src, members=members, filter="data")


def _detect_gxx_abi_version(gtsam_pkg: Path) -> str | None:
  """Return the ``-fabi-version`` matching the gtsam wheel's pybind11 module.

  Cross-module pybind11 type sharing requires identical ``internals`` IDs, which
  embed the GCC ABI version (``cxxabiNNNN`` = ``__GXX_ABI_VERSION``), read here
  off the wheel's own module.
  """
  pat = re.compile(rb"pybind11_internals_v\d+_[a-z_]*cxxabi(\d+)")
  for so in gtsam_pkg.glob("*.so"):
    m = pat.search(so.read_bytes())
    if m:
      # __GXX_ABI_VERSION == 1000 + fabi-version (e.g. cxxabi1014 -> 14).
      return str(int(m.group(1)) - 1000)
  print(
    "  [warn] could not read the pybind11 ABI version from the gtsam wheel; "
    "omitting -fabi-version (cross-module type sharing may fail)"
  )
  return None


def _detect_cxx11_abi(libgtsam: Path) -> str:
  """Return ``"0"`` / ``"1"`` for ``-D_GLIBCXX_USE_CXX11_ABI`` matching the
  wheel, read off its ``NoiseModelFactor::print`` symbol. A mismatch yields
  undefined-symbol errors at import.
  """
  try:
    out = subprocess.run(
      ["nm", "-D", str(libgtsam)], capture_output=True, text=True, check=True
    ).stdout
  except Exception:
    print(
      f"  [warn] nm unavailable; assuming _GLIBCXX_USE_CXX11_ABI=0 ({libgtsam.name})"
    )
    return "0"  # manylinux convention
  for line in out.splitlines():
    if "NoiseModelFactor5print" in line:
      return "1" if "__cxx11" in line else "0"
  print(
    f"  [warn] NoiseModelFactor symbol not found in {libgtsam.name}; "
    "assuming _GLIBCXX_USE_CXX11_ABI=0"
  )
  return "0"


def _includes(gtsam_src: Path, generated: Path, boost_root: Path) -> list[str]:
  import numpy
  import pybind11

  return [
    pybind11.get_include(),
    sysconfig.get_path("include"),
    numpy.get_include(),
    str(gtsam_src),
    str(generated),
    str(gtsam_src / "gtsam" / "3rdparty" / "Eigen"),
    str(gtsam_src / "gtsam" / "3rdparty" / "metis" / "include"),
    str(boost_root),
  ]


def build(force: bool = False) -> Path:
  """Compile the extension into the package; return its path."""
  import pybind11  # noqa: PLC0415

  if sys.platform == "win32":
    raise SystemExit("Windows is not supported; build on macOS or Linux")
  if int(pybind11.__version__.split(".")[0]) != 2:
    required = importlib.metadata.requires("specialist") or []
    pin = next((r for r in required if r.startswith("pybind11")), "pybind11==2.x")
    raise SystemExit(
      f"pybind11 {pybind11.__version__} found; need the 2.x series (internals v5) "
      f"to match the gtsam wheel — `uv pip install '{pin}'`"
    )
  # A wheel that does not match the pinned headers compiles cleanly and breaks
  # the ABI at runtime.
  gtsam_version = importlib.metadata.version("gtsam")
  if gtsam_version != GTSAM_TAG:
    raise SystemExit(
      f"installed gtsam {gtsam_version} != pinned headers {GTSAM_TAG}; bump "
      f"GTSAM_TAG and re-vendor cpp/gtsam_generated/ from the matching source"
    )
  viba_pkg = Path(__file__).resolve().parent
  repo = _repo_root(viba_pkg)
  src = viba_pkg / "cpp" / "mei_factor.cpp"
  generated = viba_pkg / "cpp" / "gtsam_generated"
  gtsam_src = repo / ".build" / f"gtsam-{GTSAM_TAG}"
  out = viba_pkg / ("_mei_factor_cpp" + str(sysconfig.get_config_var("EXT_SUFFIX")))
  inputs = (src, Path(__file__), *generated.rglob("*.h"))
  newest_input = max(p.stat().st_mtime for p in inputs)
  if out.exists() and not force and out.stat().st_mtime >= newest_input:
    print(f"{out.name} is up to date")
    return out

  gtsam_pkg, libgtsam = _gtsam_paths()
  _ensure_headers(gtsam_src)
  boost_root = repo / ".build" / f"boost-{BOOST_TAG}"
  _ensure_boost(boost_root)
  cxx = os.environ.get("CXX", "c++")
  # No fused multiply-add: the arithmetic stays numpy's.
  cmd = [cxx, "-O3", "-ffp-contract=off", "-shared", "-std=c++17", "-fPIC"]
  if sys.platform == "darwin":
    cmd += ["-stdlib=libc++", "-undefined", "dynamic_lookup"]
  else:
    cmd += [f"-D_GLIBCXX_USE_CXX11_ABI={_detect_cxx11_abi(libgtsam)}"]
    abi_version = _detect_gxx_abi_version(gtsam_pkg)
    if abi_version is not None:
      cmd += [f"-fabi-version={abi_version}"]
  for inc in _includes(gtsam_src, generated, boost_root):
    cmd += ["-I", inc]
  cmd += [str(src), "-o", str(out), str(libgtsam)]
  # Resolve libgtsam at runtime relative to the installed extension, so the
  # binary does not depend on the build environment. macOS needs no rpath: the
  # wheel's libgtsam has an absolute install name and `import gtsam` has already
  # loaded it, so dyld resolves it by leaf.
  if sys.platform != "darwin":
    cmd += [
      "-Wl,-rpath,$ORIGIN/../../gtsam.libs",
      "-Wl,-rpath,$ORIGIN/../../gtsam",
      "-Wl,--no-as-needed",
    ]
  print(f"compiling {src.name} -> {out.name}")
  _run(cmd)
  return out


def main() -> None:
  sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
  build(force=True)
  subprocess.run(
    [
      sys.executable,
      "-c",
      "import gtsam, specialist.vi_ba._mei_factor_cpp; print('import OK')",
    ],
    check=True,
  )


if __name__ == "__main__":
  main()
