"""MarketCast: probabilistic scenario engine for Indian equities.

Internal research code. It never outputs buy or sell calls (dev_plan/00_START_HERE.md rule 11).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "0.2.0"


def _fix_windows_dll_path() -> None:
    """Make conda's MKL loadable when python.exe is started WITHOUT `conda activate`.

    The .bat launchers (and any IDE) call <env>\\python.exe directly. NumPy then imports fine, but MKL loads
    its kernel DLLs (mkl_avx2, mkl_intel_thread, ...) lazily on the first BLAS call by searching PATH, and
    <env>\\Library\\bin is only on PATH after activation. The symptom is a silent native crash (exit code
    0xC06D007F, no traceback) at the first matrix product. Putting the folder on PATH here, before any
    linear algebra runs, removes that failure mode. Found the hard way on 2026-11-16.
    """
    if os.name != "nt":
        return
    # Second trap of the same family: conda's NumPy (MKL) and pip's PyTorch each ship their own Intel OpenMP
    # runtime (libiomp5md.dll). The first BLAS call made after both are loaded aborts the interpreter
    # ("OMP: Error #15", again without a Python traceback). Intel's documented switch lets the two coexist;
    # it must be set before the second runtime initialises, hence here, at package import.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    lib_bin = Path(sys.prefix) / "Library" / "bin"
    if lib_bin.is_dir() and str(lib_bin).lower() not in os.environ.get("PATH", "").lower():
        os.environ["PATH"] = f"{lib_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        try:
            os.add_dll_directory(str(lib_bin))
        except (OSError, AttributeError):
            pass


def _preload_torch_on_windows() -> None:
    """Third trap, same family: DLL load ORDER. Once conda's OpenMP runtime has been USED (an arch fit, a
    scikit-learn or LightGBM call), a later `import torch` fails with "WinError 127 ... shm.dll or one of its
    dependencies", because torch's DLLs bind to the already-initialised, different libiomp5md. Loading torch
    FIRST works in every order tried, so when torch is installed it is imported here, once, before anything
    else. It costs about two seconds per command; set MARKETCAST_NO_TORCH_PRELOAD=1 to skip it for pure data work.
    """
    if os.name != "nt" or os.environ.get("MARKETCAST_NO_TORCH_PRELOAD"):
        return
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: F401
        except OSError:       # a broken torch install must not stop data commands; training will report it properly
            pass


_fix_windows_dll_path()
_preload_torch_on_windows()

# Project root = the "Market Cast" folder (src/marketcast/__init__.py -> parents[2]).
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
PIT = DATA / "pit"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
