"""Test bootstrap. Importing `marketcast` FIRST applies the Windows DLL / OpenMP fixes in marketcast/__init__.py
before any test module pulls in torch or runs a BLAS call (see that file for the two crashes this prevents)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))        # lets tests import the `synth` helper

import marketcast  # noqa: E402,F401
