"""Shared pytest setup for the lunus/sf test suite.

Two jobs:

1. Put the REPOSITORY ROOT on sys.path so ``import lunus.sf.<module>``
   resolves when pytest is run from inside lunus/sf/ rather than from the
   repo root. A cctbx module build already arranges this, so the insertion
   is a no-op there.

2. Provide skip markers for the optional heavy dependencies. Nothing here
   needs all of them: the numpy reference tests need neither torch nor cctbx,
   and only the gemmi-parity checks need gemmi. A partial environment should
   skip, not error.
"""

import os
import sys

import pytest

# tests/ -> sf/ -> lunus/ -> repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _importable(name):
    try:
        __import__(name)
    except ImportError:
        return False
    return True


HAVE_TORCH = _importable("torch")
HAVE_CCTBX = _importable("cctbx")
HAVE_GEMMI = _importable("gemmi")
HAVE_MPI4PY = _importable("mpi4py")

requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="PyTorch not installed")
requires_cctbx = pytest.mark.skipif(not HAVE_CCTBX, reason="cctbx not installed")
requires_gemmi = pytest.mark.skipif(not HAVE_GEMMI, reason="gemmi not installed")
requires_mpi4py = pytest.mark.skipif(not HAVE_MPI4PY, reason="mpi4py not installed")


# Path to the scratch area holding the OUTPUTS of the end-to-end comparison.
# Those are gitignored; the inputs it runs on (top_bfac.pdb.gz, traj.xtc, the
# 7FPV files) are tracked in examples/compare_gemmi/. Produce the outputs with
# examples/compare_gemmi/run_xtraj.sh.
COMPARE_DATA_ENV = "LUNUS_SF_COMPARE_DATA"


@pytest.fixture
def compare_data_dir():
    path = os.environ.get(COMPARE_DATA_ENV)
    if not path or not os.path.isdir(path):
        pytest.skip(
            "set %s to a directory holding top_bfac.pdb and traj.xtc "
            "(see examples/compare_gemmi/run_xtraj.sh)" % COMPARE_DATA_ENV
        )
    return path
