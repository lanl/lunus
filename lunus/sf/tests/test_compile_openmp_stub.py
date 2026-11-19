"""density_torch's compiled cores stay correct after cctbx's stub OpenMP
library is loaded.

cctbx built without OpenMP ships libomptbx.so, whose omp_get_num_threads()
and omp_get_thread_num() answer "one thread, number 0". scitbx.fftpack loads
it into the global scope (so do cctbx.miller, cctbx.xray and iotbx.pdb), and
on Linux an inductor CPU kernel loaded afterwards calls those stubs while its
threads come from torch's real runtime: every thread runs every loop in full.
One carbon gave F(000) = 23.99 for 5.998 on four threads, with no error.
density_torch must see the stubs and run its eager cores while torch has more
than one thread (density_torch._foreign_openmp_library), and keep the compiled
ones on a single thread, where the stubs' answers are right.

Run in a fresh interpreter, since what matters is what is loaded before the
first kernel; compiles a few kernels, so it takes some seconds. Before the
guard this was seen only as test_torch_pipeline.py failing when it ran after
a test that imports one of those cctbx modules.
"""

import os
import subprocess
import sys

import pytest

import lunus

torch = pytest.importorskip("torch", reason="PyTorch not installed")

CHILD = r"""
import sys, tempfile, os, shutil
CACHE = tempfile.mkdtemp(prefix="lunus_ompstub_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
THREADS = int("@THREADS@")
import warnings
import numpy as np, torch
torch.set_num_threads(THREADS)
try:
    import scitbx.fftpack
except ImportError:
    print("no-cctbx")
    shutil.rmtree(CACHE, ignore_errors=True)
    raise SystemExit(0)
import lunus.sf.density_torch as dt
from lunus.sf.cell_utils import orth_matrix, grid_shape_for_resolution
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.structure_factor_torch import structure_factors_one_config

D = torch.float64
cell = (30.0, 34.0, 26.0, 80.0, 95.0, 110.0)
M_np = orth_matrix(*cell)
grid = grid_shape_for_resolution(*cell[:3], 2.0, 1.5)
A, lam, offs, rad, tw, _ = build_element_kernels_torch(
    ["C", "N", "O"], IT92_COEFFS, 30.0, blur=0.0, grid_shape=grid,
    orth_matrix_np=M_np, cutoff=1e-5, device="cpu", dtype=D)
hkl = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, -3, 2], [3, 3, 3]])
vol = abs(np.linalg.det(M_np))

def run(n, seed, grad):
    g = torch.Generator().manual_seed(seed)
    frac = torch.rand(n, 3, generator=g, dtype=D).requires_grad_(grad)
    idx = torch.randint(0, 3, (n,), generator=g)
    F = structure_factors_one_config(
        frac, idx, torch.ones(n, dtype=D), A[idx], lam[idx], offs,
        rad[idx], grid, torch.tensor(M_np, dtype=D), vol, hkl, tw, blur=0.0)
    if not grad:
        return F.detach(), torch.zeros(n, 3, dtype=D)
    F.abs().sum().backward()
    return F.detach(), frac.grad

# with and without gradients: the two take different kernels, and both were wrong
CASES = ((50, 0, True), (50, 1, False), (1, 2, True), (1, 3, False))
def global_owner(name):
    # the library the global scope resolves a name to, found independently of
    # density_torch's own check; "" when the name is not global
    import ctypes
    scope = ctypes.CDLL(None)
    if not hasattr(scope, name) or not os.path.exists("/proc/self/maps"):
        return ""
    address = ctypes.cast(getattr(scope, name), ctypes.c_void_p).value
    for line in open("/proc/self/maps"):
        low, high = (int(x, 16) for x in line.split()[0].split("-"))
        if low <= address < high:
            return line.split()[-1]
    return ""

stub = global_owner("omp_get_thread_num")
if stub and stub == global_owner("GOMP_parallel"):
    stub = ""                                 # a real runtime, not a stub
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    compiled = [run(*case) for case in CASES]
proven = len(dt._COMPILE_PROVEN)
dt._COMPILE_FAILED = True                     # the eager path, as the reference
eager = [run(*case) for case in CASES]
err_F = max(float((c[0] - e[0]).abs().max()) for c, e in zip(compiled, eager))
err_g = max(float((c[1] - e[1]).abs().max()) for c, e in zip(compiled, eager))
warned = any("OpenMP thread queries" in str(w.message) for w in caught)
print(proven, err_F, err_g, warned, os.path.basename(stub) or "none")
shutil.rmtree(CACHE, ignore_errors=True)
"""


def run_child(threads):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(lunus.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.pop("OMP_NUM_THREADS", None)          # the child sets its own count
    out = subprocess.run([sys.executable, "-c", CHILD.replace("@THREADS@", str(threads))],
                         capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, "child exited %d\n%s" % (out.returncode, out.stderr[-2000:])
    fields = out.stdout.splitlines()[-1].split()
    if fields == ["no-cctbx"]:
        pytest.skip("cctbx (scitbx.fftpack) not installed")
    r = dict(zip(("proven", "err_F", "err_g", "warned", "stub"), fields))
    if r["stub"] == "none":
        pytest.skip("no stub OpenMP library in the global scope here (not "
                    "Linux, or a cctbx built with OpenMP)")
    return r


def test_density_is_right_on_several_threads_with_the_openmp_stub():
    r = run_child(threads=4)
    assert float(r["err_F"]) < 1e-10 and float(r["err_g"]) < 1e-8, (
        "density differs from eager after %s was loaded: max |dF| %s, "
        "max |dgrad| %s" % (r["stub"], r["err_F"], r["err_g"]))
    assert int(r["proven"]) == 0, "compiled CPU cores ran against the OpenMP stub"
    assert r["warned"] == "True", "no warning that the compiled cores were dropped"


def test_one_thread_keeps_the_compiled_cores_with_the_openmp_stub():
    r = run_child(threads=1)
    assert r["warned"] == "False", "the compiled cores were dropped on one thread"
    if int(r["proven"]) == 0:
        pytest.skip("torch.compile did not run here (no compiler backend)")
    assert float(r["err_F"]) < 1e-10 and float(r["err_g"]) < 1e-8, (
        "compiled density differs from eager on one thread: max |dF| %s, "
        "max |dgrad| %s" % (r["err_F"], r["err_g"]))
