"""density_torch's compiled cores stay correct when something in the process
has left RTLD_GLOBAL in the dlopen flags.

Every inductor CPU kernel library exports a function named `kernel` and calls
it through the PLT, so with RTLD_GLOBAL every kernel library after the first
calls the first one's code: silently wrong densities, then a segfault in the
scatter. lunus's own import no longer leaves the flag set, but cctbx's
boost_adaptbx leaves it whenever an extension fails to import, so the compiled
cores clear it while they run (density_torch._kernels_load_locally). Run in a
fresh interpreter that sets the flag before importing anything, as such a leak
would; compiles a few kernels, so it takes some seconds. The second test
loads a foreign compiled kernel globally first, which no dlopen flag can
undo, and checks that the cores fall back to eager instead
(density_torch._kernel_symbol_is_global).
"""

import os
import subprocess
import sys

import pytest

import lunus

torch = pytest.importorskip("torch", reason="PyTorch not installed")

CHILD = r"""
import sys, tempfile, os, ctypes
sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)
CACHE = tempfile.mkdtemp(prefix="lunus_rtld_")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
DEV = "@DEV@"
LATE_FOREIGN_AT = int("@LATE@")
import warnings
import numpy as np, torch
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
    orth_matrix_np=M_np, cutoff=1e-5, device=DEV, dtype=D)
hkl = torch.tensor([[1, 0, 0], [0, 2, 0], [1, -3, 2], [3, 3, 3]], device=DEV)
vol = abs(np.linalg.det(M_np))

def run(n, seed):
    g = torch.Generator().manual_seed(seed)
    frac = torch.rand(n, 3, generator=g, dtype=D).to(DEV).requires_grad_(True)
    idx = torch.randint(0, 3, (n,), generator=g).to(DEV)
    F = structure_factors_one_config(
        frac, idx, torch.ones(n, dtype=D, device=DEV), A[idx], lam[idx], offs,
        rad[idx], grid, torch.tensor(M_np, dtype=D, device=DEV), vol, hkl, tw,
        blur=0.0)
    F.abs().sum().backward()
    return F.detach().cpu(), frac.grad.cpu()

def kernel_libs():
    # inductor kernel libraries mapped into this process (Linux)
    try:
        return len({l.split()[-1] for l in open("/proc/self/maps")
                    if CACHE in l and l.rstrip().endswith(".so")})
    except OSError:
        return -1

# The 7th call (7 atoms, seed 6) recompiles AFTER both cores are proven, on
# this torch build, so it exercises the proven path's guard too.
SIZES = (50, 50, 50, 50, 120, 50, 7)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    compiled = [run(SIZES[0], 0)]
    libs_first = kernel_libs()
    for s, n in enumerate(SIZES):
        if s == 0:
            continue
        if s == LATE_FOREIGN_AT:
            # other torch.compile code loads its kernel globally AFTER lunus's
            # cores are proven (test_density_falls_back_after_its_cores_are_proven)
            late = torch.compile(lambda x: torch.exp(x) * 2.0 + 1.0, dynamic=True)
            late(torch.ones(5, dtype=torch.float64))
        compiled.append(run(n, s))
    libs_end = kernel_libs()
proven = len(dt._COMPILE_PROVEN)
kernel_global = hasattr(ctypes.CDLL(None), "kernel")
dt._COMPILE_FAILED = True                     # the eager path, as the reference
eager = [run(n, s) for s, n in enumerate(SIZES)]
err_F = max(float((c[0] - e[0]).abs().max()) for c, e in zip(compiled, eager))
err_g = max(float((c[1] - e[1]).abs().max()) for c, e in zip(compiled, eager))
warned = any("RTLD_GLOBAL is set" in str(w.message) for w in caught)
fell_back = any("already been loaded" in str(w.message) for w in caught)
still_global = bool(sys.getdlopenflags() & os.RTLD_GLOBAL)
print(proven, err_F, err_g, warned, still_global, fell_back, libs_first, libs_end,
      kernel_global)
import shutil
shutil.rmtree(CACHE, ignore_errors=True)
"""

# Some other torch.compile code runs first and loads its kernel globally. The
# linker searches the global scope before a library's own, so that `kernel`
# captures lunus's kernels however they are loaded (a segfault, measured);
# density_torch must see it and fall back to the eager cores.
FOREIGN = """
foreign = torch.compile(lambda x: torch.exp(x) * 2.0 + 1.0, dynamic=True)
try:
    foreign(torch.ones(5, dtype=torch.float64))
except Exception:                 # no compiler backend: nothing was loaded
    pass
import ctypes
if not hasattr(ctypes.CDLL(None), "kernel"):
    print("no-foreign-kernel")
    import shutil
    shutil.rmtree(CACHE, ignore_errors=True)
    raise SystemExit(0)
"""


def run_child(code, device="cpu", late_foreign_at=-1):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(lunus.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    out = subprocess.run([sys.executable, "-c", code.replace("@DEV@", device).replace("@LATE@", str(late_foreign_at))],
                         capture_output=True, text=True, env=env, timeout=900)
    assert out.returncode == 0, (
        "child exited %d (a segfault is -11): compiled kernels loaded with "
        "RTLD_GLOBAL\n%s" % (out.returncode, out.stderr[-2000:]))
    fields = out.stdout.splitlines()[-1].split()
    if fields == ["no-foreign-kernel"]:
        return None
    names = ("proven", "err_F", "err_g", "warned", "still_global", "fell_back",
             "libs_first", "libs_end", "kernel_global")
    return dict(zip(names, fields))


needs_dlopen = pytest.mark.skipif(not hasattr(sys, "getdlopenflags"),
                                  reason="no dlopen flags on this platform")


@pytest.fixture(scope="module")
def preset_cpu():
    return run_child(CHILD)


@needs_dlopen
def test_compiled_density_is_right_with_rtld_global_set(preset_cpu):
    r = preset_cpu
    assert r["fell_back"] == "False", (
        "lunus dropped its own compiled cores with only RTLD_GLOBAL set: a "
        "false positive, or one of its kernels loaded globally")
    if int(r["proven"]) == 0:
        pytest.skip("torch.compile did not run here (no compiler backend); "
                    "the eager fallback cannot show the defect")
    assert float(r["err_F"]) < 1e-10 and float(r["err_g"]) < 1e-8, (
        "compiled density differs from eager with RTLD_GLOBAL set: "
        "max |dF| %s, max |dgrad| %s" % (r["err_F"], r["err_g"]))
    assert r["warned"] == "True", "no warning that RTLD_GLOBAL was cleared"
    assert r["still_global"] == "True", "the caller's RTLD_GLOBAL was not put back"
    assert r["kernel_global"] == "False", "a lunus kernel library was loaded globally"


@needs_dlopen
def test_recompile_after_the_cores_are_proven_is_guarded_too(preset_cpu):
    """The same run, read for the proven path: kernel libraries loaded after
    the first call (a recompile, through _COMPILE_PROVEN's wrapper) must be
    loaded locally as well. Without that guard the run above segfaults at the
    recompile; this checks the recompile happened, so that test means it."""
    r = preset_cpu
    if int(r["proven"]) == 0 or int(r["libs_first"]) < 0:
        pytest.skip("no compiled cores, or no /proc/self/maps, here")
    if int(r["libs_end"]) <= int(r["libs_first"]):
        pytest.skip("this torch build did not recompile after the first call, "
                    "so the proven path loaded no kernel library")
    assert r["kernel_global"] == "False"


@needs_dlopen
def test_density_falls_back_when_a_kernel_is_already_global():
    marker = "import lunus.sf.density_torch as dt"
    r = run_child(CHILD.replace(marker, FOREIGN + marker))
    if r is None:
        pytest.skip("torch.compile did not load a CPU kernel here")
    assert r["fell_back"] == "True", "no warning that the compiled cores were dropped"
    assert int(r["proven"]) == 0, "compiled cores ran with a global `kernel` loaded"
    assert float(r["err_F"]) < 1e-12 and float(r["err_g"]) < 1e-12


@needs_dlopen
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_cores_stay_compiled_with_a_global_cpu_kernel():
    """CUDA cores load no CPU kernel library, so a foreign CPU `kernel` in
    the global scope must not send them to eager (it would cost the compiled
    speed-up on the GPU for nothing)."""
    marker = "import lunus.sf.density_torch as dt"
    r = run_child(CHILD.replace(marker, FOREIGN + marker), device="cuda")
    if r is None:
        pytest.skip("torch.compile did not load a CPU kernel here")
    assert r["fell_back"] == "False", "a global CPU kernel disabled the CUDA cores"
    assert int(r["proven"]) == 2, "the CUDA cores did not run compiled"
    assert float(r["err_F"]) < 1e-10 and float(r["err_g"]) < 1e-8


@needs_dlopen
def test_density_falls_back_after_its_cores_are_proven():
    """The likelier order: lunus's CPU cores compile and are proven first,
    then other torch.compile code loads a kernel globally (the window puts
    the caller's RTLD_GLOBAL back after every call), then lunus runs again.
    The proven path must see the global `kernel` and go eager, or the next
    kernel it loads binds to the foreign one (a segfault, measured)."""
    r = run_child(CHILD, late_foreign_at=4)
    if int(r["proven"]) == 0:
        pytest.skip("torch.compile did not run here (no compiler backend)")
    assert r["fell_back"] == "True", "no fallback after a foreign global kernel"
    assert int(r["proven"]) == 2, "the cores were not proven before the foreign kernel"
    assert float(r["err_F"]) < 1e-10 and float(r["err_g"]) < 1e-8, (
        "results differ from eager: max |dF| %s, max |dgrad| %s" % (r["err_F"], r["err_g"]))
