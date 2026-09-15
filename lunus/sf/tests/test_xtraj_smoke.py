"""
Does xtraj still run? One tiny calculation per engine, and one optimization.

These are smoke tests, not accuracy tests. They drive the real command line on
the tracked two-frame example in examples/xtraj/, so they exercise everything
the unit tests cannot reach: argument parsing, the mdtraj read, each engine's
branch, the reductions and the file writes. A broken engine fails here; a
slightly wrong one does not, and is not meant to.

The cross-engine comparison runs at gemmi_cutoff=1e-4 rather than xtraj's
default of 1e-2. The cutoff is the density below which an atom's contribution
is dropped, and it applies to the torch engine as well as to gemmi; cctbx
builds its own grid (algorithm="fft" by default) with sampling it chooses
itself, and has no equivalent knob. Correlation of |F| against cctbx:

    cutoff     d_min 3.0, B=180        d_min 1.5, B=45
               all      high shell     all      high shell
    1e-2       0.9148     0.6478       0.9744     0.9682
    1e-4       0.9999     0.9997       0.9999     0.9993

NOTE THE B FACTORS, because they are an artefact of the fixture, not a
property of the cutoff: xtraj sets b_iso = 20*d_min^2, so the coarse d_min
this test uses for speed forces B=180. A large B flattens atoms and lowers
their peak density, so a FIXED ABSOLUTE cutoff discards more of them -- which
is why the penalty looks far worse at d_min 3.0 than at 1.5, and why it should
be milder still at the B of a real run. gemmi's own default is 1e-5.

1e-4 is used here because it makes the comparison mean something at either B,
and it is cheap: 1.47x the density calculation at d_min 3.0, 1.16x at 1.5.

sfall is excluded: it shells out to CCP4.

These are the slowest tests in the suite (a few seconds each, dominated by
interpreter start-up and the cctbx import). Run just them with

    python -m pytest lunus/sf/tests/test_xtraj_smoke.py -q
"""
import os
import subprocess
import sys

import numpy as np
import pytest

# tests/ -> sf/ -> lunus/ -> repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
XTRAJ = os.path.join(REPO_ROOT, "lunus", "command_line", "xtraj.py")
EXAMPLE = os.path.join(REPO_ROOT, "examples", "xtraj")
TOP = os.path.join(EXAMPLE, "top_ref.pdb")
TRAJ = os.path.join(EXAMPLE, "traj_ref.xtc")

# Two frames at low resolution: enough to exercise every branch, small enough
# to stay a smoke test. d_min is kept coarse but not so coarse that to_aniso()
# runs out of the four resolution shells a cubic spline needs. On gemmi_cutoff,
# see the module docstring -- the default of 1e-2 is too coarse to compare
# engines against each other.
COMMON = ["first=0", "last=1", "chunk=2", "d_min=3.0", "gemmi_cutoff=0.0001"]


def _missing():
    missing = []
    for module in ("cctbx", "mdtraj", "h5py", "gemmi"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if not os.path.isfile(TOP) or not os.path.isfile(TRAJ):
        missing.append("examples/xtraj data")
    return missing


pytestmark = pytest.mark.skipif(
    bool(_missing()), reason="xtraj needs %s" % ", ".join(_missing() or ["-"]))


def _run(tmp_path, *args):
    """Run xtraj in tmp_path; return its stdout, failing loudly if it does not."""
    command = [sys.executable, XTRAJ, "top=%s" % TOP, "traj=%s" % TRAJ] + list(args)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run(command, cwd=str(tmp_path), env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        pytest.fail("xtraj failed (%d)\n%s\n%s"
                    % (done.returncode, done.stdout[-4000:], done.stderr[-4000:]))
    return done.stdout


def _amplitudes(path):
    """|F| from an mtz, as numpy, indexed consistently between calls."""
    from iotbx import mtz

    array = mtz.object(str(path)).as_miller_arrays()[0]
    array = array.map_to_asu().sort(by_value="packed_indices")
    return np.abs(array.data().as_numpy_array())


ENGINES = ["cctbx", "gemmi", "torch"]


@pytest.fixture(scope="module")
def calculated(tmp_path_factory):
    """One run per engine, shared by the tests below."""
    results = {}
    for engine in ENGINES:
        if engine == "torch":
            try:
                __import__("torch")
            except ImportError:
                continue
        out = tmp_path_factory.mktemp(engine)
        args = COMMON + ["engine=%s" % engine,
                         "fcalc=fcalc.mtz", "icalc=icalc.mtz", "diffuse=diffuse.hkl"]
        if engine == "torch":
            args.append("torch_compile=False")
        _run(out, *args)
        results[engine] = out
    return results


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_runs_and_writes_its_outputs(calculated, engine):
    if engine not in calculated:
        pytest.skip("%s not available" % engine)
    out = calculated[engine]

    for name in ("fcalc.mtz", "icalc.mtz", "diffuse.hkl"):
        path = out / name
        assert path.is_file(), "%s did not write %s" % (engine, name)
        assert path.stat().st_size > 0, "%s wrote an empty %s" % (engine, name)

    amplitudes = _amplitudes(out / "fcalc.mtz")
    assert amplitudes.size > 100, "suspiciously few reflections"
    assert np.isfinite(amplitudes).all(), "%s produced non-finite F" % engine
    assert amplitudes.max() > 0.0, "%s produced an all-zero map" % engine


@pytest.mark.parametrize("engine", [e for e in ENGINES if e != "gemmi"])
def test_engines_agree_with_gemmi(calculated, engine):
    """At a cutoff where the comparison means something -- see the docstring."""
    if engine not in calculated or "gemmi" not in calculated:
        pytest.skip("need both %s and gemmi" % engine)

    reference = _amplitudes(calculated["gemmi"] / "fcalc.mtz")
    other = _amplitudes(calculated[engine] / "fcalc.mtz")
    assert other.shape == reference.shape, "%s produced a different Miller set" % engine

    correlation = np.corrcoef(reference, other)[0, 1]
    assert correlation > 0.99, "%s vs gemmi: correlation %.4f" % (engine, correlation)


def test_optimization_runs(tmp_path):
    """
    The do_opt path end to end, using the calculation's own diffuse map as the
    data. That makes the answer knowable: correlating a calculation with
    itself is 1, so the initial correlation must come back at 1 and the
    optimizer must find no frame worth dropping.
    """
    args = COMMON + ["engine=gemmi", "fcalc=fcalc.mtz", "icalc=icalc.mtz",
                     "diffuse=diffuse.mtz"]
    _run(tmp_path, *args)
    assert (tmp_path / "diffuse.mtz").is_file()

    log = _run(tmp_path, *(args + ["do_opt=True", "corr_aniso=True",
                                   "apply_symmetry=P1",
                                   "ID_file=%s" % (tmp_path / "diffuse.mtz")]))

    assert "Sweeping candidates in blocks of" in log, log[-2000:]
    assert (tmp_path / "selected_frames.ndx").is_file()
    assert (tmp_path / "deleted_frames.txt").is_file()

    line = [l for l in log.splitlines() if "Initial correlation after filtering" in l]
    assert line, log[-2000:]
    assert float(line[0].split("=")[1]) > 0.99, line[0]

    kept = [l for l in (tmp_path / "selected_frames.ndx").read_text().splitlines()
            if l.strip() and not l.startswith("[")]
    assert len(kept) == 2, "both frames should survive: %r" % kept
