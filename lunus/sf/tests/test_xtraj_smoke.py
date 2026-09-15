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

    cutoff    d_min 4.0, B=320     d_min 3.0, B=180     d_min 1.5, B=45
              all    high shell    all    high shell    all    high shell
    1e-2     0.8822    0.6427     0.9148    0.6478     0.9744    0.9682
    1e-4     0.9998    0.9997     0.9999    0.9997     0.9999    0.9993

The first column is what this test runs. NOTE THE B FACTORS: they are an
artefact of the fixture, not a property of the cutoff. xtraj sets
b_iso = 20*d_min^2 when it is not reading B from the model, so the coarse
d_min these tests use for speed forces a very large one. A large B flattens
atoms and lowers their peak density, so a FIXED ABSOLUTE cutoff discards more
of them -- which is the whole trend above, and why the penalty should be
milder still at the B of a real run. gemmi's own default is 1e-5;
docs/solvent-design.md measures what the choice costs in R factors.

1e-4 is used here because it makes the comparison mean something at any of
those B, and it is cheap: 1.47x the density calculation at d_min 3.0, 1.16x
at 1.5.

sfall is excluded: it shells out to CCP4.

These are the slowest tests in the suite (a few seconds each, dominated by
interpreter start-up and the cctbx import). Run just them with

    python -m pytest lunus/sf/tests/test_xtraj_smoke.py -q
"""
import functools
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
# to stay a smoke test. Every second here is d_min: 4.0 gives 143k reflections
# against 340k at 3.0, and the whole file runs in 12 s rather than 25. It must
# not go so coarse that to_aniso() runs out of the four resolution shells a
# cubic spline needs. On gemmi_cutoff see the module docstring -- the default
# of 1e-2 is too coarse to compare engines against each other.
COMMON = ["first=0", "last=1", "chunk=2", "d_min=4.0", "gemmi_cutoff=0.0001"]


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


@functools.lru_cache(maxsize=None)
def _amplitudes(path):
    """|F| from an mtz, as numpy, indexed consistently between calls.

    Cached: the comparisons below read the same few files repeatedly, and at
    this size the read plus map_to_asu plus sort costs more than the
    calculation that produced them.
    """
    from iotbx import mtz

    array = mtz.object(str(path)).as_miller_arrays()[0]
    array = array.map_to_asu().sort(by_value="packed_indices")
    return np.abs(array.data().as_numpy_array())


ENGINES = ["cctbx", "gemmi", "torch"]

# gemmi writes an mtz so test_optimization_runs can feed it back rather than
# recalculating; the other two keep the plain-text .hkl writer covered.
DIFFUSE = {"cctbx": "diffuse.hkl", "gemmi": "diffuse.mtz", "torch": "diffuse.hkl"}


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
                         "fcalc=fcalc.mtz", "icalc=icalc.mtz",
                         "diffuse=%s" % DIFFUSE[engine]]
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

    for name in ("fcalc.mtz", "icalc.mtz", DIFFUSE[engine]):
        path = out / name
        assert path.is_file(), "%s did not write %s" % (engine, name)
        assert path.stat().st_size > 0, "%s wrote an empty %s" % (engine, name)

    amplitudes = _amplitudes(str(out / "fcalc.mtz"))
    assert amplitudes.size > 100, "suspiciously few reflections"
    assert np.isfinite(amplitudes).all(), "%s produced non-finite F" % engine
    assert amplitudes.max() > 0.0, "%s produced an all-zero map" % engine


@pytest.mark.parametrize("engine", [e for e in ENGINES if e != "gemmi"])
def test_engines_agree_with_gemmi(calculated, engine):
    """At a cutoff where the comparison means something -- see the docstring."""
    if engine not in calculated or "gemmi" not in calculated:
        pytest.skip("need both %s and gemmi" % engine)

    reference = _amplitudes(str(calculated["gemmi"] / "fcalc.mtz"))
    other = _amplitudes(str(calculated[engine] / "fcalc.mtz"))
    assert other.shape == reference.shape, "%s produced a different Miller set" % engine

    correlation = np.corrcoef(reference, other)[0, 1]
    assert correlation > 0.99, "%s vs gemmi: correlation %.4f" % (engine, correlation)


def _shuffled_subset(source, destination, fraction=0.7, seed=0):
    """Write a shuffled subset of an mtz, preserving each reflection's value.

    The point is a NON-DEGENERATE fixture. Handed the calculation's own map
    unchanged, the calculated and experimental reflection lists are identical,
    so common_set_selection()'s two match columns are the same permutation and
    a mapping error cannot show itself -- verified by mutation: swapping the
    columns leaves the whole-pipeline test passing. A shuffled subset makes the
    columns genuinely different while leaving every value attached to the
    reflection it belongs to, so the self-correlation is still 1 if and only if
    the mapping is right.
    """
    from cctbx.array_family import flex
    from iotbx import mtz

    array = mtz.object(str(source)).as_miller_arrays()[0]
    rng = np.random.default_rng(seed)
    n = array.indices().size()
    keep = rng.permutation(n)[: int(fraction * n)]
    array = array.select(flex.size_t(keep.astype(np.uint64)))
    array.as_mtz_dataset("ID").mtz_object().write(file_name=str(destination))
    return array.indices().size()


def test_optimization_runs(calculated, tmp_path):
    """
    The do_opt path end to end, against a shuffled subset of the calculation's
    own diffuse map. That makes the answer knowable -- correlating a
    calculation with itself is 1, so the initial correlation must come back at
    1 and the optimizer must find no frame worth dropping -- while keeping the
    calculated-to-experimental mapping non-trivial. See _shuffled_subset.

    The forward calculation is the module fixture's gemmi run rather than a
    fresh one, which is why that engine writes an mtz.
    """
    if "gemmi" not in calculated:
        pytest.skip("gemmi not available")
    fraction = 0.7
    source = calculated["gemmi"] / "diffuse.mtz"
    n_expt = _shuffled_subset(source, tmp_path / "data.mtz", fraction=fraction)

    args = COMMON + ["engine=gemmi", "fcalc=fcalc.mtz", "icalc=icalc.mtz",
                     "diffuse=diffuse.mtz"]
    log = _run(tmp_path, *(args + ["do_opt=True", "corr_aniso=True",
                                   "apply_symmetry=P1",
                                   "ID_file=%s" % (tmp_path / "data.mtz")]))

    # The one intermediate between reading the data and the first correlation.
    # Asserting it localises a mapping failure here, rather than leaving it to
    # surface downstream as a correlation that is merely wrong.
    common = [l for l in log.splitlines() if l.startswith("do_opt: common set =")]
    assert common, log[-2000:]
    n_common = int(common[0].split("=")[1].split()[0])
    assert n_common == n_expt, ("every experimental reflection is a subset of "
                               "the calculated ones, so all %d should match, "
                               "not %d -- %s" % (n_expt, n_common, common[0]))

    assert "Sweeping candidates in blocks of" in log, log[-2000:]
    assert (tmp_path / "selected_frames.ndx").is_file()
    assert (tmp_path / "deleted_frames.txt").is_file()

    line = [l for l in log.splitlines() if "Initial correlation after filtering" in l]
    assert line, log[-2000:]
    assert float(line[0].split("=")[1]) > 0.99, line[0]

    kept = [l for l in (tmp_path / "selected_frames.ndx").read_text().splitlines()
            if l.strip() and not l.startswith("[")]
    assert len(kept) == 2, "both frames should survive: %r" % kept
