"""xtraj with a topology that holds alternate conformers.

mdtraj keeps only the first conformer of each atom and cctbx keeps them all,
so the trajectory's rows and xtraj's scatterers are then different atoms.
The cctbx engine stopped on an assertion; the torch engine ran to the end and
wrote structure factors from mismatched coordinates (an OG scattered as a
carbon). xtraj must refuse such a topology up front, on every engine, and
still run a one-conformer topology with a selection.
"""

import os
import subprocess
import sys

import pytest

from conftest import HAVE_TORCH, REPO_ROOT

pytest.importorskip("cctbx")
md = pytest.importorskip("mdtraj")
# xtraj imports these at module scope, so without them it never reaches the check.
pytest.importorskip("h5py")
pytest.importorskip("gemmi")

XTRAJ = os.path.join(REPO_ROOT, "lunus", "command_line", "xtraj.py")
HEADER = "CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 1\n"
N = "ATOM      1  N   SER A   1       2.000   3.000   4.000  1.00 20.00           N\n"
TWO_CONFORMERS = HEADER + N + """\
ATOM      2  CB ASER A   1       3.000   3.000   4.000  0.50 20.00           C
ATOM      3  CB BSER A   1       3.300   3.000   4.000  0.50 20.00           C
ATOM      4  OG ASER A   1       3.000   4.400   4.000  0.50 20.00           O
ATOM      5  OG BSER A   1       3.300   4.400   4.000  0.50 20.00           O
END
"""
ONE_CONFORMER = HEADER + N + """\
ATOM      2  CB  SER A   1       3.000   3.000   4.000  1.00 20.00           C
ATOM      3  OG  SER A   1       3.000   4.400   4.000  1.00 20.00           O
END
"""
ENGINES = ["cctbx"] + (["torch"] if HAVE_TORCH else [])


def run_xtraj(tmp_path, topology, engine, selection, top_name="top.pdb"):
    """Write topology as top.pdb and a two-frame trajectory of it, then run
    xtraj on the file top_name in tmp_path (top.pdb unless the caller wrote
    another, such as an mmCIF copy)."""
    top = tmp_path / "top.pdb"
    top.write_text(topology)
    t = md.load(str(top))
    md.join([t, t]).save_xtc(str(tmp_path / "traj.xtc"))
    args = [sys.executable, XTRAJ, "top=%s" % top_name, "traj=traj.xtc", "first=0",
            "last=1", "chunk=2", "d_min=4.0", "engine=%s" % engine,
            "selection=%s" % selection]
    if engine == "torch":
        args.append("torch_compile=False")
    env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               TORCH_COMPILE_DISABLE="1")
    return subprocess.run(args, cwd=str(tmp_path), env=env, capture_output=True, text=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_xtraj_refuses_a_topology_with_alternate_conformers(tmp_path, engine):
    done = run_xtraj(tmp_path, TWO_CONFORMERS, engine, "altloc ' ' or altloc A")
    assert done.returncode != 0, "%s ran on a topology with alternates" % engine
    out = done.stdout + done.stderr
    assert "has 5 atom records but mdtraj reads 3 atoms" in out, out[-2000:]
    assert "alternate conformers" in out


@pytest.mark.parametrize("engine", ENGINES)
def test_xtraj_runs_one_conformer_with_a_selection(tmp_path, engine):
    done = run_xtraj(tmp_path, ONE_CONFORMER, engine, "name N or name OG")
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    if engine == "torch":
        assert "2 of 3 atoms selected" in done.stdout


# One conformer, but the kept conformer still labelled A and interleaved with
# blank-altloc atoms: the hierarchy orders it N, C, O, CB, OG while the file
# and mdtraj order it N, CB, OG, C, O. Counts agree, so xtraj runs -- and this
# is what the refusal message tells people to make. A selection must then
# pick the named atoms, which only the file-order mask does.
INTERLEAVED_A = HEADER + N + """\
ATOM      2  CB ASER A   1       3.000   3.000   4.000  1.00 20.00           C
ATOM      3  OG ASER A   1       3.000   4.400   4.000  1.00 20.00           O
ATOM      4  C   SER A   1       5.000   7.000   9.000  1.00 20.00           C
ATOM      5  O   SER A   1       6.000   8.000  11.000  1.00 20.00           O
END
"""
ONLY_C_AND_O = HEADER + """\
ATOM      4  C   SER A   1       5.000   7.000   9.000  1.00 20.00           C
ATOM      5  O   SER A   1       6.000   8.000  11.000  1.00 20.00           O
END
"""


def fcalc(path):
    from iotbx.reflection_file_reader import any_reflection_file

    return any_reflection_file(file_name=str(path)).as_miller_arrays()[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_xtraj_selects_the_named_atoms_in_file_order(tmp_path, engine):
    import numpy as np

    run_dir, ref_dir = tmp_path / "sel", tmp_path / "ref"
    run_dir.mkdir()
    ref_dir.mkdir()
    done = run_xtraj(run_dir, INTERLEAVED_A, engine, "name C or name O")
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    done = run_xtraj(ref_dir, ONLY_C_AND_O, engine, "all")
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    sel, ref = fcalc(run_dir / "fcalc.mtz").common_sets(fcalc(ref_dir / "fcalc.mtz"))
    F, F_ref = np.abs(np.array(sel.data())), np.abs(np.array(ref.data()))
    assert np.max(np.abs(F - F_ref)) < 1e-3 * F_ref.max(), (
        "%s: selection=name C or name O gave other atoms (max |F| diff %.3g of %.3g)"
        % (engine, np.max(np.abs(F - F_ref)), F_ref.max()))


@pytest.mark.parametrize("engine", ENGINES)
def test_xtraj_refuses_alternates_with_the_default_selection(tmp_path, engine):
    done = run_xtraj(tmp_path, TWO_CONFORMERS, engine, "all")
    assert done.returncode != 0
    assert "has 5 atom records but mdtraj reads 3 atoms" in done.stdout + done.stderr


@pytest.mark.parametrize("engine", ENGINES)
def test_xtraj_refuses_an_mmcif_topology_with_altloc_labels(tmp_path, engine):
    """cctbx builds an mmCIF topology's scatterers in altloc-grouped order and
    mdtraj reads file order, so the counts agree but the atoms do not."""
    import gemmi

    pdb = tmp_path / "top.pdb"
    pdb.write_text(INTERLEAVED_A)
    gemmi.read_structure(str(pdb)).make_mmcif_document().write_file(str(tmp_path / "top.cif"))
    done = run_xtraj(tmp_path, INTERLEAVED_A, engine, "name C or name O",
                     top_name="top.cif")
    assert done.returncode != 0, "%s ran on an mmCIF topology with altloc labels" % engine
    assert "carries altloc labels" in done.stdout + done.stderr
