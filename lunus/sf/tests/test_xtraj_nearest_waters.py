"""xtraj's nearest_waters= end to end, against the calculation it stands for.

nearest_waters= keeps, in every frame, the N selected waters nearest the
solute. That is the same calculation as a trajectory holding the solute and N
water slots, where each frame's slots are filled with that frame's N nearest
waters. So the tests write both -- a box of a short peptide, an ion and 60
waters that jump to new random positions every frame, and the reduced
trajectory built from it by brute-force distances -- run xtraj on each
through the real command line, and compare the structure factors.

The peptide straddles the box boundary, so its nearest waters are found
through the periodic wrap. The reduced trajectory is built here with a full
distance matrix and a cubic minimum image, sharing nothing with the grid
search in lunus/sf/nearest_waters.py.

A few seconds per xtraj run, four runs.
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

pytest.importorskip("cctbx")
pytest.importorskip("h5py")
pytest.importorskip("gemmi")
md = pytest.importorskip("mdtraj")

BOX = 24.0
N_WATERS = 60
N_FRAMES = 4
D_MIN = "d_min=3.0"

# (atom name, element, x, y, z): three alanines lying across x = 0.
PEPTIDE = [
    [("N", "N", -3.2, 1.0, 1.5), ("CA", "C", -2.0, 1.8, 1.2), ("C", "C", -0.8, 0.9, 1.0),
     ("O", "O", -0.9, -0.3, 1.1), ("CB", "C", -2.1, 2.7, 0.0), ("HA", "H", -1.9, 2.4, 2.1)],
    [("N", "N", 0.4, 1.5, 0.7), ("CA", "C", 1.6, 0.8, 0.4), ("C", "C", 2.8, 1.7, 0.2),
     ("O", "O", 2.7, 2.9, 0.3), ("CB", "C", 1.5, -0.1, -0.8), ("HA", "H", 1.7, 0.2, 1.3)],
    [("N", "N", 4.0, 1.1, -0.1), ("CA", "C", 5.2, 1.9, -0.4), ("C", "C", 6.4, 1.0, -0.6),
     ("O", "O", 6.3, -0.2, -0.5), ("CB", "C", 5.1, 2.8, -1.6), ("HA", "H", 5.3, 2.5, 0.5)],
]
ION = (12.0, 13.0, 14.0)            # half a box from the peptide
WATER_H = np.array([[0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def atom_records():
    """The full system's atoms in file order: (name, resname, chain, resseq,
    element, B). B differs atom by atom, and the lowest is on the peptide so
    that the blur chosen from the minimum B is the same for any subset that
    keeps the peptide."""
    records = []
    for r, residue in enumerate(PEPTIDE):
        for a, (name, element, _, _, _) in enumerate(residue):
            records.append((name, "ALA", "A", r + 1, element, 10.0 + r + 0.5 * a))
    records.append(("NA", "NA", "I", 1, "NA", 25.0))
    for w in range(N_WATERS):
        records.append(("O", "HOH", "W", w + 1, "O", 20.0 + 0.4 * w))
        records.append(("H1", "HOH", "W", w + 1, "H", 22.0 + 0.4 * w))
        records.append(("H2", "HOH", "W", w + 1, "H", 23.0 + 0.4 * w))
    return records


def frames():
    """(N_FRAMES, n_atoms, 3) in Angstrom, on a 0.01 A lattice so the xtc's
    0.001 nm storage holds them exactly. Waters are redrawn every frame."""
    rng = np.random.default_rng(11)
    peptide = np.array([atom[2:] for residue in PEPTIDE for atom in residue])
    out = []
    for _ in range(N_FRAMES):
        oxygens = rng.uniform(0.0, BOX, size=(N_WATERS, 3))
        waters = np.stack([oxygens, oxygens + WATER_H[0], oxygens + WATER_H[1]],
                          axis=1).reshape(-1, 3)
        jitter = rng.normal(scale=0.1, size=peptide.shape)
        out.append(np.concatenate([peptide + jitter, [ION], waters]))
    return np.round(np.array(out), 2)


def write_system(folder, records, xyz):
    """A PDB topology and an xtc trajectory for the given atoms and frames."""
    lines = ["CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1" % (BOX, BOX, BOX)]
    for i, ((name, resname, chain, resseq, element, b), pos) in enumerate(
            zip(records, xyz[0])):
        lines.append("ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s"
                     % (i + 1, name if len(name) == 4 else " " + name, resname,
                        chain, resseq, pos[0], pos[1], pos[2], 1.0, b, element))
    lines.append("END")
    top = folder / "top.pdb"
    top.write_text("\n".join(lines) + "\n")
    traj = folder / "traj.xtc"
    md.Trajectory(
        xyz=xyz / 10.0, topology=md.load_topology(str(top)),
        unitcell_lengths=np.full((len(xyz), 3), BOX / 10.0),
        unitcell_angles=np.full((len(xyz), 3), 90.0)).save_xtc(str(traj))
    return top, traj


def water_distances(xyz_frame, records, ref_rows):
    """Brute force: each water oxygen's distance to its nearest reference
    atom, by the cubic minimum image."""
    oxygens = np.array([i for i, r in enumerate(records)
                        if r[1] == "HOH" and r[4] == "O"])
    d = xyz_frame[oxygens][:, None, :] - xyz_frame[ref_rows][None, :, :]
    d -= BOX * np.round(d / BOX)
    return np.sqrt((d ** 2).sum(axis=2)).min(axis=1)


def reduced_rows(records, nearest, keep):
    """File-order rows of the atoms a frame keeps: every non-water atom that
    passes `keep`, and the passing atoms of the waters in `nearest`."""
    chosen = set(int(w) + 1 for w in nearest)
    return [i for i, r in enumerate(records)
            if keep(r) and (r[1] != "HOH" or r[3] in chosen)]


def run(folder, top, traj, *args):
    command = [sys.executable, XTRAJ, "top=%s" % top, "traj=%s" % traj,
               "fcalc=fcalc.mtz", "icalc=icalc.mtz", "diffuse=diffuse.hkl",
               D_MIN] + list(args)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run(command, cwd=str(folder), env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        pytest.fail("xtraj failed (%d)\n%s\n%s"
                    % (done.returncode, done.stdout[-4000:], done.stderr[-4000:]))
    return done.stdout


def mtz_data(path):
    from iotbx import mtz

    array = mtz.object(str(path)).as_miller_arrays()[0]
    return array.map_to_asu().sort(by_value="packed_indices").data().as_numpy_array()


def assert_same_structure_factors(got, expected, rtol):
    for name in ("fcalc.mtz", "icalc.mtz"):
        a, b = mtz_data(got / name), mtz_data(expected / name)
        assert a.shape == b.shape
        assert a.size > 100
        scale = np.abs(b).max()
        assert scale > 0
        assert np.abs(a - b).max() <= rtol * scale, (
            "%s: largest difference %.3g against a largest value of %.3g"
            % (name, np.abs(a - b).max(), scale))


def test_shell_count_and_per_frame_choice(tmp_path):
    """nearest_waters=first: the count comes from the up-front shell
    measurement, and each frame then holds that many nearest waters.
    nearest_waters_ref= keeps the ion out of the distance measurement while
    it stays in the calculation."""
    records, xyz = atom_records(), frames()
    cutoff = 8.0
    ref_rows = [i for i, r in enumerate(records) if r[1] == "ALA" and r[4] != "H"]

    distances = [water_distances(frame, records, ref_rows) for frame in xyz]
    counts = [int((d <= cutoff).sum()) for d in distances]
    assert len(set(counts)) > 1, "the shell count should fluctuate in this test"
    assert abs(np.mean(counts) % 1 - 0.5) > 0.01, "mean on a rounding boundary"
    n = int(round(float(np.mean(counts))))
    nearest = [np.argsort(d)[:n] for d in distances]
    assert len(set(tuple(sorted(w)) for w in nearest)) == N_FRAMES, (
        "every frame should choose different waters in this test")

    full = tmp_path / "full"
    full.mkdir()
    top, traj = write_system(full, records, xyz)
    out = run(full, top, traj, "first=0", "last=%d" % (N_FRAMES - 1),
              "nearest_waters=first", "nearest_waters_cutoff=%g" % cutoff,
              "nearest_waters_ref=not resname NA")
    assert "counted on frames [0, 1, 2, 3]: %s" % counts in out
    assert "keeping %d waters per frame" % n in out

    # The reduced system: the same non-water atoms and n water slots.
    rows = [reduced_rows(records, w, lambda r: True) for w in nearest]
    reduced = tmp_path / "reduced"
    reduced.mkdir()
    top, traj = write_system(
        reduced, [records[i] for i in rows[0]],
        np.array([frame[r] for frame, r in zip(xyz, rows)]))
    run(reduced, top, traj, "first=0", "last=%d" % (N_FRAMES - 1))

    assert_same_structure_factors(full, reduced, rtol=1e-6)
    np.testing.assert_allclose(np.loadtxt(str(full / "diffuse.hkl")),
                               np.loadtxt(str(reduced / "diffuse.hkl")),
                               rtol=1e-4, atol=1e-4)

    # And it is not what keeping every water gives.
    everything = tmp_path / "everything"
    everything.mkdir()
    top, traj = write_system(everything, records, xyz)
    run(everything, top, traj, "first=0", "last=%d" % (N_FRAMES - 1))
    a, b = mtz_data(full / "fcalc.mtz"), mtz_data(everything / "fcalc.mtz")
    assert np.abs(a - b).max() > 0.01 * np.abs(b).max()


def test_torch_per_atom_b_follows_the_chosen_waters(tmp_path):
    """engine=torch with use_top_bfacs=True builds its per-atom kernels once;
    under nearest_waters= each frame must take the rows of the waters it
    chose. Every water here has its own B, and the frame chooses waters other
    than the ones the setup sized its arrays with. Hydrogens are deselected,
    so each water is its oxygen alone; with no nearest_waters_ref= the ion
    counts as solute."""
    pytest.importorskip("torch")
    records, xyz = atom_records(), frames()
    n, frame = 7, 2
    heavy = lambda r: r[4] != "H"
    ref_rows = [i for i, r in enumerate(records) if r[1] != "HOH" and heavy(r)]
    nearest = np.argsort(water_distances(xyz[frame], records, ref_rows))[:n]
    assert sorted(nearest) != list(range(n)), (
        "the frame should not choose the first n waters in this test")

    common = ["first=%d" % frame, "last=%d" % frame, "engine=torch",
              "use_top_bfacs=True", "torch_compile=False"]

    full = tmp_path / "full"
    full.mkdir()
    top, traj = write_system(full, records, xyz)
    run(full, top, traj, "selection=not element H", "nearest_waters=%d" % n, *common)

    # One frame, written as frame 0 of its own trajectory.
    rows = reduced_rows(records, nearest, heavy)
    reduced = tmp_path / "reduced"
    reduced.mkdir()
    top, traj = write_system(reduced, [records[i] for i in rows],
                             np.repeat(xyz[frame][rows][None], frame + 1, axis=0))
    run(reduced, top, traj, *common)

    assert_same_structure_factors(full, reduced, rtol=1e-4)
