"""Atoms on special positions, through the deposited-model path of the tools.

build_density(..., expand_symmetry=True) places each atom once and then adds
the density under every operation of the space group, so an atom whose site
is fixed by some operations lands on itself order_z / multiplicity times.
cctbx's direct summation sums over the same operations with the scatterer
weight occupancy * multiplicity / order_z, which counts such an atom once.
read_model must therefore hand build_density that weight, not the raw
occupancy. A P222 toy with an oxygen on a 2-fold (counted twice with the raw
occupancy) and a sulfur on the 222 point (four times) shows the difference
at full size; the carbon on a general position is a control.
"""

import io
import os
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

pytest.importorskip("cctbx")
torch = pytest.importorskip("torch")

# tools/ is not a package; the tools put it on sys.path themselves, as does
# tests/test_pdb_rfactor.py.
TOOLS = os.path.join(REPO_ROOT, "lunus", "sf", "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from fit_solvent_rfactor import build_density, read_model  # noqa: E402
from lunus.sf.structure_factor_torch import compute_fcalc  # noqa: E402

D_MIN = 1.5
GRID = (48, 60, 64)


def toy_structure(o_occupancy=1.0):
    from cctbx import crystal, xray

    cs = crystal.symmetry(unit_cell=(20, 24, 28, 90, 90, 90), space_group_symbol="P 2 2 2")
    xrs = xray.structure(crystal_symmetry=cs)
    xrs.add_scatterer(xray.scatterer(label="O1", site=(0.0, 0.0, 0.3),
                                     occupancy=o_occupancy, scattering_type="O"))
    xrs.add_scatterer(xray.scatterer(label="S1", site=(0.0, 0.0, 0.0),
                                     occupancy=1.0, scattering_type="S"))
    xrs.add_scatterer(xray.scatterer(label="C1", site=(0.13, 0.27, 0.41),
                                     occupancy=1.0, scattering_type="C"))
    xrs.scattering_type_registry(table="it1992")
    xrs.set_b_iso(20.0)
    for sc in xrs.scatterers():
        assert sc.multiplicity() > 0, "site symmetry was not applied"
    return xrs


def direct_F(xrs):
    from cctbx import miller

    xrs.scattering_type_registry(table="it1992")   # the table lunus.sf uses
    ms = miller.build_set(xrs.crystal_symmetry(), anomalous_flag=False, d_min=D_MIN)
    f = xrs.structure_factors(d_min=D_MIN, algorithm="direct").f_calc().common_set(ms)
    return np.array(f.indices()), np.array(f.data())


def lunus_F(xrs, elements, b, occ, frac, hkl):
    dens, orth, volume = build_density(xrs, elements, b, occ, frac, GRID, "cpu",
                                       torch.float64, True, log=io.StringIO(),
                                       cutoff=1e-5)
    return compute_fcalc(dens, volume, torch.tensor(hkl)).numpy()


def r_factor(F, F_ref):
    return float(np.abs(np.abs(F) - np.abs(F_ref)).sum() / np.abs(F_ref).sum())


def phase_agreement(F, F_ref):
    weight = np.abs(F) * np.abs(F_ref)
    return float(np.sum(weight * np.cos(np.angle(F) - np.angle(F_ref))) / weight.sum())


def write_toy(xrs, tmp_path):
    """The toy as a PDB file, the way the tools are handed a model."""
    path = tmp_path / "toy.pdb"
    path.write_text(xrs.as_pdb_file())
    return str(path)


def test_read_model_counts_special_positions_once(tmp_path):
    """Through read_model, the tools' own entry point: R against cctbx direct
    summation on the structure as read is zero to grid precision, and the
    phases agree."""
    xrs = toy_structure()
    hkl, F_ref = direct_F(xrs)

    xrs_read, elements, b, occ, frac, _, _, _ = read_model(write_toy(xrs, tmp_path),
                                                           log=io.StringIO())
    assert xrs_read.space_group().type().lookup_symbol() == "P 2 2 2"
    assert xrs_read.scatterers().size() == 3
    # Full occupancy on an axis is read as a special position, as in memory.
    np.testing.assert_allclose(direct_F(xrs_read)[1], F_ref, rtol=1e-6)

    F = lunus_F(xrs_read, elements, b, occ, frac, hkl)
    assert r_factor(F, F_ref) < 1e-3, "R(|F|) %.4f vs cctbx direct" % r_factor(F, F_ref)
    assert phase_agreement(F, F_ref) > 0.9999

    np.testing.assert_allclose(occ, [sc.weight() for sc in xrs_read.scatterers()])
    assert occ[0] == pytest.approx(0.5)    # O on the 2-fold: multiplicity 2 of 4
    assert occ[1] == pytest.approx(0.25)   # S on the 222 point: multiplicity 1 of 4
    assert occ[2] == pytest.approx(1.0)    # C on a general position


def test_half_occupancy_on_a_twofold_is_one_full_atom(tmp_path):
    """The reader's default: an atom with occupancy below 1 is read as a
    general position, so a file stating 0.5 for an atom on a 2-fold (the
    pre-cctbx convention, e.g. 1CM1's CA 600) comes out as one full atom --
    the same F as the file stating 1.0 for it -- and lunus.sf agrees with
    cctbx on that reading. The tools must keep the default: switching it off
    would halve such atoms in both methods at once."""
    hkl, F_full = direct_F(toy_structure(o_occupancy=1.0))

    xrs_read, elements, b, occ, frac, _, _, _ = read_model(
        write_toy(toy_structure(o_occupancy=0.5), tmp_path), log=io.StringIO())
    o = xrs_read.scatterers()[0]
    assert o.occupancy == pytest.approx(0.5)
    assert o.multiplicity() == 4           # read as a general position
    assert occ[0] == pytest.approx(0.5)    # weight == occupancy

    F_ref = direct_F(xrs_read)[1]
    np.testing.assert_allclose(np.abs(F_ref), np.abs(F_full), rtol=1e-6)
    F = lunus_F(xrs_read, elements, b, occ, frac, hkl)
    assert r_factor(F, F_ref) < 1e-3
    assert r_factor(F, F_full) < 1e-3


def test_raw_occupancy_overcounts():
    """The defect, stated as a number: the deposited occupancy handed to the
    expanding path counts the oxygen twice and the sulfur four times."""
    xrs = toy_structure()
    hkl, F_ref = direct_F(xrs)
    from fit_solvent_rfactor import model_arrays   # the seam this fix added

    elements, b, occ, frac, _, _, _ = model_arrays(xrs, log=io.StringIO())
    raw = np.array(xrs.scatterers().extract_occupancies())
    assert r_factor(lunus_F(xrs, elements, b, raw, frac, hkl), F_ref) > 1.0
    assert r_factor(lunus_F(xrs, elements, b, occ, frac, hkl), F_ref) < 1e-3


def test_match_atoms_by_position():
    """compare_three_way pairs gemmi's atoms (chain by chain) with cctbx's
    scatterers (file order) by position: reversed order, a wrapped cell
    boundary, a coincident altloc pair that must pair off rather than both
    claim one scatterer, an element check, and a loud failure when an atom
    has no partner."""
    from compare_three_way import match_atoms_by_position

    orth = np.diag([20.0, 24.0, 28.0])
    frac_b = [[0.10, 0.20, 0.30], [0.50, 0.50, 0.50], [0.50, 0.50, 0.50],
              [0.999, 0.001, 0.75]]
    elements_b = ["O1-", "C", "C", "Ca2+"]
    frac_a = [[-0.001, 1.001, 0.75], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
              [1.10, 0.20, -0.70]]
    elements_a = ["Ca", "C", "C", "O"]
    order = match_atoms_by_position(frac_a, elements_a, frac_b, elements_b, orth)
    assert order[0] == 3 and order[3] == 0
    assert sorted(order[1:3]) == [1, 2]
    assert len(set(order)) == 4

    # Minimum image: one copy reduced to just below 1, the other to just
    # above 0 (an atom moved onto x = 0 from a file x slightly negative).
    assert match_atoms_by_position([[0.99998, 0.5, 0.5]], ["N"],
                                   [[0.00002, 0.5, 0.5]], ["N"], orth) == [0]

    with pytest.raises(ValueError):
        match_atoms_by_position([[0.5, 0.5, 0.5]], ["N"], frac_b, elements_b, orth)
    with pytest.raises(ValueError):   # 2.8 A away
        match_atoms_by_position([[0.5, 0.5, 0.6]], ["C"], frac_b, elements_b, orth)


def write_off_axis(xrs, tmp_path, old, new, name):
    """as_pdb_file writes the snapped site; put one atom back where the
    file would have had it."""
    lines = xrs.as_pdb_file().splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith("ATOM"))
    assert old in lines[first]
    lines[first] = lines[first].replace(old, new)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_match_covers_the_snap_onto_a_special_position(tmp_path):
    """A full-occupancy atom 0.15 A off a 2-fold: cctbx moves it onto the
    axis, gemmi reads the file coordinate, and the two must still pair --
    this is the model compare_three_way's weight handling exists for."""
    gemmi = pytest.importorskip("gemmi")
    from compare_three_way import file_fractionals, match_atoms_by_position
    from lunus.sf.cell_utils import orth_matrix

    path = write_off_axis(toy_structure(), tmp_path, "   0.000   0.000   8.400",
                          "   0.150   0.000   8.400", "off_axis.pdb")
    xrs_read, elements, b, occ, frac, _, _, _ = read_model(path, log=io.StringIO())
    np.testing.assert_allclose(frac[0], [0.0, 0.0, 0.3], atol=1e-12)   # snapped
    assert occ[0] == pytest.approx(0.5)

    st = gemmi.read_structure(path)
    atoms = [cra.atom for cra in st[0].all()]
    assert st.cell.fractionalize(atoms[0].pos).x == pytest.approx(0.0075)
    M = orth_matrix(*xrs_read.unit_cell().parameters())
    order = match_atoms_by_position(
        [st.cell.fractionalize(a.pos).tolist() for a in atoms],
        [a.element.name for a in atoms], file_fractionals(path), elements, M)
    assert order == [0, 1, 2]


def test_match_survives_a_snap_larger_than_any_tolerance(tmp_path):
    """Where symmetry elements intersect, cctbx's move onto the exact site
    exceeds min_distance_sym_equiv: an O 0.56 A from the 422 point of P422
    is placed on it. Matching on file coordinates does not care how far the
    scatterer moved; matching against the snapped site would."""
    gemmi = pytest.importorskip("gemmi")
    from cctbx import crystal, xray
    from compare_three_way import file_fractionals, match_atoms_by_position
    from lunus.sf.cell_utils import orth_matrix

    cs = crystal.symmetry(unit_cell=(20, 20, 28, 90, 90, 90), space_group_symbol="P 4 2 2")
    xrs = xray.structure(crystal_symmetry=cs)
    xrs.add_scatterer(xray.scatterer(label="O1", site=(0.0, 0.0, 0.0),
                                     occupancy=1.0, scattering_type="O"))
    xrs.add_scatterer(xray.scatterer(label="C1", site=(0.13, 0.27, 0.41),
                                     occupancy=1.0, scattering_type="C"))
    xrs.set_b_iso(20.0)
    path = write_off_axis(xrs, tmp_path, "   0.000   0.000   0.000",
                          "   0.240   0.500   0.060", "p422.pdb")

    xrs_read, elements, b, occ, frac, _, _, _ = read_model(path, log=io.StringIO())
    np.testing.assert_allclose(frac[0], [0.0, 0.0, 0.0], atol=1e-12)
    assert occ[0] == pytest.approx(1.0 / 8)                   # 422 point: 1 of 8
    moved = np.linalg.norm(np.array([0.240, 0.500, 0.060]))
    assert moved > xrs_read.min_distance_sym_equiv()          # 0.56 A > 0.5 A

    st = gemmi.read_structure(path)
    atoms = [cra.atom for cra in st[0].all()]
    M = orth_matrix(*xrs_read.unit_cell().parameters())
    order = match_atoms_by_position(
        [st.cell.fractionalize(a.pos).tolist() for a in atoms],
        [a.element.name for a in atoms], file_fractionals(path), elements, M)
    assert order == [0, 1]
    with pytest.raises(ValueError):      # against the snapped sites it cannot
        match_atoms_by_position(
            [st.cell.fractionalize(a.pos).tolist() for a in atoms],
            [a.element.name for a in atoms], frac, elements, M)


def test_special_position_weights_flag_keeps_deposited_occupancies(tmp_path):
    """read_model(..., special_position_weights=False) hands back the file's
    occupancies: for a complete cell expanded as a fold, or placed once."""
    xrs = toy_structure()
    path = write_toy(xrs, tmp_path)
    log = io.StringIO()
    xrs_raw, _, _, occ, _, _, _, _ = read_model(path, log=log, special_position_weights=False)
    np.testing.assert_allclose(occ, [1.0, 1.0, 1.0])
    assert "0 on special positions" in log.getvalue()     # read as general, unsnapped
    assert "2 atom(s) cctbx would put on special positions" in log.getvalue()
    assert [sc.multiplicity() for sc in xrs_raw.scatterers()] == [4, 4, 4]
    _, _, _, occ_w, _, _, _, _ = read_model(path, log=io.StringIO())
    np.testing.assert_allclose(occ_w, [0.5, 0.25, 1.0])

    # model_arrays on a structure read WITH snapping still honours the flag,
    # and says which atoms it left at their deposited occupancy.
    from fit_solvent_rfactor import model_arrays
    log = io.StringIO()
    _, _, occ_m, _, _, _, _ = model_arrays(xrs, log=log, special_position_weights=False)
    np.testing.assert_allclose(occ_m, [1.0, 1.0, 1.0])
    assert "deposited occupancies kept for the 2 atom(s)" in log.getvalue()


ALTLOC_PDB = """\
CRYST1   20.000   24.000   28.000  90.00  90.00  90.00 P 2 2 2
ATOM      1  N   SER A   1       2.000   3.000   4.000  1.00 20.00           N
ATOM      2  CB ASER A   1       3.000   3.000   4.000  0.50 20.00           C
ATOM      3  CB BSER A   1       3.300   3.000   4.000  0.50 20.00           C
ATOM      4  OG ASER A   1       3.000   4.400   4.000  0.50 20.00           O
ATOM      5  OG BSER A   1       3.300   4.400   4.000  0.50 20.00           O
END
"""


def test_file_fractionals_follow_the_scatterer_order_with_altlocs(tmp_path):
    """iotbx's hierarchy groups atoms by altloc, so its atoms() are not in
    scatterer order on a file whose altlocs are interleaved (7FPV: 1004 of
    3341 atoms sit elsewhere). file_fractionals must return the file's
    coordinates in scatterer order (it reads with cctbx's own reader, with
    the snapping off), not the hierarchy's."""
    from compare_three_way import file_fractionals

    path = tmp_path / "altloc.pdb"
    path.write_text(ALTLOC_PDB)
    xrs_read, elements, b, occ, frac, _, _, _ = read_model(str(path), log=io.StringIO())
    assert elements == ["N", "C", "C", "O", "O"]
    frac_file = file_fractionals(str(path))
    np.testing.assert_allclose(frac_file, frac, atol=1e-12)   # no snaps here
    # The hierarchy's own order puts both CB before both OG: CB B third.
    assert frac_file[2][0] == pytest.approx(3.3 / 20.0)


def test_file_fractionals_honour_scale_records(tmp_path):
    """cctbx (use_scale_matrix_if_available) and gemmi both fractionalize
    through the file's SCALE records, so on a file whose SCALE frame differs
    from the CRYST1 orthogonalization -- here x and y swapped -- the match
    must see the same frame on both sides. Fractionalizing the file's
    Cartesians by the CRYST1 matrix put every atom a rotation away from its
    partner and the match raised on the first atom."""
    gemmi = pytest.importorskip("gemmi")
    from compare_three_way import file_fractionals, match_atoms_by_position
    from lunus.sf.cell_utils import orth_matrix

    lines = [l for l in toy_structure().as_pdb_file().splitlines()
             if not l.startswith("SCALE")]
    cryst = next(i for i, l in enumerate(lines) if l.startswith("CRYST1"))
    lines[cryst + 1:cryst + 1] = [
        "SCALE1      0.000000  0.041667  0.000000        0.00000",
        "SCALE2      0.050000  0.000000  0.000000        0.00000",
        "SCALE3      0.000000  0.000000  0.035714        0.00000"]
    path = tmp_path / "scaled.pdb"
    path.write_text("\n".join(lines) + "\n")

    xrs_read, elements, b, occ, frac, _, _, _ = read_model(str(path), log=io.StringIO())
    # SCALE applied (x and y swapped); its rows carry 6 decimals, hence 1e-4
    np.testing.assert_allclose(frac[2], [0.27, 0.13, 0.41], atol=1e-4)
    st = gemmi.read_structure(str(path))
    atoms = [cra.atom for cra in st[0].all()]
    np.testing.assert_allclose(st.cell.fractionalize(atoms[2].pos).tolist(),
                               [0.27, 0.13, 0.41], atol=1e-4)
    np.testing.assert_allclose(file_fractionals(str(path)), frac, atol=1e-12)
    M = orth_matrix(*xrs_read.unit_cell().parameters())
    order = match_atoms_by_position(
        [st.cell.fractionalize(a.pos).tolist() for a in atoms],
        [a.element.name for a in atoms], file_fractionals(str(path)), elements, M)
    assert order == [0, 1, 2]


def test_run_rule_for_special_position_weights():
    """Weights only for an asymmetric unit that the expansion completes."""
    from fit_solvent_rfactor import use_special_position_weights

    assert use_special_position_weights(False, False) is True
    assert use_special_position_weights(True, False) is False     # fold
    assert use_special_position_weights(False, True) is False     # placed once
    assert use_special_position_weights(True, True) is False


def test_run_passes_the_flags_to_read_model(tmp_path, monkeypatch):
    """The wiring: run() hands read_model the rule's answer. read_model is
    the first call in run() that does any work, so a stand-in that records
    its keyword and stops is enough."""
    import fit_solvent_rfactor as fsr

    seen = {}

    def stand_in(path, log=None, special_position_weights=True):
        seen["flag"] = special_position_weights
        raise RuntimeError("stop here")

    monkeypatch.setattr(fsr, "read_model", stand_in)
    pdb = write_toy(toy_structure(), tmp_path)
    for extra, expected in (([], True), (["--no-special-position-weights"], False),
                            (["--no-expand-symmetry"], False)):
        args = fsr.build_parser().parse_args([pdb, str(tmp_path / "none.cif")] + extra)
        with pytest.raises(RuntimeError, match="stop here"):
            fsr.run(args)
        assert seen["flag"] is expected, extra


def test_opt_out_keeps_file_coordinates_too(tmp_path):
    """With special_position_weights=False nothing is moved onto a symmetry
    element either: four full-occupancy O atoms 0.3 A around a 4-fold (a
    complete cell with every copy its own) stay a ring, read as general
    positions. With the default they are snapped onto the axis."""
    ring = ["CRYST1   20.000   20.000   28.000  90.00  90.00  90.00 P 4"]
    for i, (x, y) in enumerate([(0.3, 0.0), (0.0, 0.3), (-0.3, 0.0), (0.0, -0.3)]):
        ring.append("ATOM  %5d  O   HOH A%4d    %8.3f%8.3f%8.3f  1.00 20.00           O"
                    % (i + 1, i + 1, x, y, 10.0))
    path = tmp_path / "ring.pdb"
    path.write_text("\n".join(ring + ["END"]) + "\n")

    xrs_raw, _, _, occ_raw, frac_raw, _, _, _ = read_model(
        str(path), log=io.StringIO(), special_position_weights=False)
    assert [sc.multiplicity() for sc in xrs_raw.scatterers()] == [4, 4, 4, 4]
    np.testing.assert_allclose(occ_raw, 1.0)
    np.testing.assert_allclose(np.abs(frac_raw[:, :2]).max(axis=1), 0.3 / 20.0, atol=1e-9)

    xrs_w, _, _, occ_w, frac_w, _, _, _ = read_model(str(path), log=io.StringIO())
    assert [sc.multiplicity() for sc in xrs_w.scatterers()] == [1, 1, 1, 1]
    np.testing.assert_allclose(occ_w, 0.25)
    np.testing.assert_allclose(frac_w[:, :2], 0.0, atol=1e-12)


@pytest.mark.parametrize("o_occupancy", [1.0, 0.5])
def test_study_taper_writes_occ_one_and_b_iso_back(tmp_path, o_occupancy):
    """--occ-one and --b-iso must reach the scatterers too, or the exact
    target direct-sums a different model from the one being fitted: after
    load_system, sc.weight() equals the occ array exactly and every B is the
    override. A half-occupancy O on the 2-fold, read as general, keeps 0.5
    (one full atom on the axis, not two)."""
    from study_taper import load_system

    xrs = toy_structure(o_occupancy)
    xrs.scatterers()[2].occupancy = 0.6      # the general-position C: what the flag is for
    path = write_toy(xrs, tmp_path)
    s = load_system(path, 3.0, "cpu", torch.float64, io.StringIO(),
                    b_iso=30.0, occ_one=True)
    occ = s["occ"].numpy()
    weights = np.array([sc.weight() for sc in s["xrs"].scatterers()])
    np.testing.assert_array_equal(weights, occ)
    np.testing.assert_allclose(occ, [0.5, 0.25, 1.0])
    assert s["xrs"].scatterers()[0].occupancy == pytest.approx(o_occupancy)
    assert s["xrs"].scatterers()[2].occupancy == pytest.approx(1.0)   # written back
    b_back = np.array(s["xrs"].extract_u_iso_or_u_equiv()) * (8.0 * np.pi ** 2)
    np.testing.assert_allclose(b_back, 30.0)
    np.testing.assert_allclose(s["b"], 30.0)
