"""mask_spheres: the radius table and typing, the Gaussian fit of the soft
sphere, and the kernels placed through compute_density."""
import numpy as np
import pytest

from conftest import requires_gemmi, requires_torch

from lunus.sf import mask_spheres as ms

# np.trapz was renamed np.trapezoid in numpy 2.0; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# Every heavy atom of the standard residues, as Tinker's oplsua.prm types it
# (its "biotype" lines, mapped through the vdw table; HIP/HID/HIE are its
# histidine protonation states), extracted 2026-11-10.
TINKER_BIOTYPES = [
    ("GLY", "N", 3.250),
    ("GLY", "CA", 3.800),
    ("GLY", "C", 3.750),
    ("GLY", "O", 2.960),
    ("ALA", "N", 3.250),
    ("ALA", "CA", 3.800),
    ("ALA", "C", 3.750),
    ("ALA", "O", 2.960),
    ("ALA", "CB", 3.910),
    ("VAL", "N", 3.250),
    ("VAL", "CA", 3.800),
    ("VAL", "C", 3.750),
    ("VAL", "O", 2.960),
    ("VAL", "CB", 3.850),
    ("VAL", "CG1", 3.910),
    ("VAL", "CG2", 3.910),
    ("LEU", "N", 3.250),
    ("LEU", "CA", 3.800),
    ("LEU", "C", 3.750),
    ("LEU", "O", 2.960),
    ("LEU", "CB", 3.905),
    ("LEU", "CG", 3.850),
    ("LEU", "CD1", 3.910),
    ("LEU", "CD2", 3.910),
    ("ILE", "N", 3.250),
    ("ILE", "CA", 3.800),
    ("ILE", "C", 3.750),
    ("ILE", "O", 2.960),
    ("ILE", "CB", 3.850),
    ("ILE", "CG1", 3.905),
    ("ILE", "CG2", 3.910),
    ("ILE", "CD", 3.905),
    ("SER", "N", 3.250),
    ("SER", "CA", 3.800),
    ("SER", "C", 3.750),
    ("SER", "O", 2.960),
    ("SER", "CB", 3.905),
    ("SER", "OG", 3.070),
    ("THR", "N", 3.250),
    ("THR", "CA", 3.800),
    ("THR", "C", 3.750),
    ("THR", "O", 2.960),
    ("THR", "CB", 3.850),
    ("THR", "OG1", 3.070),
    ("THR", "CG2", 3.910),
    ("CYS", "N", 3.250),
    ("CYS", "CA", 3.800),
    ("CYS", "C", 3.750),
    ("CYS", "O", 2.960),
    ("CYS", "CB", 3.905),
    ("CYS", "SG", 3.550),
    ("PRO", "N", 3.250),
    ("PRO", "CA", 3.800),
    ("PRO", "C", 3.750),
    ("PRO", "O", 2.960),
    ("PRO", "CB", 3.905),
    ("PRO", "CG", 3.905),
    ("PRO", "CD", 3.800),
    ("PHE", "N", 3.250),
    ("PHE", "CA", 3.800),
    ("PHE", "C", 3.750),
    ("PHE", "O", 2.960),
    ("PHE", "CB", 3.905),
    ("PHE", "CG", 3.550),
    ("PHE", "CD", 3.550),
    ("PHE", "CE", 3.550),
    ("PHE", "CZ", 3.550),
    ("TYR", "N", 3.250),
    ("TYR", "CA", 3.800),
    ("TYR", "C", 3.750),
    ("TYR", "O", 2.960),
    ("TYR", "CB", 3.905),
    ("TYR", "CG", 3.550),
    ("TYR", "CD", 3.550),
    ("TYR", "CE", 3.550),
    ("TYR", "CZ", 3.550),
    ("TYR", "OH", 3.070),
    ("TRP", "N", 3.250),
    ("TRP", "CA", 3.800),
    ("TRP", "C", 3.750),
    ("TRP", "O", 2.960),
    ("TRP", "CB", 3.905),
    ("TRP", "CG", 3.550),
    ("TRP", "CD1", 3.550),
    ("TRP", "CD2", 3.550),
    ("TRP", "NE1", 3.250),
    ("TRP", "CE2", 3.550),
    ("TRP", "CE3", 3.550),
    ("TRP", "CZ2", 3.550),
    ("TRP", "CZ3", 3.550),
    ("TRP", "CH2", 3.550),
    ("HIP", "N", 3.250),
    ("HIP", "CA", 3.800),
    ("HIP", "C", 3.750),
    ("HIP", "O", 2.960),
    ("HIP", "CB", 3.905),
    ("HIP", "CG", 3.550),
    ("HIP", "ND1", 3.250),
    ("HIP", "CD2", 3.550),
    ("HIP", "CE1", 3.550),
    ("HIP", "NE2", 3.250),
    ("HID", "N", 3.250),
    ("HID", "CA", 3.800),
    ("HID", "C", 3.750),
    ("HID", "O", 2.960),
    ("HID", "CB", 3.905),
    ("HID", "CG", 3.550),
    ("HID", "ND1", 3.250),
    ("HID", "CD2", 3.550),
    ("HID", "CE1", 3.550),
    ("HID", "NE2", 3.250),
    ("HIE", "N", 3.250),
    ("HIE", "CA", 3.800),
    ("HIE", "C", 3.750),
    ("HIE", "O", 2.960),
    ("HIE", "CB", 3.905),
    ("HIE", "CG", 3.550),
    ("HIE", "ND1", 3.250),
    ("HIE", "CD2", 3.550),
    ("HIE", "CE1", 3.550),
    ("HIE", "NE2", 3.250),
    ("ASP", "N", 3.250),
    ("ASP", "CA", 3.800),
    ("ASP", "C", 3.750),
    ("ASP", "O", 2.960),
    ("ASP", "CB", 3.905),
    ("ASP", "CG", 3.750),
    ("ASP", "OD", 2.960),
    ("ASN", "N", 3.250),
    ("ASN", "CA", 3.800),
    ("ASN", "C", 3.750),
    ("ASN", "O", 2.960),
    ("ASN", "CB", 3.905),
    ("ASN", "CG", 3.750),
    ("ASN", "OD1", 2.960),
    ("ASN", "ND2", 3.250),
    ("GLU", "N", 3.250),
    ("GLU", "CA", 3.800),
    ("GLU", "C", 3.750),
    ("GLU", "O", 2.960),
    ("GLU", "CB", 3.905),
    ("GLU", "CG", 3.905),
    ("GLU", "CD", 3.750),
    ("GLU", "OE", 2.960),
    ("GLN", "N", 3.250),
    ("GLN", "CA", 3.800),
    ("GLN", "C", 3.750),
    ("GLN", "O", 2.960),
    ("GLN", "CB", 3.905),
    ("GLN", "CG", 3.905),
    ("GLN", "CD", 3.750),
    ("GLN", "OE1", 2.960),
    ("GLN", "NE2", 3.250),
    ("MET", "N", 3.250),
    ("MET", "CA", 3.800),
    ("MET", "C", 3.750),
    ("MET", "O", 2.960),
    ("MET", "CB", 3.905),
    ("MET", "CG", 3.800),
    ("MET", "SD", 3.550),
    ("MET", "CE", 3.800),
    ("LYS", "N", 3.250),
    ("LYS", "CA", 3.800),
    ("LYS", "C", 3.750),
    ("LYS", "O", 2.960),
    ("LYS", "CB", 3.905),
    ("LYS", "CG", 3.905),
    ("LYS", "CD", 3.905),
    ("LYS", "CE", 3.905),
    ("LYS", "NZ", 3.250),
    ("ARG", "N", 3.250),
    ("ARG", "CA", 3.800),
    ("ARG", "C", 3.750),
    ("ARG", "O", 2.960),
    ("ARG", "CB", 3.905),
    ("ARG", "CG", 3.905),
    ("ARG", "CD", 3.905),
    ("ARG", "NE", 3.250),
    ("ARG", "CZ", 2.250),
    ("ARG", "NH", 3.250),
]


def test_every_standard_residue_atom_matches_tinker():
    """Each heavy atom of each standard residue gets the sigma Tinker's
    OPLS-UA biotypes give it."""
    bad = []
    for res, atom, sigma in TINKER_BIOTYPES:
        got = ms.lj_sigma(res, atom, atom[0])
        if abs(got - sigma) > 1e-6:
            bad.append((res, atom, sigma, got))
    assert not bad, bad


@requires_gemmi
def test_charged_scattering_types_keep_their_type():
    """cctbx carries charges in the scattering type ('O1-'); the typing and
    the element fallback strip them."""
    assert ms.united_atom_type("ASP", "OD1", "O1-") == "O"
    assert ms.united_atom_type("LYS", "NZ", "N1+") == "N"
    assert ms.lj_sigma("ASP", "OD1", "O1-") == ms.SIGMA_UA["O"]
    assert ms.lj_sigma("ZN", "ZN", "Zn2+") == ms.lj_sigma("ZN", "ZN", "Zn")
    assert ms.lj_sigma("LIG", "O1", "O1-") == ms.lj_sigma("LIG", "O1", "O")


def test_explicit_hydrogens_are_decided_per_residue():
    """A ligand's hydrogens do not shrink the protein: only a residue that
    carries hydrogens has its sp3 carbons bare."""
    labels = ['pdb=" CB  ALA A   1 "', 'pdb=" CB  ALA A   2 "', 'pdb=" HB1 ALA A   2 "',
              'pdb=" C1  LIG A 900 "', 'pdb=" H1  LIG A 900 "']
    r = ms.sphere_radii(labels, ["C", "C", "H", "C", "H"])
    assert r[0] == 0.5 * ms.SIGMA_UA["CH3B"]          # residue 1 has no H
    assert r[1] == 0.5 * ms.SIGMA_UA["C"]             # residue 2 does
    assert r[2] == 0.5 * ms.SIGMA_UA["H"]
    forced = ms.sphere_radii(labels, ["C", "C", "H", "C", "H"], explicit_hydrogens=True)
    assert forced[0] == 0.5 * ms.SIGMA_UA["C"]


@requires_gemmi
def test_united_atom_typing_and_radii():
    """The published OPLS-UA protein types (Tinker oplsua.prm), the element
    fallback, and explicit hydrogens flattening the sp3 carbons to bare C."""
    cases = {("ALA", "CB", "C"): "CH3B", ("GLY", "CA", "C"): "CH2A", ("LEU", "CA", "C"): "CHA",
             ("PRO", "CA", "C"): "CHA", ("LEU", "CG", "C"): "CH", ("VAL", "CB", "C"): "CH",
             ("THR", "CB", "C"): "CH", ("LEU", "CD1", "C"): "CH3B", ("THR", "CG2", "C"): "CH3B", ("ILE", "CD1", "C"): "CH3",
             ("LYS", "CE", "C"): "CH2", ("PHE", "CB", "C"): "CH2", ("MET", "CE", "C"): "CH3S",
             ("MET", "CG", "C"): "CH2X", ("PRO", "CD", "C"): "CH2X", ("PHE", "CZ", "C"): "CAR",
             ("TYR", "CZ", "C"): "CAR", ("HIS", "CE1", "C"): "CAR", ("TRP", "CD2", "C"): "CAR",
             ("ASP", "CG", "C"): "C", ("GLN", "CD", "C"): "C", ("ARG", "CZ", "C"): "CZR",
             ("ALA", "C", "C"): "C", ("SER", "OG", "O"): "OH", ("TYR", "OH", "O"): "OH",
             ("ASP", "OD1", "O"): "O", ("ALA", "OXT", "O"): "O", ("HOH", "O", "O"): "OW",
             ("MET", "SD", "S"): "S", ("LYS", "NZ", "N"): "N", ("ALA", "HB1", "H"): "H",
             ("LIG", "C1", "C"): None, ("LIG", "ZN", "Zn"): None}
    for (res, name, el), expected in cases.items():
        assert ms.united_atom_type(res, name, el) == expected, (res, name, el)
    # spot values against the published table
    assert ms.SIGMA_UA["CHA"] == 3.80 and ms.SIGMA_UA["CAR"] == 3.55
    assert ms.SIGMA_UA["CZR"] == 2.25 and ms.SIGMA_UA["OW"] == 3.151 and ms.SIGMA_UA["CH3B"] == 3.91
    # a methyl carbon is a larger object than a carbonyl carbon
    assert ms.lj_sigma("ALA", "CB", "C") > ms.lj_sigma("ALA", "C", "C")
    # explicit hydrogens: sp3 carbons become bare C, aromatic and polar types stay
    assert ms.lj_sigma("ALA", "CB", "C", explicit_hydrogens=True) == ms.SIGMA_UA["C"]
    assert ms.lj_sigma("PHE", "CZ", "C", explicit_hydrogens=True) == ms.SIGMA_UA["CAR"]
    assert ms.lj_sigma("ALA", "HB1", "H", explicit_hydrogens=True) == ms.SIGMA_UA["H"]
    # the fallback keeps one scale: a bare carbon lands on the OPLS carbon
    assert abs(ms.lj_sigma("LIG", "C1", "C") - ms.SIGMA_UA["C"]) < 1e-5   # gemmi's radii are float32
    assert 2.5 < ms.lj_sigma("LIG", "ZN", "Zn") < 3.5
    assert abs(ms.contact_radius("ALA", "C", "C") - 0.5 * (3.75 + ms.SIGMA_WATER)) < 1e-9
    labels = ['pdb=" CA AMET A   1 "', 'pdb=" O   HOH A 465 "', 'pdb=" CB  ALA A   2 "']
    r = ms.sphere_radii(labels, ["C", "O", "C"], offset=0.3)
    assert np.allclose(r, [0.5 * 3.80 + 0.3, 0.5 * 3.151 + 0.3, 0.5 * 3.91 + 0.3])
    r_el = ms.sphere_radii(labels, ["C", "O", "C"], by_element=True)
    assert np.allclose(r_el, [0.5 * 3.75, 1.52 * (3.75 / 3.40), 0.5 * 3.75], atol=1e-5)
    # by_element skips the typing for every element, N and S included
    n_el = ms.sphere_radii(['pdb=" NZ  LYS A   1 "'], ["N"], by_element=True)[0]
    s_el = ms.sphere_radii(['pdb=" SD  MET A   1 "'], ["S"], by_element=True)[0]
    assert abs(n_el - 1.55 * (3.75 / 3.40)) < 1e-5 and abs(s_el - 1.80 * (3.75 / 3.40)) < 1e-5
    with pytest.raises(ValueError):
        ms.sphere_radii(labels, ["C", "O", "C"], offset=-2.0)

def test_gaussian_fit_reproduces_the_soft_sphere():
    """Five bounded Gaussians: plateau within 5%, the half-height crossing
    within 0.03 A of the radius, the tail beyond radius + 1.5 A below 1%."""
    pytest.importorskip("scipy")
    for radius in (1.5, 1.9, 2.3):
        A, lam = ms.gaussian_fit(radius, 0.5)
        assert len(A) == 5 and len(lam) == 5 and max(abs(a) for a in A) <= 20.0
        r = np.linspace(0.0, radius + 3.0, 601)
        f = sum(a * np.exp(-l * r * r) for a, l in zip(A, lam))
        t = ms.fermi(r, radius, 0.5)
        assert np.abs(f - t)[r < radius - 1.0].max() < 0.05
        crossing = r[np.argmin(np.abs(f - 0.5))]
        assert abs(crossing - radius) < 0.03
        assert np.abs(f[r > radius + 1.5]).max() < 0.01 + t[r > radius + 1.5].max()


@requires_torch
def test_sphere_kernels_place_a_sphere():
    """One atom through compute_density: unit plateau at the centre, half at
    the radius, nothing far away, and the volume of a sphere."""
    import torch

    from lunus.sf.cell_utils import orth_matrix
    from lunus.sf.density_torch import compute_density

    cell = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)
    M = orth_matrix(*cell)
    grid = (80, 80, 80)                                # 0.25 A voxels
    radius = 2.0
    A, lam, offsets, reach, taper, idx = ms.sphere_kernels([radius], grid, M, width=0.5)
    frac = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    field = compute_density(frac, idx, torch.ones(1, dtype=torch.float64), A, lam, offsets,
                            reach, grid, torch.tensor(M, dtype=torch.float64), taper,
                            compile_core=False).numpy()
    c = 40
    assert abs(field[c, c, c] - 1.0) < 0.06
    assert abs(field[c + 8, c, c] - 0.5) < 0.06                 # 2.0 A along x
    assert abs(field[c + 16, c, c]) < 0.02                      # 4.0 A
    # the volume of the SOFT sphere (a 0.5 A edge carries a long tail: 0.12
    # at radius + 1 A), not of a sharp one
    vol = field.sum() * (20.0 ** 3 / field.size)
    rr = np.linspace(0.0, 8.0, 4001)
    vol_fermi = 4.0 * np.pi * _trapezoid(rr * rr * ms.fermi(rr, radius, 0.5), rr)
    assert abs(vol - vol_fermi) < 0.03 * vol_fermi


@requires_gemmi
def test_default_radii_come_from_the_table_without_fitting(monkeypatch):
    """Every radius the default table and element fallback produce is in
    SPHERE_FIT_TABLE, so a default run never calls the fit; each tabulated
    entry meets the same quality bar as a live fit; an off-table radius
    still fits live."""
    calls = []
    real = ms.gaussian_fit

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(ms, "gaussian_fit", spy)
    for t in ms.SIGMA_UA:
        assert round(0.5 * ms.SIGMA_UA[t], 3) in ms.SPHERE_FIT_TABLE, t
    for e in ("C", "N", "O", "S", "P", "Zn", "Mg", "Ca", "Na", "Cl", "Fe", "Se"):
        ms.sphere_coefficients(0.5 * ms.lj_sigma("", "", e))
    labels = ['pdb=" CA AMET A   1 "', 'pdb=" O   HOH A 465 "', 'pdb=" CB  ALA A   2 "',
              'pdb=" SD  MET A   1 "', 'pdb="ZN   ZN  A 900 "']
    for d in ms.sphere_radii(labels, ["C", "O", "C", "S", "Zn"]):
        ms.sphere_coefficients(d)
    assert calls == []
    for radius, (A, lam) in ms.SPHERE_FIT_TABLE.items():
        r = np.linspace(0.0, radius + 3.0, 401)
        f = sum(a * np.exp(-l * r * r) for a, l in zip(A, lam))
        t = ms.fermi(r, radius, ms.EDGE_WIDTH_DEFAULT)
        assert np.abs(f - t)[r < radius - 1.0].max() < 0.08, radius
        assert abs(r[np.argmin(np.abs(f - 0.5))] - radius) < 0.04, radius
    pytest.importorskip("scipy")
    ms.sphere_coefficients(1.234)                      # off the table: fitted live
    assert len(calls) == 1
