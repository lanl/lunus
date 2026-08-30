"""
Parity between our density-threshold solvent mask and gemmi's geometric one.

This compares two DIFFERENT MODELS -- gemmi's SolventMasker excludes everything
within (vdW radius + rprobe) of an atom and then erodes by rshrink, while ours
thresholds the density that splat_density already computed -- so the target is
agreement at the level of the model difference, not to machine precision. The
question the test answers is "is the threshold mask a defensible stand-in for a
geometric one", and the yardstick is calibrated rather than invented:

    gemmi's OWN radii sets disagree with each other by more than ours
    disagrees with gemmi.

Measured on the fixture below (200 carbons clustered in a 30 A cell, 0.886
solvent fraction, 0.625 A grid), binarised voxel agreement:

    ours vs gemmi Cctbx           0.995
    gemmi Refmac vs Cctbx         0.987
    gemmi VanDerWaals vs Cctbx    0.985

and |F_mask| correlates at 0.9998 with an R-factor of ~0.036 over the low
resolution reflections where bulk solvent actually contributes.

Requires gemmi; skipped without it.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
gemmi = pytest.importorskip("gemmi", reason="gemmi not installed")

from lunus.sf.cell_utils import orth_matrix
from lunus.sf.density_torch import splat_density
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.solvent_torch import (
    f_solvent, mask_occupancy, shell_voxels, solvent_mask,
)

CELL = (30.0, 30.0, 30.0, 90.0, 90.0, 90.0)
GRID = (48, 48, 48)
B_ISO = 20.0
M_NP = orth_matrix(*CELL)
VOLUME = float(np.linalg.det(M_NP))


def blob(n=200, radius=8.0, seed=0):
    """n atoms packed into a sphere at the cell centre with a 1.4 A minimum
    separation -- a molecule-shaped region with a genuine solvent boundary,
    which uniformly random coordinates would not have."""
    rng = np.random.default_rng(seed)
    centre = np.array([15.0, 15.0, 15.0])
    pts = []
    while len(pts) < n:
        u = rng.normal(size=3)
        p = centre + radius * rng.random() ** (1 / 3.0) * u / np.linalg.norm(u)
        if all(np.linalg.norm(p - q) > 1.4 for q in pts):
            pts.append(p)
    return np.array(pts)


def gemmi_mask(cart, b_iso=B_ISO, radii_set=None):
    """gemmi's geometric solvent mask on the same cell and grid as ours."""
    if radii_set is None:
        radii_set = gemmi.AtomicRadiiSet.Cctbx
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(*CELL)
    st.spacegroup_hm = "P 1"
    model, chain = gemmi.Model("1"), gemmi.Chain("A")
    res = gemmi.Residue()
    res.name, res.seqid = "UNK", gemmi.SeqId("1")
    for i, c in enumerate(cart):
        a = gemmi.Atom()
        a.name, a.element = "C%d" % i, gemmi.Element("C")
        a.pos, a.occ, a.b_iso = gemmi.Position(*c), 1.0, b_iso
        res.add_atom(a)
    chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)

    grid = gemmi.FloatGrid()
    grid.setup_from(st)
    grid.set_size(*GRID)
    gemmi.SolventMasker(radii_set).put_mask_on_float_grid(grid, st[0])
    return np.array(grid, copy=True)


def our_density(cart, b_iso=B_ISO):
    frac = cart @ np.linalg.inv(M_NP).T
    n = len(cart)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, b_iso, 0.0, GRID, M_NP, cutoff=0.01,
        device="cpu", dtype=torch.float64,
    )
    idx = torch.zeros(n, dtype=torch.long)
    return splat_density(
        torch.tensor(frac), idx, torch.ones(n, dtype=torch.float64),
        A[idx], lam[idx], offsets, radius[idx], GRID,
        torch.tensor(M_NP), taper_w, compile_core=False,
    )


def cutoff_matching(density, target_occupancy):
    """The cutoff at which our mask calls the same FRACTION of the cell solvent
    as gemmi's does. Calibrating on occupancy is what makes the comparison
    about the shape of the boundary rather than about where each model happens
    to put it."""
    lo, hi = 1e-9, float(density.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        occ = float(mask_occupancy(solvent_mask(density, mid, 0.25 * mid)))
        if occ > target_occupancy:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@pytest.fixture(scope="module")
def fixture():
    cart = blob()
    g = gemmi_mask(cart)
    rho = our_density(cart)
    cut = cutoff_matching(rho, g.mean())
    ours = solvent_mask(rho, cut, 0.25 * cut)
    return dict(cart=cart, gemmi=g, density=rho, cutoff=cut, ours=ours)


def test_same_convention_one_is_solvent(fixture):
    """The mask that matters most to get right and is easiest to get backwards.
    Sampled at the atom centres, which must be 0 (molecule) in both."""
    idx = np.rint(fixture["cart"] @ np.linalg.inv(M_NP).T
                  * np.array(GRID)).astype(int) % np.array(GRID)
    at_atoms_gemmi = fixture["gemmi"][idx[:, 0], idx[:, 1], idx[:, 2]]
    ours_np = fixture["ours"].numpy()
    at_atoms_ours = ours_np[idx[:, 0], idx[:, 1], idx[:, 2]]
    assert at_atoms_gemmi.max() == 0.0
    assert at_atoms_ours.max() < 0.5
    # ...and far from the molecule both call it solvent.
    assert fixture["gemmi"][0, 0, 0] == 1.0
    assert ours_np[0, 0, 0] > 0.5


def test_agrees_with_gemmi_better_than_gemmi_agrees_with_itself(fixture):
    """The calibrated version of "close enough". Our threshold mask must sit
    within the spread of gemmi's own accepted radii conventions -- otherwise
    the model difference is larger than the disagreement the field already
    tolerates, and the substitution is not defensible."""
    ours_b = fixture["ours"].numpy() > 0.5
    cctbx_b = fixture["gemmi"] > 0.5
    ours_vs_cctbx = float((ours_b == cctbx_b).mean())

    others = {}
    for name in ("Refmac", "VanDerWaals"):
        other = gemmi_mask(fixture["cart"],
                           radii_set=getattr(gemmi.AtomicRadiiSet, name)) > 0.5
        others[name] = float((other == cctbx_b).mean())

    assert ours_vs_cctbx > 0.98, ours_vs_cctbx
    assert ours_vs_cctbx >= max(others.values()), (ours_vs_cctbx, others)


def test_f_mask_agrees_where_solvent_contributes(fixture):
    """What actually reaches the structure factors. Low-resolution reflections
    only, which is where a bulk solvent term contributes at all."""
    hkl = torch.tensor(
        [[h, k, l] for h in range(-4, 5) for k in range(-4, 5)
         for l in range(-4, 5) if (h, k, l) != (0, 0, 0)],
        dtype=torch.long,
    )
    orth = torch.tensor(M_NP)
    F_ours = f_solvent(fixture["ours"], VOLUME, hkl, orth)
    F_gemmi = f_solvent(
        torch.tensor(fixture["gemmi"], dtype=torch.float64), VOLUME, hkl, orth)

    a, b = np.abs(F_ours.numpy()), np.abs(F_gemmi.numpy())
    cc = float(np.corrcoef(a, b)[0, 1])
    r = float(np.abs(a - b).sum() / (a.sum() + b.sum()))
    assert cc > 0.99, cc
    assert r < 0.10, r


def test_the_matched_cutoff_is_not_transferable_across_b(fixture):
    """The documented limitation, asserted rather than described: a geometric
    mask does not move with B and a density threshold does, so the cutoff that
    reproduces gemmi at one B does not reproduce it at another. Borrowing it
    costs real agreement -- which is why the cutoff is a per-configuration
    calibration, not a constant to hard-code."""
    cart = fixture["cart"]
    g60 = gemmi_mask(cart, b_iso=60.0)
    rho60 = our_density(cart, b_iso=60.0)

    cut20 = fixture["cutoff"]
    cut60 = cutoff_matching(rho60, g60.mean())
    # 21x on this fixture; thousands-fold on a larger, denser one. The floor is
    # deliberately loose because the ratio is a property of the system, not a
    # constant -- what is constant is that it is nowhere near 1.
    assert cut60 > 10 * cut20, (cut20, cut60)

    def agreement(cut):
        m = solvent_mask(rho60, cut, 0.25 * cut).numpy() > 0.5
        return float((m == (g60 > 0.5)).mean())

    assert agreement(cut60) > agreement(cut20) + 0.01, (
        agreement(cut60), agreement(cut20))


def test_matching_gemmi_forces_a_nearly_hard_threshold(fixture):
    """A real constraint on the model, not a defect in this test.

    Matching gemmi's boundary puts the cutoff far out in the density tail,
    where the taper -- whose width cannot exceed the cutoff, or the fully
    solvent region disappears -- spans a sub-voxel distance. The mask is then
    smooth in principle and a hard threshold in practice, which is exactly what
    the gradients and the diffuse variance care about.

    Measured on this fixture: the matched cutoff is 1.45e-3 of peak density and
    leaves 31 shell voxels out of 110,592. Raising it to 1% of peak gives 599
    shell voxels -- 19x more boundary to differentiate through -- while
    agreement with gemmi goes 0.9951 to 0.9954, i.e. does not degrade at all.
    docs/solvent-design.md carries the full curve and the point where it does.
    """
    total = fixture["density"].numel()
    at_matched = shell_voxels(fixture["ours"])
    assert at_matched < 0.001 * total, (at_matched, total)

    cut = 0.01 * float(fixture["density"].max())
    wider = solvent_mask(fixture["density"], cut, 0.5 * cut)
    assert shell_voxels(wider) > 10 * max(at_matched, 1)
    agree = float(((wider.numpy() > 0.5) == (fixture["gemmi"] > 0.5)).mean())
    assert agree > 0.98, agree


# --------------------------------------------------------------------------
# mask_blur, measured where it can actually be measured

SEVENFPV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "examples", "compare_gemmi", "7FPV.pdb")


@pytest.mark.skipif(not os.path.exists(SEVENFPV), reason="7FPV.pdb not present")
def test_mask_blur_moves_the_mask_toward_the_geometric_one():
    """mask_blur's reason for existing, on a real crystal.

    IT HAS TO BE A REAL CRYSTAL, and that is why this test carries the cost of
    reading a PDB rather than using the fixture at the top of this file. On
    every synthetic model tried, the blur does nothing or slightly hurts:

        200-carbon blob (the fixture above)   0.9951 -> 0.9935
        900-carbon dense blob                 0.9962 -> 0.9933
        7FPV, P2_1 2_1 2_1                    0.8636 -> 0.9391

    A blob has one smooth, convex outer boundary that is already about as
    close to a geometric mask as a threshold can get. A crystal has solvent
    CHANNELS -- a convoluted surface threading between symmetry copies, with
    crevices between neighbouring atoms at every scale -- and that is the
    structure the probe radius exists to smooth away. This is the fourth
    conclusion in this model that a sparse fixture got wrong; see
    docs/solvent-design.md.

    Run at d_min 2.0 rather than the structure's 1.04 A because the effect is
    grid-independent (gain +0.064 here against +0.075 at 1.04 A) and the
    coarser grid costs 0.2 s instead of 1.5 s.
    """
    cctbx_pdb = pytest.importorskip("iotbx.pdb", reason="cctbx not installed")
    from lunus.sf.cell_utils import grid_shape_for_resolution
    from lunus.sf.kernel_torch import build_atom_kernels_torch
    from lunus.sf.solvent_torch import SolventModel
    from lunus.sf.symmetry_torch import (
        adjust_grid_for_symmetry, build_grid_ops_from_cctbx, symmetrize_sum,
    )

    xrs = cctbx_pdb.hierarchy.input(
        file_name=SEVENFPV, sort_atoms=False).input.xray_structure_simple()
    xrs.convert_to_isotropic()
    cell = xrs.unit_cell().parameters()
    sg = xrs.space_group()
    grid = adjust_grid_for_symmetry(
        grid_shape_for_resolution(cell[0], cell[1], cell[2], 2.0, rate=1.5),
        [np.array(o.r().as_double()).reshape(3, 3) for o in sg],
        [np.array(o.t().as_double()) for o in sg])

    elements = [s.scattering_type for s in xrs.scatterers()]
    b = np.array(xrs.extract_u_iso_or_u_equiv()) * (8.0 * np.pi ** 2)
    occ = np.array(xrs.scatterers().extract_occupancies())
    frac = np.array(xrs.sites_frac()).reshape(-1, 3)
    M = orth_matrix(*cell)
    orth = torch.tensor(M, dtype=torch.float64)

    # build_atom_kernels_torch returns PER-ATOM arrays already. Indexing them
    # by element the way build_element_kernels_torch's output must be indexed
    # silently gives every atom the kernel of whichever atom happens to sit at
    # its element's index, and still produces a plausible-looking density.
    atom_A, atom_lam, offsets, atom_r, taper_w, e2i = build_atom_kernels_torch(
        elements, sorted(set(elements)), IT92_COEFFS, b, 0.0, grid, M,
        cutoff=0.01, device="cpu", dtype=torch.float64)
    density = splat_density(
        torch.tensor(frac), torch.tensor([e2i[e] for e in elements]),
        torch.tensor(occ), atom_A, atom_lam, offsets, atom_r, grid, orth,
        taper_w, compile_core=False)
    density = symmetrize_sum(density, build_grid_ops_from_cctbx(sg, grid))

    st = gemmi.read_structure(SEVENFPV)
    st.setup_entities()
    g = gemmi.FloatGrid()
    g.set_size(*grid)
    g.set_unit_cell(st.cell)
    g.spacegroup = gemmi.find_spacegroup_by_name(st.spacegroup_hm)
    gemmi.SolventMasker(gemmi.AtomicRadiiSet.Refmac).put_mask_on_float_grid(
        g, st[0])
    geometric = np.array(g, copy=False)

    def agreement(blur):
        model = SolventModel(cutoff=1.0, taper_width=0.5, mask_blur=blur,
                             check_occupancy=False)
        # Calibrated to gemmi's OWN occupancy, so this compares shape and not
        # how much of the cell each model calls solvent.
        model.calibrate(density, orth, float(geometric.mean()))
        mask = model.build_mask(density, orth).numpy()
        return float(((mask > 0.5) == (geometric > 0.5)).mean())

    bare, blurred = agreement(0.0), agreement(100.0)
    assert bare == pytest.approx(0.888, abs=0.02)
    assert blurred == pytest.approx(0.951, abs=0.02)
    assert blurred > bare + 0.04
