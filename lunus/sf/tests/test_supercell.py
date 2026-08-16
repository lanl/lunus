"""
Supercell folding: the model's own cell holding several of the cell being
calculated on.

Two separate claims:

  1. Folding is EXACT and changes nothing without solvent. Splatting atoms
     into the small grid already folds them, via the modulo in the scatter, so
     splatting onto the supercell grid and folding the density afterwards must
     give the same answer. If it does not, the two foldings disagree and every
     supercell result is suspect.

  2. Folding is the reason the solvent mask has to be built FIRST. A
     protein/solvent boundary exists in the model's own frame and does not
     survive being folded into a smaller cell -- measured on the compare_gemmi
     system, 34% solvent before and 1% after. The pipeline therefore builds the
     mask on the supercell grid and folds it afterwards, which is exact because
     folding is linear.

Plus the guard: a non-integer cell ratio has no exact grid folding, so
supercell_factors() refuses it rather than interpolating silently.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from lunus.sf.cell_utils import orth_matrix
from lunus.sf.density_torch import splat_density
from lunus.sf.elements import IT92_COEFFS
from lunus.sf.kernel_torch import build_element_kernels_torch
from lunus.sf.solvent_torch import SolventModel, mask_occupancy, solvent_mask
from lunus.sf.structure_factor_torch import structure_factors_one_config
from lunus.sf.symmetry_torch import fold_supercell, supercell_factors

CELL = (20.0, 22.0, 24.0, 90.0, 90.0, 90.0)     # the target cell
GRID = (20, 22, 24)
FACTORS = (2, 1, 3)
DTYPE = torch.float64


def _model(n_atoms=40, seed=0):
    """Atoms scattered through the SUPERCELL, expressed -- as the pipeline
    expects -- in fractional coordinates of the TARGET cell, so they run well
    outside [0, 1)."""
    rng = np.random.default_rng(seed)
    M_np = orth_matrix(*CELL)
    A, lam, offsets, radius, taper_w, _ = build_element_kernels_torch(
        ["C"], IT92_COEFFS, 20.0, 0.0, GRID, M_np, cutoff=0.01,
        device="cpu", dtype=DTYPE,
    )
    idx = torch.zeros(n_atoms, dtype=torch.long)
    frac_target = rng.random((n_atoms, 3)) * np.array(FACTORS)
    return dict(
        frac=torch.tensor(frac_target, dtype=DTYPE), element_idx=idx,
        occ=torch.ones(n_atoms, dtype=DTYPE), atom_A=A[idx], atom_lam=lam[idx],
        elem_offsets=offsets, atom_radius_ang=radius[idx], grid_shape=GRID,
        orth=torch.tensor(M_np, dtype=DTYPE),
        cell_volume=float(np.linalg.det(M_np)), taper_width=taper_w,
        hkl=torch.tensor(
            [[h, k, l] for h in range(-3, 4) for k in range(-3, 4)
             for l in range(-3, 4) if (h, k, l) != (0, 0, 0)],
            dtype=torch.long),
    )


def _sf(s, **kw):
    return structure_factors_one_config(
        s["frac"], s["element_idx"], s["occ"], s["atom_A"], s["atom_lam"],
        s["elem_offsets"], s["atom_radius_ang"], s["grid_shape"], s["orth"],
        s["cell_volume"], s["hkl"], s["taper_width"], compile_core=False, **kw)


# --------------------------------------------------------------------------
# the arithmetic

def test_factors_from_cells():
    target = (88.451, 88.451, 39.823, 90, 90, 90)
    model = (2 * 88.451, 2 * 88.451, 5 * 39.823, 90, 90, 90)
    assert supercell_factors(model, target) == (2, 2, 5)


def test_a_non_integer_ratio_is_refused():
    """The compare_gemmi case: 68.392 against a = 88.451 is not a supercell of
    anything, and there is no exact folding for it."""
    with pytest.raises(ValueError, match="not an integer supercell"):
        supercell_factors((68.392, 91.116, 198.088, 90, 90, 90),
                          (88.451, 88.451, 39.823, 90, 90, 90))


def test_mismatched_angles_are_refused():
    with pytest.raises(ValueError, match="angles"):
        supercell_factors((20.0, 20.0, 20.0, 90, 90, 120),
                          (10.0, 10.0, 10.0, 90, 90, 90))


def test_folding_conserves_and_is_linear():
    g = torch.arange(4 * 3 * 6, dtype=DTYPE).reshape(4, 3, 6)
    folded = fold_supercell(g, (2, 1, 3))
    assert folded.shape == (2, 3, 2)
    assert torch.equal(folded.sum(), g.sum())              # nothing lost

    h = torch.randn(4, 3, 6, dtype=DTYPE)
    assert torch.allclose(fold_supercell(g + h, (2, 1, 3)),
                          folded + fold_supercell(h, (2, 1, 3)))


# --------------------------------------------------------------------------
# 1. exactness against the folding the splat already does

def test_supercell_without_solvent_changes_nothing():
    """Splatting into the small grid folds the atoms by modulo; splatting into
    the supercell grid and folding the density does it afterwards. Same
    answer, or the two paths disagree about what folding means."""
    s = _model()
    direct = _sf(s, solvent=None)
    viasuper = _sf(s, solvent=None, supercell=FACTORS)
    rel = ((direct - viasuper).abs() / direct.abs()).max()
    assert float(rel) < 1e-12, float(rel)


def test_supercell_rejects_symmetry_expansion():
    """A model spanning several cells already contains the symmetry copies, so
    expanding again double-counts -- and a summed mask is not a mask."""
    s = _model()
    fake_ops = [object()]
    with pytest.raises(ValueError, match="cannot be combined"):
        _sf(s, solvent=None, supercell=FACTORS, grid_ops=fake_ops)


# --------------------------------------------------------------------------
# 2. why the order matters at all

def test_folding_destroys_the_boundary_that_the_mask_needs():
    """The measurement behind the design: a mask thresholded AFTER folding
    describes a cell with no empty space left in it, while the same model
    masked in its own frame has a real solvent region."""
    s = _model(n_atoms=60)
    n = FACTORS
    scale = torch.tensor(n, dtype=DTYPE)
    big_grid = tuple(g * f for g, f in zip(GRID, n))

    rho_super = splat_density(
        s["frac"] / scale, s["element_idx"], s["occ"], s["atom_A"],
        s["atom_lam"], s["elem_offsets"], s["atom_radius_ang"], big_grid,
        s["orth"] * scale, s["taper_width"], compile_core=False)
    rho_folded = fold_supercell(rho_super, n)

    cut_s = 0.01 * float(rho_super.max())
    cut_f = 0.01 * float(rho_folded.max())
    occ_super = float(mask_occupancy(solvent_mask(rho_super, cut_s, 0.5 * cut_s)))
    occ_folded = float(mask_occupancy(solvent_mask(rho_folded, cut_f, 0.5 * cut_f)))

    assert occ_super > 0.5, occ_super            # a real solvent region
    assert occ_folded < occ_super - 0.1, (occ_super, occ_folded)


def test_the_pipeline_masks_before_folding():
    """End to end: with supercell=, F_total must match the mask built on the
    supercell grid and folded -- not one built from the folded density."""
    s = _model(n_atoms=60)
    n = FACTORS
    scale = torch.tensor(n, dtype=DTYPE)
    big_grid = tuple(g * f for g, f in zip(GRID, n))

    rho_super = splat_density(
        s["frac"] / scale, s["element_idx"], s["occ"], s["atom_A"],
        s["atom_lam"], s["elem_offsets"], s["atom_radius_ang"], big_grid,
        s["orth"] * scale, s["taper_width"], compile_core=False)
    cutoff = 0.01 * float(rho_super.max())
    model = SolventModel(cutoff=cutoff, taper_width=0.5 * cutoff,
                         check_occupancy=False)

    from lunus.sf.solvent_torch import f_solvent, f_total
    from lunus.sf.structure_factor_torch import compute_fcalc, reciprocal_inv_d2

    mask_folded = fold_supercell(
        solvent_mask(rho_super, cutoff, 0.5 * cutoff), n)
    F_protein = compute_fcalc(
        fold_supercell(rho_super, n), s["cell_volume"], s["hkl"], s["orth"])
    expected = f_total(
        F_protein, f_solvent(mask_folded, s["cell_volume"], s["hkl"], s["orth"]),
        reciprocal_inv_d2(s["orth"], s["hkl"]),
        k_sol=model.k_sol, b_sol=model.b_sol, hkl=s["hkl"])

    got = _sf(s, solvent=model, supercell=n)
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)

    # And it is genuinely a different answer from masking after folding.
    naive = _sf(s, solvent=model)
    assert not torch.allclose(got, naive, rtol=1e-3, atol=1e-6)
